import time
import torch
import tqdm
from lib.config import cfg
import os
import imageio
import cv2
import numpy as np
from lib.networks import embedder
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, ep_tqdm):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)

            batch['epoch'] = epoch
            output, loss, loss_stats, image_stats = self.network(batch, is_train=True)

            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40) 
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            lr = {'lr2': optimizer.param_groups[0]['lr'],
                  'lr1': optimizer.param_groups[-1]['lr']}
            recorder.update_lr_stats(lr)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        latent_index_record = []
        index = 0

        for batch in tqdm.tqdm(data_loader):
            index = index + 1
            batch = self.to_cuda(batch)
            batch['epoch'] = epoch
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch, is_train=False)
                if evaluator is not None:
                    evaluator.evaluate(output, batch, epoch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

            # texture map
            if batch['latent_index'] in latent_index_record:
                continue

            # with torch.no_grad():
            # tex_size = cfg.texture_size
            tex_size = 256
            num_pixel = 24 * tex_size * tex_size
            chunk = 24 * 65536
            i_onehot = torch.eye(24)[torch.arange(24)].cuda().unsqueeze(1).expand(-1,tex_size*tex_size,-1) #24,256*256,24
            umap, vmap = torch.meshgrid(torch.linspace(0,1,tex_size).to(self.device), 
                                            torch.linspace(0,1,tex_size).to(self.device))
            uv_stack = torch.stack((umap, vmap), 2).view(-1,1,2)   # 256*256,1,2

            uv_encoding = embedder.uv_embedder(uv_stack.view(1,-1,2)) # 1, 256*256, 42
            uv_encoding = uv_encoding.expand(24,-1,-1)  #  24, 256*256, 42
            iuv_encoding = torch.cat((i_onehot, uv_encoding), -1)  # 24, mask, 24 + 42
            iuv_encoding = iuv_encoding.view(-1, iuv_encoding.shape[-1]) # 24 * mask, 24+42
            expand_view = torch.Tensor([1,0,0]).to(self.device)[None,None].expand(24,tex_size*tex_size,-1).view(-1, 3)
            viewdirs_encoding = embedder.view_embedder(expand_view)   # 24 * mask, 27
            rgb_pred_list  = []
            
            for i in range(0, num_pixel, chunk):
                rgb_pred_chunk = self.network.net.implicit_tex_model.get_rgb(iuv_encoding[i:i+chunk], batch['poses'], viewdirs_encoding[i:i+chunk])
                rgb_pred_list.append(rgb_pred_chunk)
            
            rgb_pred = torch.cat(rgb_pred_list, dim=0)
            Texture_pose = rgb_pred.view(24,tex_size,tex_size,3).permute(0,2,1,3) # rotate the texture
            # Read the image in BGR format
            image_bgr = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV-Volumes/data/result/UVvolume_ZJU/zju377/comparison/texture_static_frame0000_epoch0400-2.png')
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (tex_size * 6, tex_size * 4))
            TextureIm_pose = np.array(image_rgb)
            # TextureIm_pose = np.zeros((tex_size * 4, tex_size * 6, 3), dtype=np.uint8)
            for i in range(len(Texture_pose)):
                x = i // 6 * tex_size
                y = i % 6 * tex_size
                Texture_pose[i]  = torch.from_numpy(TextureIm_pose[x:x + tex_size, y:y + tex_size]).float()/255
            
            #  Texture_pose = torch.from_numpy(Texture_pose).float()/255
            # directly sample from UV texture
            iuv_map = output["iuv_body"]
            i_map = F.softmax(iuv_map[..., :24], dim=-1) #14945, 24
            u_map = iuv_map[..., 24:48] # 14945, 24
            v_map = iuv_map[..., 48:]   # 14945, 24
            i_onehot = torch.eye(24)[torch.arange(24)].cuda().unsqueeze(1).expand(-1,i_map.shape[0],-1).detach()  #24,mask,24
            uv_map = torch.stack((u_map, v_map), -1)  # mask, 24, 2
            uv_encoding = embedder.uv_embedder(uv_map.view(1,-1,2)) # 1, mask*24, 42
            uv_encoding = uv_encoding.view(-1,24,42).transpose(1,0)  #  24, mask, 42
            iuv_encoding = torch.cat((i_onehot, uv_encoding), -1)  # 24, mask, 24 + 42
            iuv_encoding = iuv_encoding.view(-1, iuv_encoding.shape[-1]) # 24 * mask, 24+42
            
            iuv_view = iuv_encoding.view(24, -1, 24 + 42)
            uv = iuv_view[...,24:26]  # 24, mask, 2
            
            # the shape of uv is [24, mask, 2], the output shape is [24, 3, 1, mask]
            # uv_map and uv are equal
            # grid = (2*uv_map.permute(1,0,2).unsqueeze(1)-1)
            grid = (2*uv.unsqueeze(1)-1)
            texture_gridsample = torch.nn.functional.grid_sample(Texture_pose.permute(0,3,1,2),
                                grid, 
                                mode='bilinear', align_corners=False) 
            # outputsize is (mask, 3)
            rgb_pred = (i_map.permute(1,0).unsqueeze(2) * texture_gridsample.permute(0,2,3,1).view(24,-1,3)).sum(0)
            rgb_gt = batch['rgb']
            rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
            rgb_padding[output['T_last'] < cfg.T_threshold] = rgb_pred
            # crop rgb pred at box
            mask_at_box = batch['mask_at_box'][0]
            H, W = batch['H'][0], batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_pred_crop[mask_at_box] = rgb_padding[0]
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            rgb_pred_crop = rgb_pred_crop[y:y + h, x:x + w]
            cv2.imwrite(f'sample_{index}.png', rgb_pred_crop.cpu().detach().numpy()[..., [2, 1, 0]]*255)  # RGB to BGR
            
            # generate the sample image 
            masksize = iuv_view.shape[1]
            expand_view = torch.Tensor([1,0,0]).to(self.device)[None,None].expand(24, masksize,-1).view(-1, 3)
            viewdirs_encoding = embedder.view_embedder(expand_view)   # top_k * mask, 27
            rgb_pred = self.network.net.implicit_tex_model.get_rgb(iuv_encoding, batch['poses'], viewdirs_encoding)
            rgb_pred = (i_map.permute(1,0).unsqueeze(2) * rgb_pred.view(24,-1,3)).sum(0)
            rgb_gt = batch['rgb']
            rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
            rgb_padding[output['T_last'] < cfg.T_threshold] = rgb_pred
            # crop rgb pred at box
            mask_at_box = batch['mask_at_box'][0]
            H, W = batch['H'][0], batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_pred_crop[mask_at_box] = rgb_padding[0]
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            rgb_pred_crop = rgb_pred_crop[y:y + h, x:x + w]
            cv2.imwrite('sample2.png', rgb_pred_crop.cpu().detach().numpy()[..., [2, 1, 0]]*255) # RGB to BGR
            
            rgb_padding= output["rgb_map"]
            # crop rgb pred at box
            mask_at_box = batch['mask_at_box'][0]
            H, W = batch['H'][0], batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_pred_crop[mask_at_box] = rgb_padding[0]
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            rgb_pred_crop = rgb_pred_crop[y:y + h, x:x + w]
            cv2.imwrite('sample3.png', rgb_pred_crop.cpu().detach().numpy()[..., [2, 1, 0]]*255)
            
            Texture_pose = Texture_pose.cpu().detach().numpy() * 255
            TextureIm_pose = np.zeros((tex_size * 4, tex_size * 6, 3), dtype=np.uint8)
            for i in range(len(Texture_pose)):
                x = i // 6 * tex_size
                y = i % 6 * tex_size
                TextureIm_pose[x:x + tex_size, y:y + tex_size] = Texture_pose[i]     

            result_dir = os.path.join(cfg.result_dir, 'comparison')
            os.system('mkdir -p {}'.format(result_dir))
            frame_index = batch['frame_index'].item()
            cv2.imwrite(
                '{}/texture_static_frame{:04d}_epoch{:04d}.png'.format(result_dir, frame_index, epoch), 
                TextureIm_pose[..., [2, 1, 0]])
            
            print('{}/texture_static_frame{:04d}_epoch{:04d}.png'.format(result_dir, frame_index, epoch))

            # view rgb
            '''
            rgbs = []
            alpha = np.linspace(-np.pi, np.pi, 6)
            beta = np.linspace(-np.pi, np.pi, 6)
            for a in alpha:
                for b in beta:
                    x = np.cos(a) * np.cos(b)
                    z = np.sin(a) * np.cos(b)
                    y = np.sin(b)
                    viewdir = torch.Tensor([x,y,z])[None,None].expand(24, tex_size*tex_size, -1).to(self.device) #24, 256*256, 3

                    viewdirs_encoding = embedder.view_embedder(viewdir.view(-1, 3))
                    rgb_pred = self.network.net.implicit_tex_model.get_rgb(iuv_encoding, batch['poses'], viewdirs_encoding)
                    rgb_dynamic = rgb_pred.view(24,tex_size,tex_size,3)

                    Texture = rgb_dynamic.cpu().numpy() * 255   #24, 256, 256, 3
                    TextureIm = np.zeros((tex_size * 4, tex_size * 6, 3), dtype=np.uint8)
                    for i in range(len(Texture)):
                        x = i // 6 * tex_size
                        y = i % 6 * tex_size
                        TextureIm[x:x + tex_size, y:y + tex_size] = Texture[i]

                    rgbs.append(TextureIm)

            imageio.mimsave(
                '{}/texture_dynamic_frame{:04d}_epoch{:04d}.gif'.format(result_dir, frame_index, epoch), rgbs, fps=5)
            '''
            latent_index_record.append(batch['latent_index'])


        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
