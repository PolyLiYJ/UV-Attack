import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import uv_volumes
import torch.nn.functional as F
import math
import numpy as np
import cv2
from lib.networks.perceptual_loss import Perceptual_loss
from lib.networks import embedder

def cv2tensor(imgpath, size):
    cloth1 = cv2.imread(imgpath) # chess
    cloth1 = cv2.resize(cloth1, (size, size))
    cloth1 = cv2.cvtColor(cloth1, cv2.COLOR_BGR2RGB) 
    cloth1 = torch.tensor(np.array(cloth1)/255)
    return cloth1


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = uv_volumes.Renderer(self.net)

        self.mse = lambda x, y : torch.mean((x - y) ** 2)
        self.entroy= torch.nn.CrossEntropyLoss()
        self.iLoss_weight = ExponentialAnnealingWeight(cfg.iLossMax, cfg.iLossMin, cfg.exp_k)
        self.uvLoss_weight = ExponentialAnnealingWeight(cfg.uvLossMax, cfg.uvLossMin, cfg.exp_k)
        self.vgg_loss = Perceptual_loss()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        
        self.tex_size =  128
        
        self.cloth_index = [0,1]
        self.cloth_num = len(self.cloth_index)
        
        cloth1 = cv2tensor("/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth.png", self.tex_size)
        self.cloth1 = cloth1.requires_grad_(False).to(self.device) # H,W,3
        
        cloth2 = cv2tensor("/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png", self.tex_size)
        self.cloth2 = cloth2.requires_grad_(False).to(self.device) # H,W,3
        tex_size = self.tex_size
        i_onehot = torch.eye(24)[torch.arange(24)].cuda().unsqueeze(1).expand(-1,tex_size*tex_size,-1)[self.cloth_index,:,:].requires_grad_(False) #2,256*256,24
        umap, vmap = torch.meshgrid(torch.linspace(0,1,tex_size).to(self.device), 
                                            torch.linspace(0,1,tex_size).to(self.device))

        # uv_map = torch.stack((umap, vmap), 2).view(-1, 2) 
        # uv_encoding = uv_map.expand(self.cloth_num, -1,-1)  #  2, 256*256, 2
        
        uv_stack = torch.stack((umap, vmap), 2).view(-1,1,2)   # 256*256,1,2
        uv_encoding = embedder.uv_embedder(uv_stack.view(1,-1,2)).expand(self.cloth_num,-1,-1)  #  2, 256*256, 42
        self.iuv_encoding = torch.cat((i_onehot, uv_encoding), -1).requires_grad_(False)  # 2, 256*256, 24 + 42
        expand_view = torch.Tensor([1,0,0]).to(self.device)[None,None].expand(self.cloth_num, tex_size*tex_size,-1).view(-1, 3)
        self.viewdirs_encoding = embedder.view_embedder(expand_view)   # top_k * mask, 27


    def forward(self, batch, is_train=True):
        batch['is_train'] = is_train
        ret = self.renderer.render(batch)
        epoch = batch['epoch']
        mask_at_dp, mask_at_body, mask_at_bg = \
          batch['mask_at_dp'], batch['mask_at_body'], batch['mask_at_bg']

        rgb_pred, delta_rgb_pred = ret['rgb_map'][..., :3], ret['rgb_map'][..., 3:]
        rgb_gt = batch['rgb']
        i_map, uv_map = ret['iuv_map'][..., :24], ret['iuv_map'][..., 24:]
        i_gt, uv_gt = batch['iuv'][..., :24], batch['iuv'][..., 24:]
        scalar_stats = {}
        loss = 0

        # iLoss_weight = self.iLoss_weight.getWeight(epoch)
        iLoss_weight = cfg.iLoss_weight
        # uvLoss_weight = self.uvLoss_weight.getWeight(epoch)
        uvLoss_weight = cfg.uvLoss_weight

        if is_train:
          rgb_loss = self.mse(rgb_pred[0] + delta_rgb_pred[0], rgb_gt[mask_at_body])
          i_at_dp_loss = self.entroy(i_map[0], i_gt.max(-1)[1][mask_at_dp]) \
                          * iLoss_weight
          uv_at_dp_loss = self.mse(uv_map[0], uv_gt[mask_at_dp]) \
                          * uvLoss_weight

        else:
          rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
          rgb_padding[ret['T_last'] < cfg.T_threshold] = rgb_pred + delta_rgb_pred
          ret['rgb_map'] = rgb_padding
          rgb_loss = self.mse(rgb_padding, rgb_gt)

          i_at_dp_loss = self.entroy(i_map[mask_at_dp], i_gt.max(-1)[1][mask_at_dp]) \
                          * iLoss_weight
          uv_at_dp_loss = self.mse(uv_map[mask_at_dp], uv_gt[mask_at_dp]) \
                          * uvLoss_weight

        scalar_stats.update({'rgb_loss': rgb_loss, 
                              'i_at_dp_loss': i_at_dp_loss,
                              'uv_at_dp_loss': uv_at_dp_loss})

        loss += i_at_dp_loss + uv_at_dp_loss + rgb_loss

        if cfg.use_TL2Loss :
            TL2Loss_weight = cfg.TLoss_weight
            TL2_loss = 0
            if mask_at_bg.sum() != 0:
                TL2_loss += torch.mean((1. - ret['T_last'][mask_at_bg]) ** 2) * TL2Loss_weight
            if (mask_at_body).sum() != 0:
                TL2_loss += torch.mean((ret['T_last'][mask_at_body]) ** 2) * TL2Loss_weight

            scalar_stats['TL2_loss'] = TL2_loss
            loss += TL2_loss
 
        if cfg.use_vggLoss:
            mask_at_box = batch['mask_at_box'][0]
            H, W = batch['H'][0], batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            sh = mask_at_box.sum()
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            
            # crop rgb gt
            rgb_gt_box = torch.cuda.FloatTensor(sh, 3).fill_(0.)
            rgb_gt_box[mask_at_body[0]] = rgb_gt[mask_at_body]
            rgb_gt_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_gt_crop[mask_at_box] = rgb_gt_box

            rgb_gt_crop = rgb_gt_crop[y:y + h, x:x + w]
            rgb_gt_crop = rgb_gt_crop.permute(2,0,1)[None].detach()

            if is_train:
                # crop rgb pred at body
                rgb_padding_box = torch.cuda.FloatTensor(sh, 3).fill_(0.).detach()
                rgb_padding_box[mask_at_body[0]] = rgb_pred[0] + delta_rgb_pred[0]
                rgb_padding = torch.cuda.FloatTensor(H, W, 3).fill_(0.).detach()
                rgb_padding[mask_at_box] = rgb_padding_box
                rgb_pred_crop = rgb_padding[y:y + h, x:x + w]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1)[None]
            else:
                # crop rgb pred at box
                rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
                rgb_pred_crop[mask_at_box] = rgb_padding[0]
                rgb_pred_crop = rgb_pred_crop[y:y + h, x:x + w]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1)[None]

            # img = rgb_pred_crop[0].permute(1,2,0).detach().cpu().numpy()*255
            # cv2.imwrite("pred.png", img[...,[2,1,0]])
            # img = rgb_gt_crop[0].permute(1,2,0).detach().cpu().numpy()*255
            # cv2.imwrite("gt.png", img[...,[2,1,0]])
            vgg_loss = self.vgg_loss(rgb_pred_crop*2-1, rgb_gt_crop*2-1).squeeze() * cfg.vggLoss_weight
            scalar_stats.update({'vgg_loss': vgg_loss})                      
            loss += vgg_loss
        
        if cfg.use_clothLoss:
            '''
            tex_size = self.tex_size
            i_onehot =  torch.eye(24)[torch.arange(24)][self.cloth_index,:].cuda().detach()
            # i_onehot = i_onehot[:,0,:]
            latent_theta = self.net.implicit_tex_model.pose2latent(torch.cat((i_onehot, batch['poses'].expand(self.cloth_num,-1)), -1))   #24, 24+128 -> 24, 512 * 32 * 32
            decode_latent = self.net.implicit_tex_model.latent_decoder(latent_theta.view(-1, 512, 2, 2)) # 2, 512, 2, 2 -> 2, 128, 64, 64
            # the shape of uv is [24, mask, 2]
            decode_latent_gridsample = nn.functional.grid_sample(decode_latent[self.cloth_index,:,:],
                                (2*self.iuv_encoding[:,:,24:26].unsqueeze(1)-1), 
                                mode='bilinear', align_corners=False) # 2, 1, mask, 2 - > 2, 128, 1, mask
            hyper = (decode_latent_gridsample.squeeze(2).transpose(2,1)).contiguous().view(-1, 128)   # 2 * mask, 128

            iuv_encoding = self.iuv_encoding.view((-1, self.iuv_encoding.shape[-1]))
            
            feature = self.net.implicit_tex_model.rgb_mapping_1(torch.cat((iuv_encoding.detach().requires_grad_(False), hyper), -1))
            feature = self.net.implicit_tex_model.rgb_mapping_2(torch.cat((iuv_encoding.detach().requires_grad_(False), hyper, feature), -1))
            Texture_pose = self.net.implicit_tex_model.rgb_mapping_3(torch.cat((self.viewdirs_encoding.detach().requires_grad_(False), feature), -1))

            Texture_pose = torch.sigmoid(Texture_pose).view(self.cloth_num,self.tex_size,self.tex_size,3)
            '''
            # # this will OOM
            # Texture_pose = self.net.implicit_tex_model.get_rgb(self.iuv_encoding, batch['poses'], self.viewdirs_encoding)
            # # exchange the H and W, other with grid sampling from the Texture_pose will have false
            # Texture_pose = Texture_pose.view(24,self.tex_size,self.tex_size,3)
            cloth_loss = cfg.clothLoss_weight * (torch.norm((ret['Texture_pose'][0] - self.cloth2),2) + torch.norm((ret['Texture_pose'][1] - self.cloth1),2))
            scalar_stats['cloth_loss'] = cloth_loss
            loss =  cloth_loss

        scalar_stats.update({'loss': loss})

        scalar_stats.update({'iLoss_weight': torch.tensor(iLoss_weight), 
                            'uvLoss_weight': torch.tensor(uvLoss_weight)})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


class CosineAnnealingWeight():
    def __init__(self, max, min, Tmax):
        super().__init__()
        self.max = max
        self.min = min
        self.Tmax = Tmax

    def getWeight(self, Tcur):
        return self.min + (self.max - self.min) * (1 + math.cos(math.pi * Tcur / self.Tmax)) / 2


class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))