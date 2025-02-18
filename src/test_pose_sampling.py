from UV_Volumes.lib.networks.renderer.uv_volumes import Renderer
from UV_Volumes.lib.networks import nts
from UV_Volumes.lib.config import def_cfg
from UV_Volumes.lib.config import yacs
import torch.nn.functional as F
from UV_Volumes.lib.utils.net_utils import load_network
from pose_sampling import PoseSampler
import torch
import os
import cv2
device = torch.device("cuda:0")
import numpy as np
# Configure logging


def to_cuda(batch, device):
    for k in batch:
        if k == 'meta' or k == 'img_name' or  isinstance(batch[k], str) :
            continue
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to(device) for b in batch[k]]
        else:
            batch[k] = batch[k].to(device)
    return batch

def create_uv_cfg(device, uvcfg_file_name):
    uv_cfg = def_cfg()
    with open(uvcfg_file_name, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        uv_cfg.merge_from_other_cfg(parent_cfg)

    uv_cfg.merge_from_other_cfg(current_cfg)
    uv_cfg.trained_model_dir = os.path.join(uv_cfg.trained_model_dir, uv_cfg.task, uv_cfg.exp_name)
    uv_cfg.record_dir = os.path.join(uv_cfg.record_dir, uv_cfg.task, uv_cfg.exp_name)
    uv_cfg.result_dir = os.path.join(uv_cfg.result_dir, uv_cfg.task, uv_cfg.exp_name)
    uv_cfg.cfg_dir = os.path.join(uv_cfg.cfg_dir, uv_cfg.task, uv_cfg.exp_name)
    uv_cfg.device = [int(device[-1])]
    return uv_cfg

class Test_Pose_sampler():
    def __init__(self,batch_size=1):
        self.uvvolume_network = nts.Network().to(device).eval()
        self.uvvolume_renderer = Renderer(self.uvvolume_network)
        self.pose_sampler = PoseSampler()
        self.device = device
        self.data_loader = torch.utils.data.DataLoader(self.pose_sampler,
                                              batch_size = batch_size,
                                              num_workers=4)
        self.uv_cfg = create_uv_cfg("cuda:0", "./UV_Volumes/configs/zju_mocap_exp/377.yaml")
        load_network(self.uvvolume_network,
                            self.uv_cfg.trained_model_dir,
                            resume=self.uv_cfg.resume,
                            epoch=self.uv_cfg.test.epoch)
        # image_bgr = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/result/UVvolume_wild/Peter_chess/comparison/texture_static_frame0013_epoch0599.png')
        image_bgr = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/result/UVvolume_ZJU/zju377/comparison/texture_static_frame0000_epoch0400-2.png')
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.tex_size = 256
        image_rgb = cv2.resize(image_rgb, (self.tex_size * 6, self.tex_size * 4))
        TextureIm_pose = np.array(image_rgb)
        self.Texture_pose = torch.zeros((24, 3, self.tex_size, self.tex_size)).to(self.device)
        for i in range(len(self.Texture_pose)):
            x = i // 6 * self.tex_size
            y = i % 6 * self.tex_size
            self.Texture_pose[i]  = torch.from_numpy(TextureIm_pose[x:x + self.tex_size, y:y + self.tex_size]).float().permute(2,0,1)/255  
    def generate_images(self, ):
        for idx, action_batch in enumerate(self.data_loader):
            if idx > 50:
                break
            action_batch = to_cuda(action_batch, device)
            action_batch["epoch"] = 0
            with torch.no_grad():
                output = self.uvvolume_renderer.render(action_batch)
            # get the iuv data, i is the index of human parts, uv is the predicted uv coordinates, see densepsoe,
            iuv_map = output["iuv_body"].to(device)
            i_map = F.softmax(iuv_map[..., :24], dim=-1) #mask, 24
            u_map = iuv_map[..., 24:48] # mask, 24
            v_map = iuv_map[..., 48:]   # mask, 24
            uv_map = torch.stack((u_map, v_map), -1)  # mask, 24, 2
            grid = (2*uv_map.permute(1,0,2).unsqueeze(1)-1)
            texture_gridsample = torch.nn.functional.grid_sample(self.Texture_pose,
                                grid, 
                                mode='bilinear', align_corners=False) 
            # outputsize is (mask, 3)
            rgb_pred = (i_map.permute(1,0).unsqueeze(2) * texture_gridsample.permute(0,2,3,1).view(24,-1,3)).sum(0)
            rgb_gt = action_batch['rgb']
            rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
            rgb_padding[output['T_last'] < self.uv_cfg.T_threshold] = rgb_pred
            # crop rgb pred at box
            mask_at_box = action_batch['mask_at_box'][0]
            H, W = action_batch['H'][0], action_batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            rgb_pred_crop[mask_at_box] = rgb_padding[0]
            rgb_pred_crop = rgb_pred_crop.permute(2,0,1).unsqueeze(0).repeat(1, 1, 1, 1)
            x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
            adv_batch = rgb_pred_crop[:, :, y:y + h, x:x + w]
            cv2.imwrite(f"results/test_uv_pose_sampling_{idx}.png", rgb_pred_crop[0].permute(1,2,0).detach().cpu().numpy()[...,[2,1,0]]*255)
            PoseSampler.visualize_vertices(action_batch["xyz"][0,::,:,:].detach().cpu().numpy(),f"results/recover_{idx}.png")
            
            mask_at_dp, mask_at_body, mask_at_bg = \
                action_batch['mask_at_dp'], action_batch['mask_at_body'], action_batch['mask_at_bg']
                
            i_map = i_map.detach().cpu().numpy()
            max_indices = np.argmax(i_map, axis=1)
            uv_map = uv_map.detach().cpu().numpy()
            # Step 2: Use these indices to select the corresponding elements from the UV map
            uv_result = np.zeros((uv_map.shape[0], 3))
            uv_result[:,0:2] = uv_map[np.arange(uv_map.shape[0]), max_indices] # mask 3
            uv_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.) #1,77815,3
            uv_padding[output['T_last'] < self.uv_cfg.T_threshold] = torch.tensor(uv_result).float().to(device)
            mask_at_box = action_batch['mask_at_box'][0]
            H, W = action_batch['H'][0], action_batch['W'][0]
            mask_at_box = mask_at_box.reshape(H, W)
            uv_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
            uv_pred_crop[mask_at_box] = uv_padding[0]
            cv2.imwrite(f"results/uvmap_{idx}.png", uv_pred_crop.detach().cpu().numpy()[...,[2,1,0]]*255)
            
if __name__ == "__main__":
    model_path = '/home/yjli/AIGC/Adversarial_camou/SPIN/data/SMPL_NEUTRAL.pkl'
    # model_path = "/home/yjli/AIGC/humannerf/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    pose_sampler = PoseSampler()
    poses = pose_sampler.sample_poses() # numpy
    shapes = pose_sampler.sample_shapes(same=True) #numpy
    pose_sampler.visualize_GMM()
    np.savetxt('poses.txt', pose_sampler.poses)

    # pose_rot = pose_sampler.pose2rot_batch(poses) #numpy
    # vertices = pose_sampler.generate_vertices(pose_rot, shapes) # tensor
    # vertices = vertices.detach().cpu().numpy()
    # PoseSampler.visualize_vertices_3D(vertices)
    
    test_sampler =  Test_Pose_sampler()
    test_sampler.generate_images()