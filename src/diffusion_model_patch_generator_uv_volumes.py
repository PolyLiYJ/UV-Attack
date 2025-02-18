import torch
import os
import sys
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.abspath(''))
from stable_diffusion_generator_small import StableDiffusionGenerator
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
import lpips
import cv2
import time
from datetime import datetime
import argparse
import numpy as np
import scipy
import scipy.interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt
from generator import *
from load_data import *
from transformers import DeformableDetrForObjectDetection
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
import os
from diffusers.utils.torch_utils import randn_tensor
import pickle
from UV_Volumes.lib.networks.renderer.uv_volumes import Renderer
from UV_Volumes.lib.config import yacs
from UV_Volumes.lib.config.config import cfg
from UV_Volumes.lib.networks import nts
from UV_Volumes.lib.datasets import make_data_loader
from UV_Volumes.lib.utils.net_utils import  load_network
from UV_Volumes.TPS_tradition import warp_image_cv
# import ultralytics
from ultralytics import YOLO
from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from train_util import *
import logging
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
)

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class PatchTrainer(object):
    def __init__(self, args):
        self.args = args
        self.uv_cfg = cfg
        if args.device is not None:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"
        if self.args.use_GMM:
            from pose_sampling import PoseSampler
            self.pose_sampler = PoseSampler(pose_folder = './zju_mocap/CoreView_377/new_params',)
        self.yolo_model = YOLOv3Darknet().eval().to(device)
        self.yolo_model.load_darknet_weights('arch/weights/yolov3.weights')
        self.yolo_model.eval()
        
        logging.basicConfig(filename= args.save_path + 'logging.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if args.arch == "rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
            self.model.eval()
        elif args.arch == "yolov3":
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
            self.model.eval()
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
            self.model.eval()
        elif args.arch == "deformable-detr":
            self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
            self.model.eval()
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
            self.model.eval()
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True).eval().to(device)
            self.model.eval()
        elif args.arch == "retina":
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).eval().to(device)
            self.model.eval()
        elif args.arch == "fcos":
            self.model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True).eval().to(device)
            self.model.eval()
        elif args.arch == "ssd":
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True).eval().to(device)
        # elif args.arch == "yolov5":
        #     self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(device)
        elif args.arch == "yolov8":
            self.model = YOLO('yolov8n.pt', task="detect")
        elif args.arch == "yolov9":
            self.model = YOLO('yolov9c.pt', task="detect")
        elif args.arch == "ensemble":
            pass
        else:
            raise NotImplementedError
        
        self.lpips_loss = lpips.LPIPS(net='alex').eval().to(self.device)
        self.batch_size = args.batch_size

        self.patch_transformer = PatchTransformer().to(device)
        if args.arch == "rcnn":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
        elif args.arch == "yolov3":
            self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "detr":
            self.prob_extractor = DetrMaxProbExtractor(0, 80, self.img_size).to(device)
        elif args.arch == "deformable-detr":
            self.prob_extractor = DeformableDetrProbExtractor(0,80,self.img_size).to(device)
        elif args.arch == "ensemble":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
            self.prob_extractor2 = YOLOv3MaxProbExtractor(0, 80, self.yolo_model, self.img_size).to(device)
        self.tv_loss = TotalVariation()

        self.alpha = args.alpha
        self.azim = torch.zeros(self.batch_size)
        self.blend_params = None
        self.sampler_probs = torch.ones([36]).to(device)
        self.train_loader = self.get_loader('./data/background', True)
        self.test_loader = self.get_loader('./data/new_background', True)

        self.epoch_length = len(self.train_loader)
        print(f'One training epoch has {len(self.train_loader.dataset)} images')
        print(f'One test epoch has {len(self.test_loader.dataset)} images')

        color_transform = ColorTransform('./data/color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 864
        self.fig_size_H_t = 484
        self.fig_size_W_t = 700
        resolution = 4
        h, w, h_t, w_t = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution), int(self.fig_size_H_t / resolution), int(self.fig_size_W_t / resolution)
        self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t
        
        self.expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
        self.expand_kernel.weight.data.fill_(0)
        self.expand_kernel.bias.data.fill_(0)
        for i in range(3):
            self.expand_kernel.weight[i, i, :, :].data.fill_(1)
        
        self.prompts = self.args.prompt
        self.negative_prompts = "cartoon, unrealistic"
        
        self.diffusion_rate =  args.diffusion_rate
        self.diffusion_steps = args.diffusion_steps
        
        self.sd_cfg = {"pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "diffusion_steps": self.diffusion_steps,
                "guidance_scale": 7.5,
                "min_step_percent": 0.02,
                "max_step_percent": 0.98,
                "do_classifier_free_guidance": args.do_classifier_free_guidance,
                "half_precision_weights": args.half_precision_weights}
          
        if  self.sd_cfg["half_precision_weights"] == True:
            self.weight_type = torch.float16
            self.color_transform = self.color_transform.to(self.weight_type)
        else:
            self.weight_type = torch.float32

        self.stable_diffusion_model = StableDiffusionGenerator(self.sd_cfg, device)
        self.preprocessor = VaeImageProcessor(do_resize = False, vae_scale_factor=8)
        
        # load the UV-volume model
        self.uvvolume_network = nts.Network().to(self.device).eval()
        self.uvvolume_renderer = Renderer(self.uvvolume_network)
        self.action_dataloader = make_data_loader(self.uv_cfg, is_train=True)
        load_network(self.uvvolume_network,
                            self.uv_cfg.trained_model_dir,
                            resume=self.uv_cfg.resume,
                            epoch=self.uv_cfg.test.epoch)
        
        self.tex_size = 256
        image_bgr = cv2.imread(self.args.pretrained_texture_stack_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.tex_size * 6, self.tex_size * 4))
        TextureIm_pose = np.array(image_rgb)
        self.Texture_pose = torch.zeros((24, 3, self.tex_size, self.tex_size)).to(self.device)
        for i in range(len(self.Texture_pose)):
            x = i // 6 * self.tex_size
            y = i % 6 * self.tex_size
            texture = TextureIm_pose[x:x + self.tex_size, y:y + self.tex_size].transpose(1,0,2)
            self.Texture_pose[i]  = torch.from_numpy(texture).float().permute(2,0,1)/255  
            
        tps_para_front = torch.load( "./UV_Volumes/data/trained_model/TPS/tps_para_front.pt")    
        self.theta_tensor = tps_para_front["theta_tensor"].to(self.device).to(self.weight_type)
        self.c_src = tps_para_front["c_src_tensor"].to(self.device).to(self.weight_type)
        self.c_dst = tps_para_front["c_dst_tensor"].to(self.device).to(self.weight_type)

    def prepare_latents(self, batch_size, num_channels_latents, height=256, width=256, dtype=torch.float16, device=torch.device("cuda")):
        vae_scale_factor = 8
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        latents = randn_tensor(shape, device=device, dtype=dtype)
        return latents 
    
    def prepare_image(self, image_path = None):
        # We first use black-box optimization to find a image.
        image = Image.open(image_path)
        # Define the desired image size for Stable Diffusion
        desired_size = (256, 256)
        # Resize the image
        resized_image = image.resize(desired_size)
        channels = len(resized_image.split())
        # Convert the image to tensor and normalize pixel values
        if channels == 4: 
            resized_image = resized_image.convert("RGB")
        input_image = self.preprocessor.preprocess(resized_image).to(self.weight_type).to(self.device)
        adv_input_image = input_image.to(self.weight_type).to(self.device).detach().clone()
        return adv_input_image

    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=4)
        return loader

    def init_tensorboard(self, name=None):
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        fname = self.args.save_path.split('/')[-1] # the detector name
        print("save tensorboard at:",f'runs_new/{TIMESTAMP}_{fname}' )
        return SummaryWriter(f'runs_new/{TIMESTAMP}_{fname}')

    def sample_cameras(self, theta=None, elev=None):
        if theta is not None:
            if isinstance(theta, float) or isinstance(theta, int):
                self.azim = torch.zeros(self.batch_size).fill_(theta)
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone()
            elif isinstance(theta, np.ndarray):
                self.azip = torch.from_numpy(theta)
            else:
                raise ValueError
        else:
            if self.alpha > 0:
                exp = (self.alpha * self.sampler_probs).softmax(0)
                azim = torch.multinomial(exp, self.batch_size, replacement=True)
                self.azim_inds = azim
                azim = azim.to(exp)
                self.azim = (azim + azim.new(size=azim.shape).uniform_() - 0.5) * 360 / len(exp)
            else:
                self.azim_inds = None
                self.azim = (torch.zeros(self.batch_size).uniform_() - 0.5) * 360
        if elev is not None:
            elev = torch.zeros(self.batch_size).fill_(elev)
        else:
            elev = 10 + 8 * torch.zeros(self.batch_size).uniform_(-1, 1)
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        return

    def sample_lights(self, r=None):
        if r is None:
            r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            self.lights = AmbientLights(device=self.device)
        elif r < 0.67:
            self.lights = DirectionalLights(device=self.device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
        else:
           self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
       
        return

    def synthesis_image_uv_volume(self, background_batch, action_batch, Texture_pose, use_tps2d=False, use_tps3d=False):
        with torch.no_grad():
            output = self.uvvolume_renderer.render(action_batch)
        # get the iuv data, i is the index of human parts, uv is the predicted uv coordinates, see densepsoe,
        iuv_map = output["iuv_body"].to(self.device)
        i_map = F.softmax(iuv_map[..., :24], dim=-1) #mask, 24
        u_map = iuv_map[..., 24:48] # mask, 24
        v_map = iuv_map[..., 48:]   # mask, 24
        uv_map = torch.stack((u_map, v_map), -1)  # mask, 24, 2
        grid = (2*uv_map.permute(1,0,2).unsqueeze(1)-1)
        texture_gridsample = torch.nn.functional.grid_sample(Texture_pose,
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
        
        batchsize = background_batch.shape[0]
        rgb_pred_crop = rgb_pred_crop.permute(2,0,1).unsqueeze(0).repeat( batchsize, 1, 1, 1)
        x, y, w, h = cv2.boundingRect(mask_at_box.detach().cpu().numpy().astype(np.uint8))
        adv_batch = rgb_pred_crop[:, :, y:y + h, x:x + w]
        
        # Desired output shape
        desired_shape = (batchsize, 3, 416, 416)

        # Calculate padding values for each dimension
        pad_height = max(0, desired_shape[2] - adv_batch.size(2))
        pad_width =  max(0, desired_shape[3] -  adv_batch.size(3))

        # Pad the tensor
        adv_batch = torch.nn.functional.pad(adv_batch, (0, pad_width, 0, pad_height), value=0)

        # add a illumination channel
        summed_tensor = torch.sum(adv_batch, dim=1, keepdim=True)
        adv_batch = torch.cat((adv_batch, summed_tensor), dim=1)
        
        p_background_batch, gt = self.patch_transformer(background_batch, adv_batch) #gt is the ground truth boxes
        return p_background_batch, gt , output
    
    def update_UVmap_via_latent(self, latents, prompts, negative_prompts, texture = None):
        # camouflage:
        if texture is None:
            tex_texture  = self.stable_diffusion_model.edit_latent(latents, prompts,negative_prompts)
            # tex_trouser  = self.stable_diffusion_model_trouser(prompts,negative_prompts, latent_trouser, latent_noises_trouser)
            tex_texture  = self.color_transform(tex_texture)
        else:
            tex_texture = texture
            
        tex_texture = F.interpolate(tex_texture, (128, 128), mode="bilinear")
        
        tex_texture_repeat = tex_texture.repeat(1,1, 2, 2).permute(0,1,3,2) # rotate 90 degree
        # cv2.imwrite("tshirt_front.png", (Tshirt_front[0].permute(1,2,0).detach().cpu().numpy()*255)[...,[2,1,0]]) # RGB 2 BGR
        
        wraped, _ = warp_image_cv(tex_texture.repeat(1,1, 2, 2), self.c_src, self.c_dst, self.theta_tensor, dshape=(1,3,256,256))
        tex_texture_wraped = wraped.permute(0,1,3,2)

        Texture_pose = self.Texture_pose.clone().detach().to(self.device)
        Texture_pose[0] = tex_texture_repeat # back of upper body
        Texture_pose[1] = tex_texture_repeat
        # Texture_pose[1] = tex_texture_wraped  # front of upper body
        Texture_pose[6] = tex_texture_repeat # front of left leg 
        Texture_pose[7] = tex_texture_repeat # front of right leg 
        Texture_pose[8] = tex_texture_repeat # back of left leg 
        Texture_pose[9] = tex_texture_repeat # back  of right leg 
        Texture_pose[10] = tex_texture_repeat # 10-13 are lower legs
        Texture_pose[11] = tex_texture_repeat
        Texture_pose[12] = tex_texture_repeat 
        Texture_pose[13] = tex_texture_repeat
        Texture_pose[14] = tex_texture_repeat # 14-17 are big arms
        Texture_pose[15] = tex_texture_repeat
        Texture_pose[16] = tex_texture_repeat 
        Texture_pose[17] = tex_texture_repeat
        Texture_pose[18] = tex_texture_repeat # 18-21 are forearms
        Texture_pose[19] = tex_texture_repeat
        Texture_pose[20] = tex_texture_repeat 
        Texture_pose[21] = tex_texture_repeat

        return tex_texture, Texture_pose

    # compute the KL divergence between the adverasarial dirstribution and standard normal distruibution
    def compute_kl_divergence(self, mu, sigma):
        kl_loss = 0.5 * (torch.norm(mu, p=2) - torch.log(torch.sum(sigma)) + torch.sum(sigma))
        return kl_loss

    def load_weights(self, save_path, epoch):
        path = save_path + f"/advtexture-{epoch}.pth"
        latent_dict = torch.load(path, map_location='cpu')
        return latent_dict
    
    def to_cuda(self, batch, device):
        for k in batch:
            if k == 'meta' or k=='img_name' or k=='epoch':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(device)
        return batch

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        # the latent code to generate the adversarial pattern
        #self.prompts = "colorful repeated patterns"
        self.prompts = self.args.prompt
        self.negative_prompts = ""
        checkpoints = self.args.checkpoints
        if checkpoints == 0:
            init_latents = self.prepare_latents(batch_size=1, num_channels_latents=4, 
                                        height=256, width=256, device=self.device, 
                                        dtype = self.weight_type)
            adv_latents = torch.autograd.Variable(init_latents.type(self.weight_type), requires_grad=True).to(self.device)
        else:
            latent_dict = self.load_weights(self.args.save_path, checkpoints)
            adv_latents = latent_dict["latent"].to(self.device).to(self.weight_type)
            print("load weights from ", self.args.save_path)
            logging.info("load weights from " + self.args.save_path)
        
        if self.args.use_lpips_loss:    
            if checkpoints > 0:
                first_latent_dict = self.load_weights(self.args.save_path, 0)
                first_latents = first_latent_dict["latent"].to(self.device).to(self.weight_type)
                original_tex_texture,Texture_pose = self.update_UVmap_via_latent(first_latents,
                                                    self.prompts, self.negative_prompts)
                original_tex_texture = original_tex_texture.detach().clone().requires_grad_(False)   
            else:
                original_tex_texture,Texture_pose  = self.update_UVmap_via_latent(adv_latents.detach().clone(),
                                                    self.prompts, self.negative_prompts)
                original_tex_texture = original_tex_texture.detach().clone().requires_grad_(False)    
            
        self.optimizer = torch.optim.SGD([adv_latents], lr=self.args.lr, momentum=0.9)
        
        et0 = time.time()
       
        self.writer = self.init_tensorboard()
        args = self.args
        if args.use_GMM is False:
            self.action_dataloader = make_data_loader(self.uv_cfg, is_train=True)
        else:
            self.action_data_loader = torch.utils.data.DataLoader(self.pose_sampler,
                                                batch_size = 1,
                                                num_workers=4)
        
        # Create an iterator from the DataLoader
        iterator = iter(self.action_dataloader)
        # Fetch the first batch
        action_batch = next(iterator)

        for epoch in tqdm(range(checkpoints, args.nepoch)):
            print('######################################')
            ep_det_loss = 0
            ep_norm_loss = 0
            ep_loss = 0
            ep_mean_prob = 0
            ep_tv_loss = 0
            ep_ctrl_loss = 0
            ep_seed_loss = 0
            ep_lpips_loss = 0
            eff_count = 0  # record how many images in this epoch are really in training so that we can calculate accurate loss
            if epoch % 100 == 99:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * args.lr_decay

            # theta = 0
            # self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]]) 

            if args.action_sampling==True:
                if args.use_GMM:
                    self.action_data_loader = torch.utils.data.DataLoader(self.pose_sampler,
                                               batch_size = 1,
                                               num_workers=4)
                else:
                    self.action_dataloader = make_data_loader(self.uv_cfg, is_train=True)
                
                action_data_iter = iter(self.action_dataloader)

            for i_batch, background_batch in enumerate(self.train_loader):
                background_batch = background_batch.to(self.device)
                t0 = time.time()
                self.optimizer.zero_grad()  
                tex_texture,Texture_pose  = self.update_UVmap_via_latent(adv_latents,
                    self.prompts, self.negative_prompts) #tex_texture: 1,3,128, 128 Texture_pose:24,3,256, 256
                if args.action_sampling == True:
                    loss = 0          
                    #for action_count in range(0, action_batch_size):
                    action_batch = next(action_data_iter)

                    action_batch = self.to_cuda(action_batch, self.device)
                    action_batch['epoch'] = -1
                    p_background_batch, gt, output = self.synthesis_image_uv_volume(background_batch, action_batch, Texture_pose,  False, False)
                    
                    # plt.figure(figsize=(10, 10))
                    # plt.subplot(121)
                    # plt.imsave(args.save_path + f"/texture_tshirt-{epoch}-{i_batch}.png", tex_texture.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                    # print(args.save_path + f"/texture_tshirt-{self.uv_cfg.exp_name}-{epoch}-{i_batch}.png")
                    # plt.subplot(122)
                    # plt.imsave(args.save_path + f"/p_background_batch-{epoch}-{i_batch}.png", p_background_batch.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                    # print(args.save_path + f"/p_background_batch-{self.uv_cfg.exp_name}-{epoch}-{i_batch}.png")
                    
                    normalize = True
                    if self.args.arch == "deformable-detr" and normalize:
                        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        p_background_batch = normalize(p_background_batch)

                    try:
                        if self.args.arch == "ensemble":
                            output = self.rcnn_model(p_background_batch)
                            det_loss1, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                            output2 = self.yolo_model(p_background_batch)
                            det_loss2, max_prob_list2 = self.prob_extractor2(output2, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                            # output3 = self.detr_model(p_background_batch)
                            # det_loss3, max_prob_list3 = self.prob_extractor3(output3, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                            det_loss = (det_loss1 + det_loss2)/2
                            eff_count += 1
                        else:
                            output = self.model(p_background_batch)
                            det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                            eff_count += 1
                    except RuntimeError:  # current batch of imgs have no bbox be detected
                        continue

                    loss = loss + det_loss
                    tv_loss = torch.tensor(0)
                    lpips_loss = torch.tensor(0)
                    
                    if self.args.use_tv_loss:
                        tv_loss = self.args.tv_loss_weight * self.tv_loss(tex_texture)
                        loss += tv_loss 
                        ep_tv_loss += tv_loss.item()
                    
                    if self.args.use_lpips_loss:    
                        lpips_loss = self.args.lpips_loss_weight * self.lpips_loss(tex_texture.permute(0, 3, 1, 2), original_tex_texture)
                        loss +=  lpips_loss.mean()
                        ep_norm_loss += lpips_loss.item()
                    
                    # kl_loss = self.compute_kl_divergence(mu_tshirt, sigma_tshirt)+ self.compute_kl_divergence(mu_trousers, sigma_trousers)
                    
                    ep_mean_prob += max_prob_list.mean().item()
                    ep_det_loss += det_loss.item()
                    ep_loss += loss.item()
                    t3 = time.time()     
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = 0          
                    action_batch = self.to_cuda(action_batch, self.device)
                    action_batch['epoch'] = -1
                    p_background_batch, gt, output = self.synthesis_image_uv_volume(background_batch, action_batch, Texture_pose,  False, False)
                    normalize = True
                    if self.args.arch == "deformable-detr" and normalize:
                        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        p_background_batch = normalize(p_background_batch) 
                    output = self.model(p_background_batch)
                    try:
                        det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                        eff_count += 1
                    except RuntimeError:  # current batch of imgs have no bbox be detected
                        continue
                    
                    loss = loss + det_loss
                    tv_loss = torch.tensor(0)
                    lpips_loss = torch.tensor(0)
                    if args.use_tv_loss :
                        tv_loss = self.tv_loss(tex_texture)
                        loss += tv_loss * args.tv_loss_weight
                    if args.use_lpips_loss:    
                        lpips_loss = args.lpips_loss_weight * self.lpips_loss(tex_texture.permute(0, 3, 1, 2), original_tex_texture)
                        loss +=  lpips_loss.mean()
                        ep_norm_loss += lpips_loss.item()
                    ep_mean_prob += max_prob_list.mean().item()
                    ep_det_loss += det_loss.item()
                    ep_tv_loss += tv_loss.item()
                    ep_norm_loss += lpips_loss.item()
                    ep_loss += loss.item()
                    t3 = time.time()     
                    loss.backward()
                    self.optimizer.step()
                if self.args.optimize_type == "latent":
                        torch.nn.utils.clip_grad_norm_(adv_latents, 0.1)
                if i_batch % 20 == 0:
                    print("iteration", i_batch, "loss", loss.detach().cpu().numpy(),
                        "det_loss", det_loss.detach().cpu().numpy(),
                        "lpips_loss", lpips_loss.detach().cpu().numpy(),
                        'tv_loss', tv_loss.detach().cpu().numpy(),
                        "lpips_loss", lpips_loss.detach().cpu().numpy(),
                        "time:", t3)
            torch.cuda.empty_cache()
            et1 = time.time()
            eff_count =  eff_count + 1e-6
            ep_det_loss = ep_det_loss / eff_count
            ep_norm_loss = ep_norm_loss / eff_count
            ep_lpips_loss = ep_lpips_loss / eff_count
            ep_loss = ep_loss / eff_count
            ep_tv_loss = ep_tv_loss / eff_count
            ep_ctrl_loss = ep_ctrl_loss / eff_count
            ep_mean_prob = ep_mean_prob / eff_count
            ep_seed_loss = ep_seed_loss / eff_count
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print(' MEAN PROB: ', ep_mean_prob)
                print('   TV LOSS: ', ep_tv_loss)
                print(' NORM LOSS: ', ep_norm_loss)
                print('EPOCH TIME: ', et1 - et0)
                logging.info('  EPOCH NR: ' + str(epoch))
                logging.info('EPOCH LOSS: ' + str(ep_loss))
                logging.info('  DET LOSS: ' + str(ep_det_loss))
                self.writer.add_scalar('epoch/total_loss', ep_loss, epoch)
                self.writer.add_scalar('epoch/norm_loss', ep_norm_loss, epoch)
                self.writer.add_scalar('epoch/tv_loss', ep_tv_loss, epoch)
                self.writer.add_scalar('epoch/det_loss', ep_det_loss, epoch)
                self.writer.add_scalar('epoch/lpips_loss', ep_lpips_loss, epoch)
                self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
                
            et0 = time.time()
            if (epoch) % 50 == 0:
                plt.figure(figsize=(10, 10))
                plt.subplot(111)
                plt.imsave(args.save_path + f"/texture_tshirt-{self.uv_cfg.exp_name}-{epoch}.png", tex_texture.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                fig = plt.figure(figsize=(10, 10))
                plt.subplot(111)
                plt.imshow(p_background_batch.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                fig.savefig(args.save_path + f"/p_background_batch-{self.uv_cfg.exp_name}-{epoch}.png")
                logging.info('save image at' + args.save_path + f"/texture_tshirt-{self.uv_cfg.exp_name}-{epoch}.png")
                logging.info('save image at' + args.save_path + f"/p_background_batch-{self.uv_cfg.exp_name}-{epoch}.png")
                arc_list = ['rcnn','yolov3']
                detect_model_list = [self.model, self.yolo_model]   
                for i in range(len(arc_list)):
                    precision, recall, avg, confs = trainer.test(arc_list[i], adv_latents, conf_thresh=0.01, iou_thresh=args.test_iou, 
                                                                        angle_sample=37, mode=args.test_mode, detect_model = detect_model_list[i])
                    asr = (confs < 0.5).mean()
                    print(arc_list[i] + " TEST ASR:", asr)
                    logging.info(arc_list[i] + " TEST ASR:" + str(asr))
                    self.writer.add_scalar(f'epoch/{arc_list[i]}-test-ASR', asr, epoch)
                    info = [precision, recall, avg, confs]
                    path = args.save_path + '/' + 'test_results_epoch' +str(epoch)
                    path = path + '_iou' + str(args.test_iou).replace('.', '') + '.pkl'
                    result = {"precision":precision, "recall":recall, "avg":avg, "confs":confs}
                    with open(path, "wb") as f:
                        pickle.dump(result, f)
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                if self.args.optimize_type == "latent":
                    latent_dict = {
                                "tex_texture": tex_texture.detach().cpu(),
                                "latent": adv_latents.detach().cpu(),
                                "ASR": torch.tensor(asr)
                                }
                path = args.save_path + f'/advtexture-{epoch}.pth'
                torch.save(latent_dict, path)

    def test(self, arch, adv_latents, conf_thresh, iou_thresh, num_of_samples=100, angle_sample=37, mode='person', detect_model = None):
        print(f'One test epoch has {len(self.test_loader.dataset)} images')
        confs = []
        self.sample_lights(r=0.1)

        total = 0.
        positives = []
        if self.args.use_GMM:
            self.test_action_dataloader = torch.utils.data.DataLoader(self.pose_sampler,
                                              batch_size = 1,
                                              num_workers=4)
        else:
            self.test_action_dataloader = make_data_loader(self.uv_cfg, is_train=False)

        
        print("test action images number: ", len(self.test_action_dataloader.dataset))
        # action_data_iter = iter(self.action_dataloader)
        with torch.no_grad():
            tex_texture,Texture_pose  = self.update_UVmap_via_latent(adv_latents,
                    self.prompts, self.negative_prompts) #tex_texture: 1,3,128, 128 Texture_pose:24,3,256, 256
            count = 0
            for i_batch, background_batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), position=0):
                background_batch = background_batch.to(self.device)
                for it, action_batch in enumerate(self.test_action_dataloader):
                    action_batch = self.to_cuda(action_batch, self.device)
                    action_batch['epoch'] = -1
                    p_background_batch, gt, output = self.synthesis_image_uv_volume(background_batch, action_batch, Texture_pose,  False, False)
                    # cv2_img = p_background_batch[0].permute(1,2,0).detach().cpu().numpy()[...,[2,1,0]] * 255
                    count = count+1
                    normalize = True
                    if arch == "deformable-detr" and normalize:
                        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        p_background_batch = normalize(p_background_batch)
                    if detect_model is not None:
                        model = detect_model
                    else:
                        model = self.model
                        
                    output = model(p_background_batch)
                    total += len(p_background_batch)  # since 1 image only has 1 gt, so the total # gt is just = the total # images
                    pos = []
                    conf_thresh = 0.0 if arch in ['rcnn'] else 0.1
                    person_cls = 0
                    output = adv_camou_utils.get_region_boxes_general(output, model, conf_thresh=conf_thresh, name=arch)

                    for i, boxes in enumerate(output):
                        if len(boxes) == 0:
                            pos.append((0.0, False))
                            continue
                        assert boxes.shape[1] == 7
                        boxes = adv_camou_utils.nms(boxes, nms_thresh=args.test_nms_thresh)
                        w1 = boxes[..., 0] - boxes[..., 2] / 2
                        h1 = boxes[..., 1] - boxes[..., 3] / 2
                        w2 = boxes[..., 0] + boxes[..., 2] / 2
                        h2 = boxes[..., 1] + boxes[..., 3] / 2
                        bboxes = torch.stack([w1, h1, w2, h2], dim=-1)
                        bboxes = bboxes.view(-1, 4).detach() * self.img_size
                        scores = boxes[..., 4]
                        labels = boxes[..., 6]

                        if (len(bboxes) == 0):
                            pos.append((0.0, False))
                            continue
                        scores_ordered, inds = scores.sort(descending=True)
                        scores = scores_ordered
                        bboxes = bboxes[inds]
                        labels = labels[inds]
                        inds_th = scores > conf_thresh
                        scores = scores[inds_th]
                        bboxes = bboxes[inds_th]
                        labels = labels[inds_th]

                        if mode == 'person':
                            inds_label = labels == person_cls
                            scores = scores[inds_label]
                            bboxes = bboxes[inds_label]
                            labels = labels[inds_label]
                        elif mode == 'all':
                            pass
                        else:
                            raise ValueError

                        if (len(bboxes) == 0):
                            pos.append((0.0, False))
                            continue
                        bboxes = bboxes.to(self.device)
                        ious = torchvision.ops.box_iou(bboxes.data,
                                                        gt[i].unsqueeze(0))  # get iou of all boxes in this image
                        noids = (ious.squeeze(-1) > iou_thresh).nonzero()
                        if noids.shape[0] == 0:
                            pos.append((0.0, False))
                        else:
                            noid = noids.min()
                            if labels[noid] == person_cls:
                                pos.append((scores[noid].item(), True))
                            else:
                                pos.append((scores[noid].item(), False))
                    positives.extend(pos)
                    confs.extend([p[0] if p[1] else 0.0 for p in pos])
        positives = sorted(positives, key=lambda d: d[0], reverse=True)
        confs = np.array(confs)
        tps = []
        fps = []
        tp_counter = 0
        fp_counter = 0
        # all matches in dataset
        for pos in positives:
            if pos[1]:
                tp_counter += 1
            else:
                fp_counter += 1
            tps.append(tp_counter)
            fps.append(fp_counter)
        precision = []
        recall = []
        for tp, fp in zip(tps, fps):
            recall.append(tp / total)
            if tp == 0:
                precision.append(0.0)
            else:
                precision.append(tp / (fp + tp))

        if len(precision) > 1 and len(recall) > 1:
            p = np.array(precision)
            r = np.array(recall)
            p_start = p[np.argmin(r)]
            samples = np.linspace(0., 1., num_of_samples)
            interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
            avg = sum(interpolated) / len(interpolated)
        elif len(precision) > 0 and len(recall) > 0:
            # 1 point on PR: AP is box between (0,0) and (p,r)
            avg = precision[0] * recall[0]
        else:
            avg = float('nan')
        return precision, recall, avg, confs
        
    def generate_video(self, adv_latents, conf_thresh=0.5, iou_thresh=0.1, video_path = "./results/video/result.mp4"):
        self.uv_cfg.test_view = [5]
        self.uv_cfg.num_train_frame = 500
        self.uv_cfg.frame_interval = 1
        self.uv_cfg.begin_ith_frame = 0
        self.test_action_dataloader = make_data_loader(self.uv_cfg, is_train=False, is_distributed=False)
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (1024, 512))
        tex_texture,Texture_pose  = self.update_UVmap_via_latent(adv_latents,
                                self.prompts, self.negative_prompts)
        for it, action_batch in tqdm(enumerate(self.test_action_dataloader)):
            action_batch = self.to_cuda(action_batch, self.device)
            action_batch['epoch'] = -1
            with torch.no_grad():
                uv_output = self.uvvolume_renderer.render(action_batch)
                # get the iuv data, i is the index of human parts, uv is the predicted uv coordinates, see densepsoe,
                iuv_map = uv_output["iuv_body"].to(self.device)
                i_map = F.softmax(iuv_map[..., :24], dim=-1) #mask, 24
                u_map = iuv_map[..., 24:48] # mask, 24
                v_map = iuv_map[..., 48:]   # mask, 24
                uv_map = torch.stack((u_map, v_map), -1)  # mask, 24, 2
                grid = (2*uv_map.permute(1,0,2).unsqueeze(1)-1)
                # USE THE NORMAL TEXTURE
                texture_gridsample = torch.nn.functional.grid_sample(self.Texture_pose,
                                    grid, 
                                    mode='bilinear', align_corners=False) 
                # outputsize is (mask, 3)
                rgb_pred = (i_map.permute(1,0).unsqueeze(2) * texture_gridsample.permute(0,2,3,1).view(24,-1,3)).sum(0)
                rgb_gt = action_batch['rgb']
                rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
                rgb_padding[uv_output['T_last'] < self.uv_cfg.T_threshold] = rgb_pred
                # crop rgb pred at box
                mask_at_box = action_batch['mask_at_box'][0]
                H, W = action_batch['H'][0], action_batch['W'][0]
                mask_at_box = mask_at_box.reshape(H, W)
                rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
                rgb_pred_crop[mask_at_box] = rgb_padding[0]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1).unsqueeze(0)   
                output = self.model(rgb_pred_crop)
                conf_thresh = 0.0 if self.args.arch in ['rcnn'] else 0.1
                output = adv_camou_utils.get_region_boxes_general(output, self.model, conf_thresh=conf_thresh, name=self.args.arch, img_size = 512)
                cv2_img = rgb_pred_crop[0].permute(1,2,0).detach().cpu().numpy()[...,[2,1,0]] * 255
                cv2_img = cv2.resize(cv2_img, (512, 512))
                cv2_img = cv2_img.astype(np.uint8)
                if output is not []:
                    color = (255, 0, 255)  # Magenta color
                    for box in output:
                        x_center, y_center, width, height, conf, cls_conf, cls_id = box.detach().cpu().numpy().tolist()[0]
                        if cls_id  < 80 and cls_conf > 0.5:
                            x1 = int((x_center - width/2)*512)
                            y1 = int((y_center - height/2)*512)
                            x2 = int((x_center + width/2)*512)
                            y2 = int((y_center + height/2)*512)
                            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 2)
                            # Put the label with a background
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_size = cv2.getTextSize(class_names[int(cls_id)]+" " +str(cls_conf), font, 0.5, 1)[0]
                            text_x = x1
                            text_y = y1
                            cv2.rectangle(cv2_img, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)
                            cv2.putText(cv2_img, class_names[int(cls_id)],(text_x, text_y), font, 0.5, (0,0,0), 1, cv2.LINE_AA)     
                            
                # USE THE ADV TEXTURE
                texture_gridsample = torch.nn.functional.grid_sample(Texture_pose,
                                    grid, 
                                    mode='bilinear', align_corners=False) 
                rgb_pred = (i_map.permute(1,0).unsqueeze(2) * texture_gridsample.permute(0,2,3,1).view(24,-1,3)).sum(0)
                rgb_gt = action_batch['rgb']
                rgb_padding = torch.cuda.FloatTensor(rgb_gt.shape).fill_(0.)
                rgb_padding[uv_output['T_last'] < self.uv_cfg.T_threshold] = rgb_pred
                # crop rgb pred at box
                mask_at_box = action_batch['mask_at_box'][0]
                H, W = action_batch['H'][0], action_batch['W'][0]
                mask_at_box = mask_at_box.reshape(H, W)
                rgb_pred_crop = torch.cuda.FloatTensor(H, W, 3).fill_(0.)
                rgb_pred_crop[mask_at_box] = rgb_padding[0]
                rgb_pred_crop = rgb_pred_crop.permute(2,0,1).unsqueeze(0)   
                output = self.model(rgb_pred_crop)
                conf_thresh = 0.0 if self.args.arch in ['rcnn'] else 0.1
                output = adv_camou_utils.get_region_boxes_general(output, self.model, conf_thresh=conf_thresh, name=self.args.arch, img_size = 512)
                cv2_img2 = rgb_pred_crop[0].permute(1,2,0).detach().cpu().numpy()[...,[2,1,0]] * 255
                cv2_img2 = cv2.resize(cv2_img2, (512, 512))
                cv2_img2 = cv2_img2.astype(np.uint8)
                if output is not []:
                    color = (255, 0, 255)  # Magenta color
                    for box in output:
                        x_center, y_center, width, height, conf, cls_conf, cls_id = box.detach().cpu().numpy().tolist()[0]
                        if cls_id  < 80 and cls_conf > 0.5:
                            x1 = int((x_center - width/2)*512)
                            y1 = int((y_center - height/2)*512)
                            x2 = int((x_center + width/2)*512)
                            y2 = int((y_center + height/2)*512)
                            cv2.rectangle(cv2_img2, (x1, y1), (x2, y2), color, 2)
                            # Put the label with a background
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_size = cv2.getTextSize(class_names[int(cls_id)]+" " +str(cls_conf), font, 0.5, 1)[0]
                            text_x = x1
                            text_y = y1
                            cv2.rectangle(cv2_img2, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)
                            cv2.putText(cv2_img2, class_names[int(cls_id)],(text_x, text_y), font, 0.5, (0,0,0), 1, cv2.LINE_AA)                          
                out.write(cv2.hconcat([cv2_img, cv2_img2]))
        out.release()
        print("save videp at " + video_path)
        
        
if __name__ == '__main__':
    print('Version 2.0')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--alpha", type=float, default=10, help='')
    parser.add_argument("--tv_loss_weight", type=float, default=0, help='')
    parser.add_argument("--lpips_loss_weight", type=float, default=0, help='')
    parser.add_argument("--use_tv_loss", type=bool, default=False, help='')
    parser.add_argument("--use_lpips_loss", type=bool, default=False, help='')
    parser.add_argument("--lr_decay", type=float, default=0.5, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--arch", type=str, default="rcnn")
    parser.add_argument("--seed_ratio", default=1.0, type=float, help='The ratio of trainable part when seed type is variable')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--mode", default="video", help='train, test, video')
    parser.add_argument("--test_iou", type=float, default=0.1, help='')
    parser.add_argument("--test_nms_thresh", type=float, default=1.0, help='')
    parser.add_argument("--test_mode", default='person', help='person, all')
    parser.add_argument("--test_suffix", default='', help='')
    parser.add_argument("--train_iou", type=float, default=0.01, help='')
    parser.add_argument('--device', default='cuda:1', help='')    
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=1000, help='')
    parser.add_argument('--checkpoints', type=int, default=300, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--save_path', default='./results/', help='')
    parser.add_argument("--optimize_type", type=str, default="latent", help='image , latent_noise or latent')
    parser.add_argument("--diffusion_rate", type=float, default=0.5, help='')
    parser.add_argument("--diffusion_steps", type=int, default=10, help="")
    parser.add_argument("--do_classifier_free_guidance", default=False, help='')
    parser.add_argument("--half_precision_weights", default=True, help='')
    parser.add_argument("--prompt", default="one horse", help='three bear, colorful repeated patterns')
    parser.add_argument("--pattern_mode", type=str, default="repeat", help='repeat, whole (for colorful repeated patterns)')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default= "./pretained_model/miniSD.ckpt",
                        help = "if use stablediffusion with 512*512, change to runwayml/stable-diffusion-v1-5")
    parser.add_argument("--action_sampling", type = bool, default=True, help='')
    parser.add_argument("--use_GMM", type = bool, default=False, help='')
    parser.add_argument("--uv_cfg_file", type = str, default="./UV_Volumes/configs/zju_mocap_exp/377.yaml", help='action sampling file name')
    parser.add_argument("--pretrained_texture_stack_path", type = str, default="./data/texture_stacks/texture_static_frame0000_epoch0399.png")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Train info:", args)
    with open(args.uv_cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)
    exp_name = current_cfg.exp_name
    args.save_path = args.save_path + "/" +  args.arch + "/" + args.optimize_type + "/" + args.prompt.replace(" ","_") + "/" + exp_name
    os.makedirs(args.save_path, exist_ok=True)

    print("save directory:", args.save_path)
    trainer = PatchTrainer(args)
    
    if args.mode == "train":
        import json
        with open(args.save_path + "/CONFIG.txt", "w") as outfile: 
            json.dump(args.__dict__, outfile, indent=2)
        trainer.train()
    elif args.mode == "test":
        save_path = "./results/rcnn/latent/one_horse/zju377/advtexture-300.pth"
        print(save_path)
        latent_dict = torch.load(save_path, map_location='cpu')
        path = "evaluation.txt"
        test_arch = ["rcnn",  "mask_rcnn",  "regina",  "ssd",  "yolov35" , "detr", "yolov8", "yolov9", "fcos"]
        for arch in test_arch:
            args.arch = arch
            trainer = PatchTrainer(args)  
            adv_latents = latent_dict["latent"].to(trainer.device).to(trainer.weight_type)
            iou_list = [ 0.5, 0.3, 0.1, 0.01]
            for iou in iou_list:
                precision, recall, avg, confs, thetas = trainer.test(args.arch, adv_latents, conf_thresh=0.01, iou_thresh=iou, angle_sample=37, mode=args.test_mode)
                for con in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    with open(path, 'a') as file:   
                        file.write(f"{arch} iou {iou} confs {con}  ASR\: {(confs < con).mean()}\n")
                        path = args.save_path + '/' + 'test_results_'+arch
                        path = path + '_iou' + str(args.iou_thresh).replace('.', '') + '.pkl'
                        result = {"precision":precision, "recall":recall, "avg":avg, "confs":confs}
                        with open(path, "wb") as f:
                            pickle.dump(result, f)
    elif args.mode == "video":
        save_path = "./results/rcnn/latent/one_horse/zju377/advtexture-300.pth"
        print(save_path)
        latent_dict = torch.load(save_path, map_location='cpu')
        adv_latents = latent_dict["latent"].to(trainer.device).to(trainer.weight_type)
        trainer.generate_video(adv_latents, conf_thresh=0.5, iou_thresh=0.1)