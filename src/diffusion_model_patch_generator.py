"""
        python diffusion_model_patch_generator.py --prompt "three bear" --pattern_mode "repeat" --checkpoints 210 --lr 0.005 --device cuda:0 --do_classifier_free_guidance False --half_precision_weights True

        python diffusion_model_patch_generator.py --prompt "two bears" --pattern_mode "repeat" --lr 0.01 --device cuda:1 --do_classifier_free_guidance True --half_precision_weights True --batch_size 4 --checkpoints 520 --test

        python diffusion_model_patch_generator.py --prompt "colorful repeated patterns" --pattern_mode "whole" --lr 0.01 --device cuda:2 --do_classifier_free_guidance False --half_precision_weights False --batch_size 4 --checkpoints 90

        python diffusion_model_patch_generator.py --optimize_type image --diffusion_rate 0.3 --diffusion_steps 20 --prompt "three bears" --pattern_mode "repeat" --lr 0.001 --device cuda:3 --do_classifier_free_guidance True --half_precision_weights False --batch_size 2 --checkpoints 0  
"""

# LPIPS loss 

import torch
import os
from stable_diffusion_generator_small import StableDiffusionGenerator
from diffusers.utils.torch_utils import randn_tensor
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d_modify import MyHardPhongShader
from diffusers.image_processor import VaeImageProcessor
import lpips

#import patch_config
import sys
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
from tps import *
from transformers import DeformableDetrForObjectDetection
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
import os
from diffusers.utils.torch_utils import randn_tensor

from pytorch3d.renderer import (
    cameras,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    BlendParams,
    TexturesUV
)

# add path for demo utils functions 
sys.path.append(os.path.abspath(''))


from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from train_util import *
import pytorch3d_modify as p3dmd
import mesh_utils as MU

class PatchTrainer(object):
    def __init__(self, args):
        self.args = args
        if args.device is not None:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        else:
            device = None
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"

        if args.arch == "rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        elif args.arch == "yolov3":
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
        elif args.arch == "deformable-detr":
            self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        else:
            raise NotImplementedError
        
        self.model.eval()
        
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
        self.tv_loss = TotalVariation()

        self.alpha = args.alpha
        self.azim = torch.zeros(self.batch_size)
        self.blend_params = None

        self.sampler_probs = torch.ones([36]).to(device)
        self.loss_history = torch.ones(36).to(device)
        self.num_history = torch.ones(36).to(device)

        self.train_loader = self.get_loader('./data/background', True)
        self.test_loader = self.get_loader('./data/background_test', True)

        self.epoch_length = len(self.train_loader)
        print(f'One training epoch has {len(self.train_loader.dataset)} images')
        print(f'One test epoch has {len(self.test_loader.dataset)} images')

        color_transform = ColorTransform('color_transform_dim6.npz')
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

        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)
        self.colors = torch.load("data/camouflage4.pth").float().to(device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]
        
        self.prompts = "A cat"
        # prompt = "add some cartoon patterns inside the green box",
        # self.negative_prompts = "cartoon, unrealistic, single colors, high varaince, too complicated"
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

        # self.stable_diffusion_model_trouser = StableDiffusionGenerator(self.sd_cfg, device)

        #self.initialize_tps2d()
        #self.initialize_tps3d()
           
    def prepare_latents(self, batch_size, num_channels_latents, height=256, width=256, dtype=torch.float16, device=torch.device("cuda")):
        vae_scale_factor = 8
        # generator = torch.Generator(device='cuda') # ge the generated seed
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
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

    def initialize_tps2d(self):
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), map_location='cpu').to(self.device)
        self.infos_tshirt = MU.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)

        locations_trouser_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/trouser_join/projections/part_all_off3p4.pt'), map_location='cpu').to(self.device)
        self.infos_trouser = MU.get_map_kernel(locations_trouser_ori, self.faces_uvs_trouser)

        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu())
        tps2d_tshirt.to(self.device)
        self.tps2d_tshirt = tps2d_tshirt

        target_control_points = p3dmd.get_points(self.trouser_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_trouser = TPSGridGen(None, target_control_points, locations_trouser_ori.cpu())
        tps2d_trouser.to(self.device)
        self.tps2d_trouser = tps2d_trouser
        return

    def initialize_tps3d(self):
        xmin, ymin, zmin = (-0.28170400857925415, -0.7323740124702454, -0.15313300490379333)
        xmax, ymax, zmax = (0.28170400857925415, 0.5564370155334473, 0.0938199982047081)
        xnum, ynum, znum = [5, 8, 5]
        max_range = (torch.Tensor([xmax, ymax, zmax]) - torch.Tensor([xmin, ymin, zmin])) / torch.Tensor(
            [xnum, ynum, znum])
        self.max_range = (max_range * self.args.tps3d_range).tolist()
        target_control_points = torch.tensor(list(itertools.product(
            torch.linspace(xmin, xmax, xnum),
            torch.linspace(ymin, ymax, ynum),
            torch.linspace(zmin, zmax, znum),
        )))
        mesh = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])

        tps3d = TPSGridGen(None, target_control_points, mesh.verts_packed().cpu())
        tps3d.to(self.device)
        self.tps3d = tps3d
        return

    def synthesis_image(self, background_batch, use_tps2d=True, use_tps3d=True):
        '''
        if use_tps2d:
            # tps_2d
            source_control_points_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi / 180 * args.tps2d_range_t, args.tps2d_range_r,
                                                            bs=self.batch_size, random=True)
            locations_tshirt = self.tps2d_tshirt(source_control_points_tshirt.to(self.device))
            source_control_points_trouser = p3dmd.get_points(self.trouser_locations_infos, torch.pi / 180 * args.tps2d_range_t, args.tps2d_range_r,
                                                             bs=self.batch_size, random=True)
            locations_trouser = self.tps2d_trouser(source_control_points_trouser.to(self.device))
        else:
            locations_tshirt = locations_trouser = None

        if use_tps3d:
            source_coordinate = self.tps3d.tps_mesh(max_range=self.max_range, batch_size=self.batch_size).view(-1, 3)
        else:
            source_coordinate = None
        '''
        # render images
        humanmesh = join_meshes_as_scene([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])
        batch_humanmesh = []
        for i in range(self.batch_size):
            batch_humanmesh.append(humanmesh)
        batch_humanmesh = join_meshes_as_batch(batch_humanmesh)
        if isinstance(self.cameras, list) or isinstance(self.cameras, tuple):
            R, T = look_at_view_transform(*self.cameras, up=((0, 1, 0),))
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        else:
            cameras = self.cameras
            
        raster_settings = RasterizationSettings(
            image_size = 500, 
            blur_radius = 0.0, 
            faces_per_pixel = 3, 
            max_faces_per_bin = 30000
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
           shader = MyHardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )
        )
        images_predicted = renderer(batch_humanmesh)
        adv_batch = images_predicted.permute(0, 3, 1, 2)
        # combine the background and human
        p_background_batch, gt = self.patch_transformer(background_batch, adv_batch) #gt is the ground truth boxes
        return p_background_batch, gt 

    def update_mesh_via_latent(self, latents, prompts, negative_prompts, texture = None):
        # camouflage:
        if texture is None:
            tex_texture  = self.stable_diffusion_model.edit_latent(latents, prompts,negative_prompts)
            # tex_trouser  = self.stable_diffusion_model_trouser(prompts,negative_prompts, latent_trouser, latent_noises_trouser)
            tex_texture  = self.color_transform(tex_texture)
        else:
            tex_texture = texture
            
        if self.args.pattern_mode == "repeat":
            tex_texture_repeat = F.interpolate(tex_texture, (100, 100), mode="bilinear").repeat(1,1, 10,10)
        else:
            tex_texture_repeat = F.interpolate(tex_texture, (1024, 1024), mode="bilinear")
        
        tex_tshirt  = tex_texture_repeat[:,:,0:340,0:864]
        tex_trouser  = tex_texture_repeat[:,:,340:340+484,0:700]

        tex_tshirt = tex_tshirt.permute(0, 2, 3, 1) #[1,340, 864, 3]
        tex_trouser = tex_trouser.permute(0, 2, 3, 1)#[1, 484, 700, 3]
        tex_texture = tex_texture.permute(0, 2, 3, 1)
    
        self.mesh_tshirt.textures = TexturesUV(maps=tex_tshirt, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)
        return tex_tshirt, tex_trouser, tex_texture
    
    def update_mesh_via_latentnoise(self, latents, noise, prompts, negative_prompts):
        # camouflage:
        tex_texture  = self.stable_diffusion_model.edit_middlelatent(latents, noise, prompts, negative_prompts)
        # tex_trouser  = self.stable_diffusion_model_trouser(prompts,negative_prompts, latent_trouser, latent_noises_trouser)
        # tex_texture = F.interpolate(tex_texture, (128, 128), mode="bilinear")
        # tex_texture_repeat  = self.color_transform(tex_texture).repeat(1,1,7,7)
        tex_texture_transformed  = self.color_transform(tex_texture)
        
        if self.args.pattern_mode == "repeat":
            tex_texture_repeat = F.interpolate(tex_texture_transformed, (128, 128), mode="bilinear").repeat(1,1, 7, 7)
        else:
            tex_texture_repeat = F.interpolate(tex_texture_transformed, (1024, 1024), mode="bilinear")
        # plt.figure(figsize=(10, 10))
        # plt.subplot(111)
        # plt.imsave(f"results/experiment/texture-diffusion.png", tex_texture_repeat.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
        
        tex_tshirt  = tex_texture_repeat[:,:,0:340,0:864].permute(0, 2, 3, 1) 
        tex_trouser  = tex_texture_repeat[:,:,340:824,0:700].permute(0, 2, 3, 1)
        tex_texture_permuted = tex_texture_transformed.permute(0, 2, 3, 1)
    
        self.mesh_tshirt.textures = TexturesUV(maps=tex_tshirt, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)
        return tex_tshirt, tex_trouser, tex_texture_permuted
     
    def update_mesh_via_image(self, adv_image, prompts, negative_prompts):
        # camouflage:
        tex_texture  = self.stable_diffusion_model.edit_image(adv_image, prompts,negative_prompts, diffusion_rate = self.diffusion_rate)
        # tex_trouser  = self.stable_diffusion_model_trouser(prompts,negative_prompts, latent_trouser, latent_noises_trouser)
        # plt.figure(figsize=(10, 10))
        # plt.subplot(111)
        # plt.imsave(f"results/experiment/texture-diffusion.png", tex_texture.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
        tex_texture = F.interpolate(tex_texture, (128, 128), mode="bilinear")
        tex_texture_repeat  = self.color_transform(tex_texture).repeat(1, 1, 7, 7)
        
        tex_tshirt  = tex_texture_repeat[:,:,0:340,0:864]
        tex_trouser  = tex_texture_repeat[:,:,340:340+484,0:700]

        tex_tshirt = tex_tshirt.permute(0, 2, 3, 1) #[1,340, 864, 3]
        tex_trouser = tex_trouser.permute(0, 2, 3, 1)#[1, 484, 700, 3]
        tex_texture = tex_texture.permute(0, 2, 3, 1)

        self.mesh_tshirt.textures = TexturesUV(maps=tex_tshirt, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)
        return tex_tshirt, tex_trouser, tex_texture

    # compute the KL divergence between the adverasarial dirstribution and standard normal distruibution
    def compute_kl_divergence(self, mu, sigma):
        kl_loss = 0.5 * (torch.norm(mu, p=2) - torch.log(torch.sum(sigma)) + torch.sum(sigma))
        return kl_loss

    def load_weights(self, save_path, epoch):
        path = save_path + '/advtexture' + str(epoch) + '.pth'
        latent_dict = torch.load(path, map_location='cpu')
        # init_tex_texture = latent_dict["tex_texture"]
        # if "adv_input_image" in latent_dict.keys():
        #     adv_data = latent_dict["adv_input_image"]
        # if "latent" in latent_dict.keys():
        #     adv_data = latent_dict["latent"]
        # if "latent_noise" in latent_dict.keys():
        #     adv_data = latent_dict["latent_noise"]
        # adv_data= adv_data.to(self.device)

        path = save_path + '/' + str(epoch) + 'info.npz'
        if os.path.exists(path):
            x = np.load(path)
            self.loss_history = torch.from_numpy(x['loss_history']).to(self.device)
            self.num_history = torch.from_numpy(x['num_history']).to(self.device)
        return latent_dict

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        # the latent code to generate the adversarial pattern
        #self.prompts = "colorful repeated patterns"
        self.prompts = self.args.prompt
        # prompt = "add some cartoon patterns inside the green box",
        # self.negative_prompts = "cartoon, unrealistic, single colors, high varaince, too complicated"
        self.negative_prompts = "high varaince"
        
        checkpoints = self.args.checkpoints
        if checkpoints > 0:
            latent_dict = self.load_weights(self.args.save_path, checkpoints)
            print("load weights from ", self.args.save_path)
            
            
           
        if self.args.optimize_type == "latent":
            init_latents = self.prepare_latents(batch_size=1, num_channels_latents=4, 
                                        height=256, width=256, device=self.device, 
                                        dtype = self.weight_type)
            adv_latents = torch.autograd.Variable(init_latents.type(self.weight_type), requires_grad=True).to(self.device)
            if checkpoints > 0:
                latent_dict = self.load_weights(self.args.save_path, checkpoints)
                adv_latents.data = latent_dict["latent"].to(self.device).to(self.weight_type)
                
                first_latent_dict = self.load_weights(self.args.save_path, 0)
                first_latents = first_latent_dict["latent"].to(self.device).to(self.weight_type)
                _, _, original_tex_texture = self.update_mesh_via_latent(first_latents,
                                                    self.prompts, self.negative_prompts)
                original_tex_texture = original_tex_texture.permute(0, 3, 1, 2).detach().clone().requires_grad_(False)   
            else:
                _, _, original_tex_texture = self.update_mesh_via_latent(adv_latents.detach().clone(),
                                                    self.prompts, self.negative_prompts)
                original_tex_texture = original_tex_texture.permute(0, 3, 1, 2).detach().clone().requires_grad_(False)    
                
            self.optimizer = torch.optim.SGD([adv_latents], lr=self.args.lr, momentum=0.9)
            
        elif self.args.optimize_type == "latent_noise":
            init_latents = self.prepare_latents(batch_size=1, num_channels_latents=4, 
                                        height=256, width=256, device=self.device,
                                        dtype = self.weight_type
                                        ).requires_grad_(False)
            adv_latents_noise = torch.zeros_like(init_latents).requires_grad_(True)
            if checkpoints > 0:
                init_latents.data = latent_dict["init_latents"].to(self.device).to(self.weight_type)
                adv_latents_noise.data = latent_dict["latent_noise"].to(self.device).to(self.weight_type)
            self.optimizer = torch.optim.SGD([adv_latents_noise], lr=0.001, momentum=0.9)
            
        elif self.args.optimize_type == "image":
            input_image = self.prepare_image(image_path = "./results/yolov3_07/image/ThreeBears.jpg")
            adv_input_image = input_image.detach().clone().requires_grad_(True)
            if checkpoints > 0:
                adv_input_image.data = latent_dict["adv_input_image"].to(self.device).to(self.weight_type)
            self.optimizer = torch.optim.SGD([adv_input_image], lr=0.001)
        
        et0 = time.time()
       
        self.writer = self.init_tensorboard()
        args = self.args
        
        for epoch in tqdm(range(checkpoints, args.nepoch)):
            print('######################################')
            ep_det_loss = 0
            ep_norm_loss = 0
            ep_loss = 0
            ep_mean_prob = 0
            ep_tv_loss = 0
            ep_ctrl_loss = 0
            ep_seed_loss = 0
            ep_log_likelihood = 0
            ep_lpips_loss = 0
            eff_count = 0  # record how many images in this epoch are really in training so that we can calculate accurate loss

            self.sampler_probs = self.loss_history / self.num_history
            if epoch % 100 == 0:
                print(self.sampler_probs) # sample cameras
            self.loss_history = self.loss_history / 2 + 1e-5
            self.num_history = self.num_history / 2 + 1e-5
            
            if epoch % 100 == 99:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * args.lr_decay

            #self.optimizer.zero_grad()
            
            # theta = 0
            # self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]]) 

            for i_batch, background_batch in enumerate(self.train_loader):
                background_batch = background_batch.to(self.device)
                t0 = time.time()
                # AG step
                self.optimizer.zero_grad()
                
                if i_batch % 10 == 0:
                    self.sample_cameras()
                    self.sample_lights()
                  
                if self.args.optimize_type == "latent":
                    tex_tshirt, tex_trouser, tex_texture = self.update_mesh_via_latent(adv_latents,
                                                    self.prompts, self.negative_prompts)

                    
                elif self.args.optimize_type == "latent_noise":
                    tex_tshirt, tex_trouser, tex_texture = self.update_mesh_via_latentnoise(init_latents, adv_latents_noise,
                                                self.prompts, self.negative_prompts)
                elif self.args.optimize_type == "image":
                    tex_tshirt, tex_trouser, tex_texture = self.update_mesh_via_image(adv_input_image,
                                                self.prompts, self.negative_prompts)

                p_background_batch, gt = self.synthesis_image(background_batch, False, False)
                
                # plt.figure(figsize=(10, 10))
                # plt.subplot(121)
                # plt.imsave(args.save_path + f"/texture_tshirt-{epoch}-{i_batch}.png", tex_texture[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                # print(args.save_path + f"/texture_tshirt-{epoch}-{i_batch}.png")
                # plt.subplot(122)
                # plt.imsave(args.save_path + f"/p_background_batch-{epoch}-{i_batch}.png", p_background_batch.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                # print(args.save_path + f"/p_background_batch-{epoch}-{i_batch}.png")
                
                t1 = time.time()
                normalize = True
                if self.args.arch == "deformable-detr" and normalize:
                    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                    p_background_batch = normalize(p_background_batch)
                    
                output = self.model(p_background_batch)

                t2 = time.time()
                try:
                    det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                    eff_count += 1
                except RuntimeError:  # current batch of imgs have no bbox be detected
                    continue
                t3 = time.time()
                if self.azim_inds is not None:
                    self.loss_history.index_put_([self.azim_inds], max_prob_list.detach(), accumulate=True)
                    self.num_history.index_put_([self.azim_inds], torch.ones_like(max_prob_list), accumulate=True)
                ''''''
                tv_loss = torch.tensor([0])
                norm_loss = torch.tensor([0])

                loss = det_loss
                if args.tv_loss > 0:
                    tv_loss = self.tv_loss(tex_tshirt) +  self.tv_loss(tex_trouser)
                    loss += tv_loss * args.tv_loss

                if self.args.optimize_type == "image":
                    norm_loss = torch.norm(adv_input_image - input_image, 2)
                    loss += 0.01*norm_loss
                    
                lpips_loss = 0.1 * self.lpips_loss(tex_texture.permute(0, 3, 1, 2), original_tex_texture)
                loss +=  lpips_loss.mean()
                
                # kl_loss = self.compute_kl_divergence(mu_tshirt, sigma_tshirt)+ self.compute_kl_divergence(mu_trousers, sigma_trousers)
                
                ep_mean_prob += max_prob_list.mean().item()
                ep_det_loss += det_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_norm_loss += norm_loss.item()
                ep_loss += loss.item()
                
                loss.backward()
                if self.args.optimize_type == "latent":
                    torch.nn.utils.clip_grad_norm_(adv_latents, 0.1)
                elif self.args.optimize_type == "latent_noise":
                    torch.nn.utils.clip_grad_norm_(adv_latents_noise, 0.1)

                self.optimizer.step()
                t4 = time.time()

                if i_batch % 20 == 0:
                    print("iteration", i_batch, "loss", loss.detach().cpu().numpy(),
                          "det_loss", det_loss.detach().cpu().numpy(),
                          "norm_loss", norm_loss.detach().cpu().numpy(),
                          'tv_loss', tv_loss.detach().cpu().numpy(),
                          "lpips_loss", lpips_loss.detach().cpu().numpy(),
                          'time:', t4-t0)

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
                self.writer.add_scalar('epoch/total_loss', ep_loss, epoch)
                self.writer.add_scalar('epoch/norm_loss', ep_norm_loss, epoch)
                self.writer.add_scalar('epoch/tv_loss', ep_tv_loss, epoch)
                self.writer.add_scalar('epoch/det_loss', ep_det_loss, epoch)
                self.writer.add_scalar('epoch/lpips_loss', ep_lpips_loss, epoch)
                self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
                
            et0 = time.time()
            
            if (epoch) % 5 == 0:
                plt.figure(figsize=(10, 10))
                plt.subplot(111)
                plt.imsave(args.save_path + f"/texture_tshirt-{epoch}.png", tex_texture[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                fig = plt.figure(figsize=(10, 10))
                plt.subplot(111)
                plt.imshow(p_background_batch.permute(0,2,3,1)[0, :, :, :3].clamp(0,1).detach().cpu().numpy())
                # plt.subplot(222)
                # plt.imshow(p_background_batch.permute(0,2,3,1)[1, :, :, :3].clamp(0,1).detach().cpu().numpy())
                # plt.subplot(223)
                # plt.imshow(p_background_batch.permute(0,2,3,1)[2, :, :, :3].clamp(0,1).detach().cpu().numpy())
                # plt.subplot(224)
                # plt.imshow(p_background_batch.permute(0,2,3,1)[3, :, :, :3].clamp(0,1).detach().cpu().numpy())
                fig.savefig(args.save_path + f"/p_background_batch-{epoch}.png")

            if (epoch) % 5 == 0:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                if self.args.optimize_type == "latent":
                    latent_dict = {
                                "tex_texture": tex_texture.detach().cpu(),
                                "latent": adv_latents.detach().cpu()
                                }
                elif  self.args.optimize_type == "latent_noise":
                    latent_dict = {
                                "tex_texture": tex_texture.detach().cpu(),
                                "init_latents": init_latents.detach().cpu(),
                                "latent_noise": adv_latents_noise.detach().cpu()
                                }
                else:
                    latent_dict = {
                                "tex_texture": tex_texture.detach().cpu(),
                                "adv_input_image": adv_input_image.detach().cpu()
                                }
                path = args.save_path + '/advtexture' + str(epoch) + '.pth'
                torch.save(latent_dict, path)

                path = args.save_path + '/' + str(epoch) + 'info.npz'
                np.savez(path, loss_history=self.loss_history.cpu().numpy(), num_history=self.num_history.cpu().numpy(), azim=self.azim.cpu().numpy())
            if (epoch) % 5 == 0:
                with torch.no_grad():
                    precision, recall, avg, confs, thetas = trainer.test(conf_thresh=0.01, iou_thresh=args.test_iou, angle_sample=37, use_tps2d=False, use_tps3d=False, mode=args.test_mode)
                    asr = (confs < 0.5).mean()
                    print("TEST ASR:", asr)
                    self.writer.add_scalar('epoch/test-ASR', asr, epoch)
            
            """
            if (epoch + 1) % 300 == 0:
                self.update_mesh(type='determinate')
                for iou_thresh in [0.01, 0.1, 0.3, 0.5]:
                    precision, recall, avg, confs, thetas = self.test(conf_thresh=0.01, iou_thresh=iou_thresh, angle_sample=37, use_tps2d=not args.disable_test_tps2d, use_tps3d=not args.disable_test_tps3d, mode=args.test_mode)
                    info = [precision, recall, avg, confs]
                    path = args.save_path + '/' + str(epoch) + 'test_results_tps'
                    path = path + '_iou' + str(iou_thresh).replace('.', '') + '_' + args.test_mode
                    path = path + '.npz'
                    np.savez(path, thetas=thetas, info=info)
            """

    def test(self, iou_thresh, num_of_samples=100, angle_sample=37, use_tps2d=True, use_tps3d=True, mode='person'):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        print(f'One test epoch has {len(self.test_loader.dataset)} images')

        thetas_list = np.linspace(-180, 180, angle_sample)
        confs = [[] for i in range(angle_sample)]
        self.sample_lights(r=0.1)

        total = 0.
        positives = []
        et0 = time.time()
        with torch.no_grad():
            j = 0
            for i_batch, background_batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), position=0):
                background_batch = background_batch.to(self.device)
                for it, theta in enumerate(thetas_list):
                    self.sample_cameras(theta=theta)
                    p_background_batch, gt = self.synthesis_image(background_batch, use_tps2d, use_tps3d)

                    normalize = True
                    if self.args.arch == "deformable-detr" and normalize:
                        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        p_background_batch = normalize(p_background_batch)

                    output = self.model(p_background_batch)
                    total += len(p_background_batch)  # since 1 image only has 1 gt, so the total # gt is just = the total # images
                    pos = []
                    # for i, boxes in enumerate(output):  # for each image
                    conf_thresh = 0.0 if self.args.arch in ['rcnn'] else 0.1
                    person_cls = 0
                    output = adv_camou_utils.get_region_boxes_general(output, self.model, conf_thresh=conf_thresh, name=self.args.arch)

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
                    confs[it].extend([p[0] if p[1] else 0.0 for p in pos])
       

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

        return precision, recall, avg, confs, thetas_list


if __name__ == '__main__':
    print('Version 2.0')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--alpha", type=float, default=10, help='')
    parser.add_argument("--tv_loss", type=float, default=0, help='')
    parser.add_argument("--lr_decay", type=float, default=0.5, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--blur", type=float, default=1, help='')
    parser.add_argument("--like", type=float, default=1, help='')
    parser.add_argument("--ctrl", type=float, default=1, help='')
    parser.add_argument("--arch", type=str, default="rcnn")
    parser.add_argument("--cdist", type=float, default=0, help='')
    parser.add_argument("--seed_type", default='fixed', help='fixed, random, variable, langevin')
    parser.add_argument("--rd_num", type=int, default=200, help='')
    parser.add_argument("--clamp_shift", type=float, default=0, help='')
    parser.add_argument("--resample_type", default=None, help='')
    parser.add_argument("--seed_temp", type=float, default=1.0, help='')
    parser.add_argument("--seed_opt", default='adam', help='')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0, help='')
    parser.add_argument("--tps2d_range_r", type=float, default=0.1, help='')
    parser.add_argument("--tps3d_range", type=float, default=0.15, help='')
    parser.add_argument("--disable_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--seed_ratio", default=1.0, type=float, help='The ratio of trainable part when seed type is variable')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--test", default=True, action='store_true', help='')
    parser.add_argument("--test_iou", type=float, default=0.1, help='')
    parser.add_argument("--test_nms_thresh", type=float, default=1.0, help='')
    parser.add_argument("--test_mode", default='person', help='person, all')
    parser.add_argument("--test_suffix", default='', help='')
    parser.add_argument("--train_iou", type=float, default=0.01, help='')
    
    parser.add_argument('--device', default='cuda:3', help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=600, help='')
    parser.add_argument('--checkpoints', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--save_path', default='/home/yjli/AIGC/Adversarial_camou/results/', help='')
    parser.add_argument("--optimize_type", type=str, default="latent", help='image , latent_noise or latent')
    parser.add_argument("--diffusion_rate", type=float, default=0.5, help='')
    parser.add_argument("--diffusion_steps", type=int, default=10, help="")
    parser.add_argument("--do_classifier_free_guidance", default=False, help='')
    parser.add_argument("--half_precision_weights", default=False, help='')
    parser.add_argument("--prompt", default="one horse", help='three bear, colorful repeated patterns')
    parser.add_argument("--pattern_mode", type=str, default="repeat", help='repeat, whole (for colorful repeated patterns)')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default= "/home/yjli/AIGC/Adversarial_camou/pretained_model/miniSD.ckpt",
                        help = "if use stablediffusion with 512*512, change to runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()
    assert args.seed_type in ['fixed', 'random', 'variable', 'langevin']

    # torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Train info:", args)
    args.save_path = args.save_path + "/" +  args.arch + "/" + args.optimize_type + "/" + args.prompt
    os.makedirs(args.save_path, exist_ok=True)
    print("save directory:", args.save_path)
    if not args.test:
        import json
        trainer = PatchTrainer(args)
        with open(args.save_path + "/CONFIG.txt", "w") as outfile: 
            json.dump(args.__dict__, outfile, indent=2)
        trainer.train()
    else:
        # train on rcnn, test on args.arc (eg. yolov3): a
        arch_list = ["deformable-detr"]
        save_path = "results/rcnn/latent/one horse/advtexture-Peter_chess-460.pth"
        # save_path = "/home/yjli/AIGC/Adversarial_camou/results/rcnn/latent/three bear/advtexture265.pth"
        # save_path = "/home/yjli/AIGC/Adversarial_camou/results/rcnn/latent/one horse/advtexture285.pth"
        print(save_path)
        count = 0
        for arch in arch_list:
            args.arch = arch
            args.device = f'cuda:{count}'
            count = count + 1
            trainer = PatchTrainer(args)  

            latent_dict = torch.load(save_path, map_location='cpu')
            adv_latents = latent_dict["latent"].to(trainer.device).to(trainer.weight_type)
                
            # Load the image using PIL
            # image = Image.open('./results/yolov3_07/latents/three bear/texture_tshirt-1.png').convert("RGB")
            # Define the transformation to convert the image to a tensor
            # transform = transforms.ToTensor()
            # Apply the transformation to the image
            # device = torch.device(args.device)
            # tensor_image = transform(image).unsqueeze(0).to(device)
            # trainer.update_mesh_via_latent(None, None, None, texture = tensor_image)
            tex_tshirt, tex_trouser, tex_texture = trainer.update_mesh_via_latent(adv_latents,
                                                        trainer.prompts, trainer.negative_prompts)
            with torch.no_grad():
                precision, recall, avg, confs, thetas = trainer.test(iou_thresh=args.test_iou, angle_sample=37, use_tps2d=False, use_tps3d=False, mode=args.test_mode)
            print(args.arch + " test ASR:", (confs < 0.5).mean())
            # info = [precision, recall, avg, confs]
            # path = args.save_path + '/' + str(epoch) + 'test_results_tps'
            # path = path + '_iou' + str(args.test_iou).replace('.', '') + '_' + args.test_mode + args.test_suffix
            # path = path + '.npz'
            # np.savez(path, thetas=thetas, info=info)