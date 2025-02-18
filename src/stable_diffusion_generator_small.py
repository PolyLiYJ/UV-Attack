from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from omegaconf import OmegaConf
from packaging import version
from typing import Optional, Any, Dict

@dataclass
class Config():
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    # pretrained_model_name_or_path: str = "/home/yjli/AIGC/Adversarial_camou/pretained_model"
    compute_unet_grad: bool = False
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    max_step_percent_annealed: float = 0.5
    anneal_start_step: Optional[int] = None  # Added type hint


    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"
    num_images_per_prompt: int = 1

    token_merging: bool = False
    token_merging_params: Dict[str, Any] = field(default_factory=dict)  # Corrected with type annotation

    """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
    max_items_eval: int = 4
    
    cache_dir: Optional[str] = None

    guidance_scale: float = 7.5
    do_classifier_free_guidance: bool = True

    fixed_size: int = -1
    diffusion_steps: int = 20

    use_sds: bool = False
        
class StableDiffusionGenerator():
    def __init__(self, cfg, device):
        self.device = device 
        self.cfg = OmegaConf.structured(Config(**cfg)) #merge the configuation
        self.configure()

    def configure(self) -> None:

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        if self.cfg.pretrained_model_name_or_path[-4:]=="ckpt":
            self.pipe = StableDiffusionPipeline.from_single_file(
                self.cfg.pretrained_model_name_or_path,
                torch_dtype=self.weights_dtype).to(self.device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                torch_dtype=self.weights_dtype).to(self.device)
            
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            torch_version = version.parse(torch.__version__)

            if torch_version >= version.parse("2.0"):
                print("PyTorch2.0 uses memory efficient attention by default.")
                
            elif not is_xformers_available():
                print("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # # Create model
        self.pipe.vae = self.pipe.vae.eval()
        self.pipe.unet = self.pipe.unet.eval()

        self.scheduler.config.num_train_timesteps = 1000
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val= None

        print(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ) :
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs
    ):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.pipe.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        return latents.to(input_dtype)


    # @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        # latents = F.interpolate(
        #     latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        # )
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

    def edit_image(
        self,
        image,
        prompt: str,
        negative_prompt: str, 
        diffusion_rate: float = 0.3, 
        **kwargs,
    ):   

        batch_size = image.shape[0]
        # Encode input prompt

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            self.device,
            self.cfg.num_images_per_prompt,
            self.cfg.do_classifier_free_guidance,
            negative_prompt,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.cfg.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device, **kwargs)
        # only backward harf of diffusion times
        timesteps = self.scheduler.timesteps[int(self.cfg.diffusion_steps*(1-diffusion_rate)):]

        # Prepare latent variables
        latents = self.encode_images(image) # latent_codes: "B 4 64 64"
        batch_size = image.shape[0]
        latents = latents.to(self.device)
        noise = torch.randn_like(latents).requires_grad_(False)
        latents = self.scheduler.add_noise(latents, noise, torch.tensor(self.num_train_timesteps*diffusion_rate).to(torch.long))  # type: ignore

        # Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
                return_dict=False,
            )[0]

            # perform guidance
            if self.cfg.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # latents = latents + latent_noise
        image = self.decode_latents(latents)
        return image
    
    def edit_latent(
        self,
        latents,
        prompt: str,
        negative_prompt: str, 
        **kwargs,
    ):   
        # Encode input prompt
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                self.device,
                self.cfg.num_images_per_prompt,
                self.cfg.do_classifier_free_guidance,
                negative_prompt,
            )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.cfg.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device, **kwargs)
        # only backward harf of diffusion times
        timesteps = self.scheduler.timesteps

        # Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
                return_dict=False,
            )[0]

            # perform guidance
            if self.cfg.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        image = self.decode_latents(latents)
        return image
    
    def edit_middlelatent(
        self,
        latents,
        noise,
        prompt: str,
        negative_prompt: str, 
        add_layer = 5,
        **kwargs,
    ):   
        # Encode input prompt
                # Encode input prompt
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                self.device,
                self.cfg.num_images_per_prompt,
                self.cfg.do_classifier_free_guidance,
                negative_prompt,
            )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.cfg.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device, **kwargs)
        # only backward harf of diffusion times
        timesteps = self.scheduler.timesteps

        # Denoising loop
        for i, t in enumerate(timesteps):
            if i < self.cfg.diffusion_steps - add_layer:
                latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                with torch.no_grad():
                    # expand the latents if we are doing classifier free guidance
                    noise_pred = self.pipe.unet(
                            latent_model_input.to(self.weights_dtype),
                            t.to(self.weights_dtype),
                            encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
                            return_dict=False,
                        )[0]
            if i >= self.cfg.diffusion_steps - add_layer:
                if i == self.cfg.diffusion_steps - add_layer:
                    latents = latents + noise
                latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.pipe.unet(
                    latent_model_input.to(self.weights_dtype),
                    t.to(self.weights_dtype),
                    encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
                    return_dict=False,
                )[0]

            # perform guidance
            if self.cfg.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        image = self.decode_latents(latents)
        return image
    
      
if __name__ == "__main__":
    '''
    Only backward part of the diffusion model. 
    The backward times = diffusion_rate * diffusion_steps
    
    miniSD is for 256*256, stable-diffusion is for 512*512, so when change model, 
    the desired_size also need to be changed
    '''
    device = torch.device("cuda:1")
    desired_size = (256, 256)
    # the backward times = diffusion_rate * diffusion_steps
    diffusion_rate = 0.5
    diffusion_steps = 30
    # "pretrained_model_name_or_path": "./pretained_model/miniSD.ckpt", # "runwayml/stable-diffusion-v1-5"
    cfg = {"pretrained_model_name_or_path": "./pretained_model/miniSD.ckpt",
            "diffusion_steps": diffusion_steps,
            "guidance_scale": 7.5,
            "min_step_percent": 0.02,
            "max_step_percent": 0.98}
    stable_diffusion_generator = StableDiffusionGenerator(cfg, device)
    # Read a single image
    from PIL import Image
    import numpy as np
    import os
    image_path = "/home/yjli/AIGC/Adversarial_camou/results/experiment/three dogs, three cats and three trees/texture-diffusion-three dogs, three cats and three trees-best-iter88.png"
    # image_path = "/home/yjli/AIGC/Adversarial_camou/results/clothes/PNG/Front.png"#
    image = Image.open(image_path)
    # Define the desired image size for Stable Diffusion
    H = desired_size[0]
    W = desired_size[1]
    # Resize the image
    resized_image = image.resize(desired_size)
    channels = len(resized_image.split())
    # Convert the image to tensor and normalize pixel values
    if channels == 4: 
        resized_image = resized_image.convert("RGB")

    preprocessor = VaeImageProcessor(do_resize = False, vae_scale_factor=8)
    input_image = preprocessor.preprocess(resized_image)
    print(input_image.shape)
    # Convert the image tensor to the appropriate data type
    input_image = input_image.to(torch.float16).to(device)
    RH, RW = H // 8 * 8, W // 8 * 8 # Let's make it a multiple of 8
    rgb_BCHW_HW8 = F.interpolate(
        input_image, (RH, RW), mode="bilinear", align_corners=False
    )
    prompt = "a cat"
    neg_prompt= "unrealistic"
    pretrained_model_name_or_path = cfg["pretrained_model_name_or_path"]
    # with torch.no_grad():
    print(rgb_BCHW_HW8.shape)
    output = stable_diffusion_generator(rgb_BCHW_HW8, prompt, neg_prompt, diffusion_rate=diffusion_rate)
    print(output.shape)
    width  = desired_size[0]
    height = desired_size[1]
    os.makedirs(f"results/experiment/{width}x{height}", exist_ok=True)
    # Create a new blank image with double width to accommodate both images side by side

    new_image = Image.new('RGB', (width * 2, height))

    # Paste the first image on the left side
    new_image.paste(resized_image, (0, 0))

    # Convert the tensor to a NumPy array
    output_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)  # Scale values to 0-255 range

    # Create a PIL Image from the NumPy array
    output_image = Image.fromarray(output_np)

    new_image.paste(output_image, (width, 0))

    # Save the concatenated image
    new_image.save(f"results/experiment/{width}x{height}/edit_image_cat{width}x{height}.jpg")
    print("save at",f"results/experiment/{width}x{height}/edit_image_cat{width}x{height}.jpg" )
    

