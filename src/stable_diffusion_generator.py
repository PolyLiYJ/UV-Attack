from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from typing import Optional, Any, Dict
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
from omegaconf import OmegaConf

@dataclass
class Config():
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    compute_unet_grad: bool = False
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = False

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    max_step_percent_annealed: float = 0.5
    anneal_start_step=None

    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"
    num_images_per_prompt: int = 1

    token_merging: bool = False
    token_merging_params: Optional[dict] =field(default_factory=dict)

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
        threestudio.info(f"Loading Stable Diffusion ...")

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

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # # Create model
        # self.vae = self.pipe.vae.eval()
        # self.unet = self.pipe.unet.eval()
        # self.vae = self.pipe.vae
        # self.unet = self.pipe.unet

        # for p in self.vae.parameters():
        #     p.requires_grad_(False)
        # for p in self.unet.parameters():
        #     p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            threestudio.debug("Start editing...")
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_latents], dim=1
                    )

                    noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                )

                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs] 

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

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

    
    def __call__(
        self,
        # latents : Float[Tensor, "B 4 DH DW"],
        # cond_latents: Float[Tensor, "B H RW RC"],
        # text_embeddings:  Float[Tensor, "B 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        prompt: str,
        negative_prompt: str, 
        #latent_noise: Float[Tensor, "B 4 DH DW"],
        output_type: str = "image",
        # prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):   

        batch_size = latents.shape[0]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
         # 3. Encode input prompt

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            self.device,
            self.cfg.num_images_per_prompt,
            self.cfg.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.cfg.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device, **kwargs)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = latents.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=torch.Generator("cuda"), eta=0.0)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # # predict the noise residual
            # if i>=self._num_timesteps-2:
            #     if i == self._num_timesteps-2:
            #         latent_model_input = latent_model_input + latent_noise
            #     noise_pred = self.pipe.unet(
            #         latent_model_input.to(self.weights_dtype),
            #         t.to(self.weights_dtype),
            #         encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
            #         return_dict=False,
            #     )[0]
            # else:
            #     with torch.no_grad():
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
            latents = self.scheduler.step(noise_pred, t, latents,  **extra_step_kwargs, return_dict=False)[0]
        
        # latents = latents + latent_noise
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        # Denormalize an image array to [0,1].
        image = (image / 2 + 0.5).clamp(0, 1)
        # image = F.interpolate(image, (512, 512), mode="bilinear")
        return image