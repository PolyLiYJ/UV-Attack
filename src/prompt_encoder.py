from transformers import AutoTokenizer, CLIPTextModel
import torch
import os
from stable_diffusion_generator import StableDiffusionGenerator
from torchvision import transforms
from diffusers.utils.torch_utils import randn_tensor
import cv2
import numpy as np
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d_modify import MyHardPhongShader

class Prompt_Encoder():
    def __init__(self, pretrained_model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

    def get_text_embeddings(self, prompt, negative_prompt, batch_size=1):
    #-> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]
        text_embeddings = text_embeddings.expand(batch_size, -1, -1)  # type: ignore
        uncond_text_embeddings = uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
        # return text_embeddings, uncond_text_embeddings