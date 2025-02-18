import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

class StableDiffusionPromptProcessor():
    ### these functions are unused, kept for debugging ###
    def __init__(
        self, pretrained_model_name_or_path, device
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.configure_text_encoder()
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Float[Tensor, "B 77 768"]:
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

        text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
        uncond_text_embeddings = self.text_encoder(
            uncond_tokens.input_ids.to(self.device)
        )[0]

        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        "caculating tokens and text embeddings"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
