import pickle 
import torch 
import os

all_prompts = pickle.load(open('captions_laion_score6.25.pkl', "rb"))
print("all_prompts:", len(all_prompts))



import torch
from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from model import Flux, FluxParams

from dataclasses import dataclass
from typing import Union, Optional, List, Any, Dict
import random

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

from safetensors.torch import load_file as load_sft
from model import Flux, FluxParams

@dataclass
class ModelSpec:
    params: FluxParams
    repo_id: str 
    repo_flow: str 
    repo_ae: str 
    repo_id_ae: str


config = ModelSpec(
        repo_id="flux-mini",
        repo_flow="flux-mini.safetensors",
        repo_id_ae="FLUX.1-schnell",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=5,
            depth_single_blocks=10,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )
)


dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("../FLUX.1-schnell", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("../FLUX.1-schnell", subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
text_encoder = CLIPTextModel.from_pretrained("../FLUX.1-schnell", subfolder="text_encoder", torch_dtype=dtype,use_safetensors=True).to(device)
tokenizer = CLIPTokenizer.from_pretrained("../FLUX.1-schnell", subfolder="tokenizer", use_safetensors=True)
text_encoder_2 = T5EncoderModel.from_pretrained("../FLUX.1-schnell", subfolder="text_encoder_2", torch_dtype=dtype,use_safetensors=True).to(device)
tokenizer_2 = T5TokenizerFast.from_pretrained("../FLUX.1-schnell", subfolder="tokenizer_2", use_safetensors=True)

ckpt_path = '../flux-mini/flux-mini.safetensors'


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



# pipe = FluxPipeline(
#     scheduler,
#     vae,
#     text_encoder,
#     tokenizer,
#     text_encoder_2,
#     tokenizer_2,
#     None
# )

# t5_output = 't5_dataset'



# for i, prompt in enumerate(all_prompts):

#     output_path = os.path.join(t5_output, 't5_{:04d}.pt'.format(i))

#     if os.access(output_path, os.F_OK):

#         print("skip:", output_path)
#         continue

#     print("prompt:", output_path)


#     prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
#         prompt=prompt,
#         prompt_2=None,
#         prompt_embeds=None,
#         pooled_prompt_embeds=None,
#         device='cuda',
#         num_images_per_prompt=1,
#         max_sequence_length=512,
#         lora_scale=0,
#     )

#     print(prompt_embeds.shape, prompt_embeds.dtype)

#     print(pooled_prompt_embeds.shape, pooled_prompt_embeds.dtype)


#     obj = {
#         'prompt_embeds' : prompt_embeds,
#         'pooled_prompt_embeds' : pooled_prompt_embeds,
#         'prompt' : prompt
#     }

#     torch.save(obj, output_path)


