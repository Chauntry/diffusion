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
import os
import time

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

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)


# text_encoder = CLIPTextModel.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="text_encoder", torch_dtype=dtype,use_safetensors=True).to(device)
# tokenizer = CLIPTokenizer.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="tokenizer", use_safetensors=True)
# text_encoder_2 = T5EncoderModel.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="text_encoder_2", torch_dtype=dtype,use_safetensors=True).to(device)
# tokenizer_2 = T5TokenizerFast.from_pretrained("/mnt/c/Data/work_code/FLUX.1-schnell", subfolder="tokenizer_2", use_safetensors=True)



text_encoder = None
tokenizer = None
text_encoder_2 = None
tokenizer_2 = None


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

def expand_dims(V, dims):
    return V[(...,) + (None,) * (dims - 1)]

def generate_new(model, noise, num_steps, timesteps, eta):
    

    noise_latent_list = []

    latent = noise

    t = timesteps[idx]


    for idx in range(num_steps):

        sigma = expand_dims(1 - t, 3)

        model_v = model(latent, t)

        x0 = latent - (1 - sigma) * model_v
        x0_eps = eta * x0 + (1 - eta**2) ** 0.5 * torch.randn(xt_g.shape)

        x1 = latent + sigma * model_v


        t = timesteps[idx + 1]
        sigma_new = expand_dims(1 - t, 3)

        latent = x0_eps * sigma_new + x1 * (1 - sigma_new)

        noise_latent_list.append(latent)

    return noise_latent_list



@torch.inference_mode()
def flux_inference(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    prompt = '  '
    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    # 3. Encode prompt
    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None

    dtype = torch.bfloat16

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    if prompt_embeds is None:
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    print(self)

    print('transformer:', self.transformer)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )


    print(timesteps)
    
    self._num_timesteps = len(timesteps)

    # Handle guidance
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if self.transformer.params.guidance_embed else None

    print('latents:', latents.dtype, 'pooled_projections:', pooled_prompt_embeds.device, 'prompt_embeds:', prompt_embeds.device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # Yield intermediate result
            latents_for_image = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents_for_image = (latents_for_image / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            latents_for_image.to
            image = self.vae.decode(latents_for_image, return_dict=False)[0]
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            torch.cuda.empty_cache()

    # Final image using good_vae
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / good_vae.config.scaling_factor) + good_vae.config.shift_factor
    image = good_vae.decode(latents, return_dict=False)[0]
    self.maybe_free_model_hooks()
    torch.cuda.empty_cache()
    
    return self.image_processor.postprocess(image, output_type=output_type)[0]



transformer = Flux(config.params).to(dtype=dtype, device=device)
if ckpt_path is not None:
    sd = load_sft(ckpt_path, device='cpu')
    missing, unexpected = transformer.load_state_dict(sd, strict=True)
    # print('missing:', missing)
    # print('unexpected:', unexpected)
    sd = None

pipe = FluxPipeline(
    scheduler,
    vae,
    None,
    None,
    None,
    None,
    transformer
)
torch.cuda.empty_cache()

# while True:
#     time.sleep(1)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

def infer(prompt, seed=42, randomize_seed=False, width=512, height=512, guidance_scale=3.5, num_inference_steps=20,
          prompt_embeds=None, pooled_prompt_embeds=None):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    img = flux_inference(
            pipe,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=vae,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds
        )
    
    return img

#         'prompt_embeds' : prompt_embeds,
#         'pooled_prompt_embeds' : pooled_prompt_embeds,
#         'prompt' : prompt


t5_output = 't5_dataset'

t5_files = os.listdir(t5_output)


num_inference_steps = 58

for t5_file in t5_files:
    t5_path = os.path.join(t5_output, t5_file)

    t5_obj = torch.load(t5_path)
    
    prompt_embeds = t5_obj['prompt_embeds']
    pooled_prompt_embeds = t5_obj['pooled_prompt_embeds']
    prompt = t5_obj['prompt']


    print(f'prompt_embeds = {prompt_embeds.dtype} {prompt_embeds.shape}')
    print(f'pooled_prompt_embeds = {pooled_prompt_embeds.dtype} {pooled_prompt_embeds.shape}')
    print(f'prompt = {prompt}')

    img = infer(None, prompt_embeds=prompt_embeds.to(device), pooled_prompt_embeds=pooled_prompt_embeds.to(device), num_inference_steps=num_inference_steps)


    img.save(f'output_image_{num_inference_steps}.png')

    break

