"""
360° Panorama Generator with SDXL Turbo
========================================
Recreates the ComfyUI 360-panorama-sdxl workflow entirely in Python.

Expected files in the same directory as this script:
  - dreamshaperXL_v21TurboDPMSDE.safetensors   (SDXL checkpoint)
  - 360RedmondResized.safetensors               (LoRA for 360° style)
  - sdxl.vae.safetensors                        (SDXL VAE)

pip install torch torchvision diffusers safetensors accelerate pillow numpy

model: https://civitai.com/api/download/models/354657?type=Model&format=SafeTensor&size=full&fp=fp16
vae: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors?download=true
lora: https://civitai.com/api/download/models/143197?type=Model&format=SafeTensor
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverSDEScheduler,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# Local model files (expected in same directory as this script)
CHECKPOINT_PATH = SCRIPT_DIR / "dreamshaperXL_lightningDPMSDE.safetensors"
LORA_PATH       = SCRIPT_DIR / "360RedmondResized.safetensors"
LORA_DETAILS_PATH = SCRIPT_DIR / "SDXLFaeTastic2400.safetensors"
VAE_PATH        = SCRIPT_DIR / "sdxl.vae.safetensors"

# Generation parameters (from the ComfyUI workflow)
PROMPT = (
    "Glowing mushrooms around pyramids amidst a cosmic backdrop, "
    "equirectangular, 360 panorama, cinematic"
)
NEGATIVE_PROMPT = (
    "boring, text, signature, watermark, low quality, bad quality, "
    "grainy, blurry, long neck, closed eyes"
)
WIDTH       = 2048
HEIGHT      = 1024
STEPS       = 8
CFG_SCALE   = 3.0
SEED        = 80484030936239
DENOISE     = 1.0

# Tiling (tileX=1, tileY=0 → circular padding on X-axis only)
TILE_X = True
TILE_Y = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# check for mps
if torch.backends.mps.is_available():
    DEVICE = "mps"

DTYPE  = torch.float16


# ──────────────────────────────────────────────
# 1. Seamless tiling patches
# ──────────────────────────────────────────────
def _make_asymmetric_forward(module, pad_h, pad_w, tile_x, tile_y):
    """Create a patched forward that applies circular/zero padding per axis."""
    original_forward = module._conv_forward

    def patched_conv_forward(input, weight, bias):
        if tile_x and tile_y:
            input = torch.nn.functional.pad(input, (pad_w, pad_w, pad_h, pad_h), mode="circular")
        elif tile_x:
            input = torch.nn.functional.pad(input, (pad_w, pad_w, 0, 0), mode="circular")
            input = torch.nn.functional.pad(input, (0, 0, pad_h, pad_h), mode="constant", value=0)
        elif tile_y:
            input = torch.nn.functional.pad(input, (0, 0, pad_h, pad_h), mode="circular")
            input = torch.nn.functional.pad(input, (pad_w, pad_w, 0, 0), mode="constant", value=0)
        else:
            return original_forward(input, weight, bias)

        return torch.nn.functional.conv2d(
            input, weight, bias, module.stride, (0, 0), module.dilation, module.groups
        )

    return patched_conv_forward


def enable_seamless_tiling(model, tile_x=True, tile_y=False):
    """Patch all Conv2d layers in a model for seamless tiling."""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            pad_h = module.padding[0]
            pad_w = module.padding[1]
            if pad_h == 0 and pad_w == 0:
                continue
            module._conv_forward = _make_asymmetric_forward(module, pad_h, pad_w, tile_x, tile_y)


# ──────────────────────────────────────────────
# 2. Load SDXL pipeline from local .safetensors
# ──────────────────────────────────────────────
print(f"Loading SDXL checkpoint from {CHECKPOINT_PATH} ...")
pipe = StableDiffusionXLPipeline.from_single_file(
    str(CHECKPOINT_PATH),
    torch_dtype=DTYPE,
    use_safetensors=True,
)

# Load separate VAE from local file
print(f"Loading VAE from {VAE_PATH} ...")
vae = AutoencoderKL.from_single_file(
    str(VAE_PATH),
    torch_dtype=DTYPE,
)
pipe.vae = vae

# Load LoRAs with individual scales
print(f"Loading main 360° LoRA from {LORA_PATH} (scale=1.0) ...")
pipe.load_lora_weights(str(LORA_PATH), adapter_name="main_lora")

if LORA_DETAILS_PATH.exists():
    print(f"Loading details LoRA from {LORA_DETAILS_PATH} (scale=0.1) ...")
    pipe.load_lora_weights(str(LORA_DETAILS_PATH), adapter_name="details_lora")
else:
    print("⚠️ Details LoRA file not found — skipping.")

print("Fusing LoRAs with correct per-adapter scales...")
pipe.fuse_lora(adapter_names=["main_lora"], lora_scale=1.0)
if LORA_DETAILS_PATH.exists():
    pipe.fuse_lora(adapter_names=["details_lora"], lora_scale=0.5)

# Move to GPU
pipe = pipe.to(DEVICE)

# ──────────────────────────────────────────────
# 3. Configure scheduler: DPM++ SDE / ddim_uniform
# ──────────────────────────────────────────────
pipe.scheduler = DPMSolverSDEScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=False,
)

# ──────────────────────────────────────────────
# 4. Enable seamless X-axis tiling on UNet + VAE
# ──────────────────────────────────────────────
print("Enabling seamless X-axis tiling on UNet and VAE decoder ...")
enable_seamless_tiling(pipe.unet, tile_x=TILE_X, tile_y=TILE_Y)
enable_seamless_tiling(pipe.vae.decoder, tile_x=TILE_X, tile_y=TILE_Y)

# ──────────────────────────────────────────────
# 5. Generate the 360° panorama
# ──────────────────────────────────────────────
print(f"Generating {WIDTH}x{HEIGHT} panorama ({STEPS} steps, cfg={CFG_SCALE}) ...")
generator = torch.Generator(DEVICE).manual_seed(SEED)

result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=CFG_SCALE,
    generator=generator,
)
panorama = result.images[0]

panorama_path = SCRIPT_DIR / "panorama_360.png"
panorama.save(panorama_path)
print(f"Saved panorama → {panorama_path}")

# ──────────────────────────────────────────────
# 6. Export merged model (LoRA baked in) for Swift
# ──────────────────────────────────────────────
# The LoRA weights were already fused into the base model by pipe.fuse_lora()
# above. Now we save all components into a single .safetensors file with
# key prefixes that match what the Swift loadStableDiffusionXLFromSingleFile()
# expects when it detects the "diffusers" format:
#
#   unet.*               → UNet weights
#   text_encoder.*       → CLIP text encoder 1
#   text_encoder_2.*     → CLIP text encoder 2 (OpenCLIP)
#   first_stage_model.*  → VAE (same prefix in both formats)
#
# In Swift you can then load with:
#   let sd = try loadStableDiffusionXLFromSingleFile(
#       url: URL(filePath: "/path/to/merged_checkpoint.safetensors"),
#       dType: .float16
#   )
# No LoRA loading step needed — the weights are already baked in.

from safetensors.torch import save_file

print("\nExporting merged checkpoint (LoRA baked in)...")

# Clean up LoRA adapter metadata (weights are already fused into base layers)
pipe.unload_lora_weights()

merged_state_dict = {}

# UNet
for k, v in pipe.unet.state_dict().items():
    merged_state_dict[f"unet.{k}"] = v.contiguous().half()

# Text encoder 1 (CLIP-L)
for k, v in pipe.text_encoder.state_dict().items():
    merged_state_dict[f"text_encoder.{k}"] = v.contiguous().half()

# Text encoder 2 (CLIP-G / OpenCLIP)
for k, v in pipe.text_encoder_2.state_dict().items():
    merged_state_dict[f"text_encoder_2.{k}"] = v.contiguous().half()

# VAE — use first_stage_model.* prefix (expected by Swift loader in both formats)
for k, v in pipe.vae.state_dict().items():
    merged_state_dict[f"first_stage_model.{k}"] = v.contiguous().half()

merged_path = SCRIPT_DIR / "merged_checkpoint.safetensors"
save_file(merged_state_dict, str(merged_path))
print(f"Saved merged checkpoint → {merged_path}")
print(f"  Total tensors: {len(merged_state_dict)}")
print(f"  File size: {merged_path.stat().st_size / (1024**3):.2f} GB")

# Free SDXL VRAM
del pipe, vae, merged_state_dict
torch.cuda.empty_cache()

print("\nDone! Output files:")
print(f"  Panorama:           {panorama_path}")
print(f"  Merged checkpoint:  {merged_path}")
print(f"\nTo load in Swift (no LoRA step needed):")
print(f'  let sd = try loadStableDiffusionXLFromSingleFile(')
print(f'      url: URL(filePath: "{merged_path}"),')
print(f'      dType: .float16')
print(f'  )')

