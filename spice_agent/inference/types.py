import torch
import PIL.Image
import numpy as np
from typing import Dict, Optional, Union, List, Any, Tuple, TypeVar
from dataclasses import dataclass

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

SUPPORTED_MODEL_REPOSITORY_IDS: List[str] = [
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

# Supported stable diffusion pipelines
TStableDiffusionPipeline = TypeVar(
    "TStableDiffusionPipeline",
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)


TStableDiffusionPipelineOutput = TypeVar(
    "TStableDiffusionPipelineOutput",
    StableDiffusionPipelineOutput,
    StableDiffusionXLPipelineOutput,
)


@dataclass
class StableDiffusionPipelineEmbedding:
    """
    Embedding representation for stable diffusion pipelines.
    """

    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None


@dataclass
class StableDiffusionPipelineTaskState:
    """
    Stable diffusion pipeline state information
    """

    step: Optional[int] = None
    progress: Optional[float] = None
    output: Optional[
        Union[
            Union[
                Tuple,
                StableDiffusionPipelineOutput,
            ],
            Union[
                Tuple,
                StableDiffusionXLPipelineOutput,
            ],
        ]
    ] = None


@dataclass
class StableDiffusionPipelineRunConfig:
    """
    Configuration arguments for stable diffusion pipeline
    """

    prompt: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0


@dataclass
class StableDiffusionImg2ImgPipelineRunConfig:
    prompt: Optional[Union[str, List[str]]] = None
    image: Optional[
        Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ]
    ] = None
    strength: float = 0.8
    num_inference_steps: int = 50
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: Optional[float] = 0.0
    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class StableDiffusionXLPipelineRunConfig:
    """
    Configuration arguments for stable diffusion xl pipeline
    """

    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    denoising_end: Optional[float] = None
    guidance_scale: float = 5.0
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None


@dataclass
class StableDiffusionXLImg2ImgPipelineRunConfig:
    """
    Configuration arguments for stable diffusion xl img2img pipeline
    """

    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    image: Optional[
        Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ]
    ] = None
    strength: float = 0.8
    num_inference_steps: int = 50
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None
    guidance_scale: float = 8.0
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5


@dataclass
class StableDiffusionXLControlNetPipelineRunConfig:
    """
    Configuration arguments for stable diffusion xl control net pipeline
    """

    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    image: Optional[
        Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ]
    ] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    controlnet_conditioning_scale: Union[float, List[float]] = 0.5
    guess_mode: bool = False
    control_guidance_start: Union[float, List[float]] = 0.0
    control_guidance_end: Union[float, List[float]] = 1.0
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None


TStableDiffusionPipelineRunConfig = TypeVar(
    "TStableDiffusionPipelineRunConfig",
    StableDiffusionPipelineRunConfig,
    StableDiffusionImg2ImgPipelineRunConfig,
    StableDiffusionXLPipelineRunConfig,
    StableDiffusionXLImg2ImgPipelineRunConfig,
    StableDiffusionXLControlNetPipelineRunConfig,
)
