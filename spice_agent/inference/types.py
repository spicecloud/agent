import torch
from typing import Optional, Dict, Union, List, Any, Tuple
from dataclasses import dataclass
from PIL import Image as PILImage
import numpy as np


# Data Classes for Stable Diffusion Pipeline -------------------------------------------
@dataclass
class InputForStableDiffusionPipeline:
    """
    Inputs for Stable Diffusion Pipeline

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
    """  # noqa

    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None


@dataclass
class InferenceOptionsForStableDiffusionPipeline:
    """
    Inference options for Stable Diffusion Pipeline

    Args:
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
    """  # noqa

    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.7


@dataclass
class OutputForStableDiffusionPipeline:
    """
    Output related arguments for Stable Diffusion Pipeline

    Args:
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
    """  # noqa

    # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True


# Data Classes for Stable Diffusion XL Pipeline ----------------------------------------
@dataclass
class InputForStableDiffusionXLPipeline:
    """
    Inputs for Stable Diffusion XL Pipeline

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
    """  # noqa

    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None


@dataclass
class InferenceOptionsForStableDiffusionXLPipeline:
    """
    Inference Options for Stable Diffusion XL Pipeline

    Args:
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            TODO
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            TODO
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            TODO
    """  # noqa

    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.7
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None


# Data Classes for Stable Diffusion XL IMG2IMG Pipeline --------------------------------


@dataclass
class InputForStableDiffusionXLImg2ImgPipeline:
    """
    Input for Stable Diffusion XL Img2Img Pipeline

    Args:
        image (`torch.FloatTensor` or `PIL.Image.Image` or `np.ndarray` or `List[torch.FloatTensor]` or `List[PIL.Image.Image]` or `List[np.ndarray]`):
            The image(s) to modify with the pipeline.
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
    """  # noqa

    image: Optional[
        Union[
            torch.FloatTensor,
            PILImage.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PILImage.Image],
            List[np.ndarray],
        ]
    ] = None

    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None


@dataclass
class InferenceOptionsForStableDiffusionXLImg2ImgPipeline:
    """
    Args:
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
            will be used as a starting point, adding more noise to it the larger the `strength`. The number of
            denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
            be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            TODO
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            TODO
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            TODO
        aesthetic_score (`float`, *optional*, defaults to 6.0):
            TODO
        negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
            TDOO


    """  # noqa

    strength: float = 0.3
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.7
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5


# Encapsulating data classes for input -------------------------------------------------

# These have the following 4 attributes:
# - input: these are inputs sent to the inference task
# - inference_options: these are options that affect the run
# - output: these are additional arguments that specify output behavior


@dataclass
class StableDiffusionPipelineInput:
    input: InputForStableDiffusionPipeline
    inference_options: InferenceOptionsForStableDiffusionPipeline
    output: OutputForStableDiffusionPipeline


@dataclass
class StableDiffusionXLPipelineInput:
    input: InputForStableDiffusionXLPipeline
    inference_options: InferenceOptionsForStableDiffusionXLPipeline
    output: OutputForStableDiffusionPipeline


@dataclass
class StableDiffusionXLImg2ImgPipelineInput:
    input: InputForStableDiffusionXLImg2ImgPipeline
    inference_options: InferenceOptionsForStableDiffusionXLImg2ImgPipeline
    output: OutputForStableDiffusionPipeline
