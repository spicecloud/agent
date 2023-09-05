import torch
import logging
import threading

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union, Generic
from compel import Compel, ReturnedEmbeddingsType
from dataclasses import asdict
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

from spice_agent.inference.types import (
    TStableDiffusionPipeline,
    StableDiffusionPipelineEmbedding,
    StableDiffusionPipelineState,
    StableDiffusionXLPipelineState,
    TStableDiffusionPipelineState,
    StableDiffusionPipelineConfig,
    StableDiffusionXLPipelineConfig,
    StableDiffusionXLImg2ImgPipelineConfig,
)

LOGGER = logging.getLogger(__name__)


class StableDiffusionPipelineEmbeddingInterface(Generic[TStableDiffusionPipeline], ABC):
    """
    An interface for generating compel embeddings
    across stable diffusion pipelines.
    """

    @abstractmethod
    def get_compel(self) -> Compel:
        ...

    @abstractmethod
    def get_pipeline(self) -> TStableDiffusionPipeline:
        ...

    def embed(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
    ) -> StableDiffusionPipelineEmbedding | None:
        """
        Uses compel to generate an embedding for prompt/negative prompt

        Args:
            prompt (`str`):
                The prompt to guide image generation.
            negative_prompt (`str`, *optional*):
                The negative prompt to guide image generation.
        """  # noqa

        # Get pipeline
        pipeline = self.get_pipeline()

        # Check if embeddings are necessary
        if pipeline.tokenizer:
            model_max_length = pipeline.tokenizer.model_max_length
            if (
                len(prompt) <= model_max_length
                and len(negative_prompt) <= model_max_length
            ):
                return None

        LOGGER.info(""" [*] Generating embeddings.""")

        # Get compel
        compel = self.get_compel()

        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if compel.requires_pooled:
            # Get pooled embeddings
            prompt_embeds, pooled_prompt_embeds = compel(prompt)
            negative_prompt_embeds, negative_pooled_prompt_embeds = compel(
                negative_prompt
            )
        else:
            prompt_embeds = compel(prompt)
            negative_prompt_embeds = compel(negative_prompt)

        # Pad prompt embeds
        [
            padded_prompt_embeds,
            padded_negative_prompt_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )

        # Generate embeddings
        embeddings = StableDiffusionPipelineEmbedding(
            prompt_embeds=padded_prompt_embeds,
            negative_prompt_embeds=padded_negative_prompt_embeds,
        )

        # Add pooled embeddings if required
        if compel.requires_pooled:
            # Return embeddings with pooled embeddings
            return StableDiffusionPipelineEmbedding(
                prompt_embeds=padded_prompt_embeds,
                negative_prompt_embeds=padded_negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )
        else:
            return embeddings


class StableDiffusionPipelineTaskObserver(Generic[TStableDiffusionPipelineState], ABC):
    """
    Observes tasks and notifies subscribers about new events within the
    task state.
    """

    def __init__(self):
        self.observers: List[Callable[[TStableDiffusionPipelineState], None]] = []

    def add_observer(self, observer: Callable[[TStableDiffusionPipelineState], None]):
        self.observers.append(observer)

    def remove_observer(
        self, observer: Callable[[TStableDiffusionPipelineState], None]
    ):
        self.observers.remove(observer)

    def run_observers(self, state: TStableDiffusionPipelineState):
        for observer in self.observers:
            observer(state)


def get_filtered_options(options: dict[str, Any], dataclass) -> dict[str, Any]:
    filtered_options = {
        key: value for key, value in options.items() if key in dataclass.__annotations__
    }

    return filtered_options


class StableDiffusionPipelineTask(
    StableDiffusionPipelineEmbeddingInterface,
    StableDiffusionPipelineTaskObserver[StableDiffusionPipelineState],
):
    def __init__(
        self, pipeline: StableDiffusionPipeline, device, options: dict[str, Any]
    ):
        super().__init__()
        self.pipeline = pipeline
        self.device = device

        # Initialize configuration using options
        filtered_options = get_filtered_options(options, StableDiffusionPipelineConfig)
        self.config = StableDiffusionPipelineConfig(**filtered_options)

        # Make sure 'prompt' is specified
        if self.config.prompt is None:
            raise ValueError("prompt cannot be None!")

        # Threading for communicating task state info
        self.progress_thread = None
        self.image_preview_thread = None
        self.state_lock = threading.Lock()

    def get_compel(self) -> Compel:
        return Compel(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
            requires_pooled=False,
            truncate_long_prompts=False,
            device=self.device,
        )

    def get_pipeline(self) -> StableDiffusionPipeline:
        return self.pipeline

    def update_progress(self, step: int):
        if self.pipeline and self.config:
            progress = (step / self.config.num_inference_steps) * 100

            with self.state_lock:
                self.run_observers(
                    StableDiffusionPipelineState(step=step, progress=progress)
                )

    @torch.no_grad()
    def update_image_preview(self, step: int, latents: torch.FloatTensor):
        if self.pipeline and self.config:
            image = self.pipeline.vae.decode(
                latents / self.pipeline.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.pipeline.run_safety_checker(
                image, self.device, torch.FloatTensor
            )

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.pipeline.image_processor.postprocess(
                image, do_denormalize=do_denormalize
            )

            output = StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

            with self.state_lock:
                self.run_observers(
                    StableDiffusionPipelineState(step=step, output=output)
                )

    def callback(self, step: int, timestep: int, latents: torch.FloatTensor) -> None:
        """
        A function that will be called every `callback_steps` steps during inference. The function will be
        called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        """  # noqa
        if self.pipeline and step > 0:
            if not self.progress_thread or not self.progress_thread.is_alive():
                self.progress_thread = threading.Thread(
                    target=self.update_progress, args=(step,)
                )
                self.progress_thread.start()

            image_preview_step_condition_satisfied = (
                step > self.config.num_inference_steps // 2
                and step != self.config.num_inference_steps
            )

            if (
                not self.image_preview_thread
                or not self.image_preview_thread.is_alive()
            ) and image_preview_step_condition_satisfied:
                self.image_preview_thread = threading.Thread(
                    target=self.update_image_preview,
                    args=(step, latents),
                )
                self.image_preview_thread.start()

    @torch.no_grad()
    def run(self, generator: Optional[torch.Generator] = None):
        # Move pipeline to device
        self.pipeline.to(self.device)

        # Configure input
        if not self.config.prompt or not self.config.negative_prompt:
            raise ValueError("prompt and negative_prompt must be defined in config!")

        compel_embeddings = self.embed(self.config.prompt, self.config.negative_prompt)
        if compel_embeddings:
            self.config.prompt_embeds = compel_embeddings.prompt_embeds  # type: ignore
            self.config.negative_prompt_embeds = (  # type: ignore
                compel_embeddings.negative_prompt_embeds
            )

            # The pipeline expects prompt and negative prompt to be None if
            # prompt embeds are defined
            self.config.prompt = None
            self.config.negative_prompt = None

        # Configure callback
        callback = None
        if self.observers:
            callback = self.callback

        result = self.pipeline(
            **asdict(self.config),
            generator=generator,
            callback=callback,
        )

        # Clean up threads
        if self.progress_thread:
            self.progress_thread.join()

        if self.image_preview_thread:
            self.image_preview_thread.join()

        return result


class StableDiffusionXLPipelineTask(
    StableDiffusionPipelineEmbeddingInterface,
    StableDiffusionPipelineTaskObserver[StableDiffusionXLPipelineState],
):
    def __init__(
        self, pipeline: StableDiffusionXLPipeline, device, options: dict[str, Any]
    ):
        super().__init__()
        self.pipeline = pipeline
        self.device = device

        # Initialize configuration using options
        filtered_options = get_filtered_options(
            options, StableDiffusionXLPipelineConfig
        )
        self.config = StableDiffusionXLPipelineConfig(**filtered_options)

        # Make sure 'prompt' is specified
        if self.config.prompt is None:
            raise ValueError("prompt cannot be None!")

        # Threading for communicating task state info
        self.progress_thread = None
        self.image_preview_thread = None
        self.state_lock = threading.Lock()

    def get_compel(self) -> Compel:
        return Compel(
            tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
            text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
            device=self.device,
        )

    def get_pipeline(self) -> StableDiffusionXLPipeline:
        return self.pipeline

    def update_progress(self, step: int):
        if self.pipeline and self.config:
            progress = (step / self.config.num_inference_steps) * 100

            with self.state_lock:
                self.run_observers(
                    StableDiffusionXLPipelineState(step=step, progress=progress)
                )

    @torch.no_grad()
    def update_image_preview(self, step: int, latents: torch.FloatTensor):
        if self.pipeline and self.config:
            # make sure the VAE is in float32 mode, as it overflows in float16
            self.pipeline.vae.to(dtype=torch.float32)

            use_torch_2_0_or_xformers = isinstance(
                self.pipeline.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    LoRAXFormersAttnProcessor,
                    LoRAAttnProcessor2_0,
                ),
            )
            if use_torch_2_0_or_xformers:
                self.pipeline.vae.post_quant_conv.to(latents.dtype)
                self.pipeline.vae.decoder.conv_in.to(latents.dtype)
                self.pipeline.vae.decoder.mid_block.to(latents.dtype)
            else:
                latents = latents.float()  # type: ignore

            image = self.pipeline.vae.decode(
                latents / self.pipeline.vae.config.scaling_factor, return_dict=False
            )[0]

            if self.pipeline.watermark:
                image = self.pipeline.watermark.apply_watermark(image)

            image = self.pipeline.image_processor.postprocess(image, output_type="pil")

            output = StableDiffusionXLPipelineOutput(images=image)

            with self.state_lock:
                self.run_observers(
                    StableDiffusionXLPipelineState(step=step, output=output)
                )

    def callback(self, step: int, timestep: int, latents: torch.FloatTensor) -> None:
        """
        A function that will be called every `callback_steps` steps during inference. The function will be
        called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        """  # noqa
        if self.pipeline and step > 0:
            if not self.progress_thread or not self.progress_thread.is_alive():
                self.progress_thread = threading.Thread(
                    target=self.update_progress, args=(step,)
                )
                self.progress_thread.start()

            image_preview_step_condition_satisfied = (
                step > self.config.num_inference_steps // 2
                and step != self.config.num_inference_steps
            )

            if (
                not self.image_preview_thread
                or not self.image_preview_thread.is_alive()
            ) and image_preview_step_condition_satisfied:
                self.image_preview_thread = threading.Thread(
                    target=self.update_image_preview,
                    args=(step, latents),
                )
                self.image_preview_thread.start()

    @torch.no_grad()
    def run(
        self,
        generator: Optional[torch.Generator] = None,
    ):
        # Move pipeline to device
        self.pipeline.to(self.device)

        # Configure input
        if not self.config.prompt or not self.config.negative_prompt:
            raise ValueError("prompt and negative_prompt must be defined in config!")

        compel_embeddings = self.embed(self.config.prompt, self.config.negative_prompt)
        if compel_embeddings:
            self.config.prompt_embeds = compel_embeddings.prompt_embeds  # type: ignore
            self.config.negative_prompt_embeds = (  # type: ignore
                compel_embeddings.negative_prompt_embeds
            )
            self.config.pooled_prompt_embeds = (
                compel_embeddings.pooled_prompt_embeds
            )  # type: ignore
            self.config.negative_pooled_prompt_embeds = (
                compel_embeddings.negative_pooled_prompt_embeds
            )  # type: ignore

            # The pipeline expects prompt and negative prompt to be None if
            # prompt embeds are defined
            self.config.prompt = None
            self.config.negative_prompt = None

        # Configure callback
        callback = None
        if self.observers:
            callback = self.callback

        result = self.pipeline(
            **asdict(self.config),
            generator=generator,
            callback=callback,
        )

        # Clean up threads
        if self.progress_thread:
            self.progress_thread.join()

        if self.image_preview_thread:
            self.image_preview_thread.join()

        return result


class StableDiffusionXLImg2ImgPipelineTask(
    StableDiffusionPipelineEmbeddingInterface,
    StableDiffusionPipelineTaskObserver[StableDiffusionXLPipelineState],
):
    def __init__(
        self,
        pipeline: StableDiffusionXLImg2ImgPipeline,
        device,
        options: dict[str, Any],
    ):
        super().__init__()
        self.pipeline = pipeline
        self.device = device

        # Initialize configuration using options
        filtered_options = get_filtered_options(
            options, StableDiffusionXLImg2ImgPipelineConfig
        )
        self.config = StableDiffusionXLImg2ImgPipelineConfig(**filtered_options)

        # Make sure 'prompt' is specified
        if self.config.prompt is None:
            raise ValueError("prompt cannot be None!")

        # Threading for communicating task state info
        self.progress_thread = None
        self.state_lock = threading.Lock()

    def get_compel(self) -> Compel:
        return Compel(
            tokenizer=self.pipeline.tokenizer_2,
            text_encoder=self.pipeline.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            truncate_long_prompts=False,
            device=self.device,
        )

    def get_pipeline(self) -> StableDiffusionXLImg2ImgPipeline:
        return self.pipeline

    def update_progress(self, step: int):
        if self.pipeline and self.config:
            progress = (step / self.config.num_inference_steps) * 100

            with self.state_lock:
                self.run_observers(
                    StableDiffusionXLPipelineState(step=step, progress=progress)
                )

    def callback(self, step: int, timestep: int, latents: torch.FloatTensor) -> None:
        """
        A function that will be called every `callback_steps` steps during inference. The function will be
        called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        """  # noqa
        if self.pipeline and step > 0:
            if not self.progress_thread or not self.progress_thread.is_alive():
                self.progress_thread = threading.Thread(
                    target=self.update_progress, args=(step,)
                )
                self.progress_thread.start()

    @torch.no_grad()
    def run(
        self,
        generator: Optional[torch.Generator] = None,
    ):
        # Move pipeline to device
        self.pipeline.to(self.device)

        # Configure input
        if not self.config.prompt or not self.config.negative_prompt:
            raise ValueError("prompt and negative_prompt must be defined in config!")

        compel_embeddings = self.embed(self.config.prompt, self.config.negative_prompt)
        if compel_embeddings:
            self.config.prompt_embeds = compel_embeddings.prompt_embeds  # type: ignore
            self.config.negative_prompt_embeds = (  # type: ignore
                compel_embeddings.negative_prompt_embeds
            )
            self.config.pooled_prompt_embeds = (
                compel_embeddings.pooled_prompt_embeds
            )  # type: ignore
            self.config.negative_pooled_prompt_embeds = (
                compel_embeddings.negative_pooled_prompt_embeds
            )  # type: ignore

            # The pipeline expects prompt and negative prompt to be None if
            # prompt embeds are defined
            self.config.prompt = None
            self.config.negative_prompt = None

        # Configure callback
        callback = None
        if self.observers:
            callback = self.callback

        result = self.pipeline(
            **asdict(self.config),
            generator=generator,
            callback=callback,
        )

        # Clean up threads
        if self.progress_thread:
            self.progress_thread.join()

        return result
