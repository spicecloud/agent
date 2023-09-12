import torch
import logging
import threading

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Generic
from compel import Compel, ReturnedEmbeddingsType
from dataclasses import asdict
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
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
    StableDiffusionPipelineTaskState,
    TStableDiffusionPipelineRunConfig,
    StableDiffusionPipelineRunConfig,
    StableDiffusionImg2ImgPipelineRunConfig,
    StableDiffusionXLPipelineRunConfig,
    StableDiffusionXLImg2ImgPipelineRunConfig,
    StableDiffusionXLControlNetPipelineRunConfig,
)

LOGGER = logging.getLogger(__name__)


class TaskObserver:
    """
    Observes tasks and notifies subscribers about new events within the
    task state.
    """

    def __init__(self):
        self.observers: List[Callable[[StableDiffusionPipelineTaskState], None]] = []

    def add_observer(
        self, observer: Callable[[StableDiffusionPipelineTaskState], None]
    ):
        self.observers.append(observer)

    def remove_observer(
        self, observer: Callable[[StableDiffusionPipelineTaskState], None]
    ):
        self.observers.remove(observer)

    def run_observers(self, state: StableDiffusionPipelineTaskState):
        for observer in self.observers:
            observer(state)


class StableDiffusionPipelineCompelEmbeddingMixin(Generic[TStableDiffusionPipeline]):
    """
    A mixin that adds compel embedding generation
    across stable diffusion pipelines.
    """

    def __init__(
        self,
        device,
        pipeline: TStableDiffusionPipeline,
    ):
        self.device = device
        self.pipeline = pipeline

    def _create_compel(self) -> Compel:
        """
        Generates compel object based on pipeline type.
        """
        if isinstance(
            self.pipeline, (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)
        ):
            return Compel(
                tokenizer=self.pipeline.tokenizer,
                text_encoder=self.pipeline.text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                requires_pooled=False,
                truncate_long_prompts=False,
                device=self.device,
            )
        elif isinstance(
            self.pipeline,
            (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline),
        ):
            return Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
                device=self.device,
            )
        elif isinstance(self.pipeline, StableDiffusionXLImg2ImgPipeline):
            return Compel(
                tokenizer=self.pipeline.tokenizer_2,
                text_encoder=self.pipeline.text_encoder_2,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
                truncate_long_prompts=False,
                device=self.device,
            )
        else:
            message = f"Compel object implementation for {type(self.pipeline)} does not exist!"  # noqa
            LOGGER.error(message)
            raise NotImplementedError(message)

    def create_compel_embeddings(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        embed_anyways: bool = True,
    ) -> StableDiffusionPipelineEmbedding | None:
        """
        Uses compel to generate an embedding for prompt/negative prompt. Note the pipeline
        must be on device before calling this function.

        Args:
            prompt (`str`):
                The prompt to guide image generation.
            negative_prompt (`str`, *optional*):
                The negative prompt to guide image generation.
        """  # noqa

        # Check if embeddings are necessary
        if not embed_anyways and self.pipeline.tokenizer:
            model_max_length = self.pipeline.tokenizer.model_max_length
            if (
                len(prompt) <= model_max_length
                and len(negative_prompt) <= model_max_length
            ):
                return None

        # Get compel
        compel = self._create_compel()
        if not compel:
            return None

        LOGGER.info(""" [*] Generating embeddings. """)

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


class StableDiffusionPipelineTaskBase(
    Generic[
        TStableDiffusionPipeline,
        TStableDiffusionPipelineRunConfig,
    ],
    ABC,
):
    """
    A base class for stable diffusion pipeline tasks
    """

    def __init__(
        self,
        device,
        pipeline: TStableDiffusionPipeline,
        pipeline_state: StableDiffusionPipelineTaskState,
        pipeline_run_config: TStableDiffusionPipelineRunConfig,
        enable_image_preview: Optional[bool] = True,
    ):
        super().__init__()

        LOGGER.info(
            f""" [*] Configuring {type(self).__name__} with run config: {type(pipeline_run_config).__name__}. """  # noqa
        )

        self.device = device

        # Configure pipeline and move to GPU
        self.pipeline = pipeline.to(self.device)
        self.state = pipeline_state

        self.observer = TaskObserver()
        self.embedding = StableDiffusionPipelineCompelEmbeddingMixin(
            self.device, self.pipeline
        )

        # Threading for communicating task state info
        self.progress_thread = None
        self.image_preview_thread = None
        self.state_lock = threading.Lock()

        self.pipeline_run_config = pipeline_run_config
        self.enable_image_preview = enable_image_preview

        # Updates pipeline run config
        self._maybe_configure_embeddings()

    def serialize_pipeline_run_config(self) -> str:
        """
        Serializes the pipeline run configuration. Useful for debugging.
        """
        import json

        def default_serializer(obj):
            try:
                return json.dumps(obj)
            except TypeError:
                try:
                    return str(obj)
                except Exception:
                    LOGGER.warn(f"serialize_pipeline_run_config failed on {obj}")
                    return "No string representation."

        pipeline_run_config_dict = asdict(self.pipeline_run_config)
        pipeline_run_config_json = json.dumps(
            pipeline_run_config_dict,
            indent=2,
            default=default_serializer,
        )

        return pipeline_run_config_json

    def _maybe_configure_embeddings(self):
        """
        Possibly configures embeddings for prompt and negative prompt. This is only done
        if necessary (see StableDiffusionPipelineCompelEmbeddingMixin)
        """
        if (
            not self.pipeline_run_config.prompt
            or not self.pipeline_run_config.negative_prompt
        ):
            raise ValueError("prompt and negative_prompt must be defined in config!")

        compel_embeddings = self.embedding.create_compel_embeddings(
            self.pipeline_run_config.prompt, self.pipeline_run_config.negative_prompt
        )
        if compel_embeddings:
            self.pipeline_run_config.prompt_embeds = (  # type: ignore
                compel_embeddings.prompt_embeds
            )
            self.pipeline_run_config.negative_prompt_embeds = (  # type: ignore
                compel_embeddings.negative_prompt_embeds
            )

            if (
                compel_embeddings.pooled_prompt_embeds is not None
                and compel_embeddings.negative_pooled_prompt_embeds is not None
            ):
                self.pipeline_run_config.pooled_prompt_embeds = (
                    compel_embeddings.pooled_prompt_embeds
                )
                self.pipeline_run_config.negative_pooled_prompt_embeds = (
                    compel_embeddings.negative_pooled_prompt_embeds
                )

            # The pipeline expects prompt and negative prompt to be None if
            # prompt embeds are defined
            self.pipeline_run_config.prompt = None
            self.pipeline_run_config.negative_prompt = None

    @abstractmethod
    def update_image_preview(self):
        """
        Updates observers with new image previews
        """
        ...

    def update_progress(self, step: int):
        """
        Updates observers with new progress information
        """
        if self.pipeline and self.pipeline_run_config:
            progress = (step / self.pipeline_run_config.num_inference_steps) * 100

            with self.state_lock:
                self.state.step = step
                self.state.progress = progress
                self.observer.run_observers(self.state)

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

            if self.enable_image_preview:
                image_preview_step_condition_satisfied = (
                    step > self.pipeline_run_config.num_inference_steps // 2
                    and step != self.pipeline_run_config.num_inference_steps
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

    def _clean_up_threads(self):
        """
        Cleans up threads associated with the instance
        """
        if self.progress_thread:
            self.progress_thread.join()

        if self.image_preview_thread:
            self.image_preview_thread.join()

    @torch.no_grad()
    def run_pipeline(
        self, generator: Optional[torch.Generator] = None
    ) -> StableDiffusionPipelineTaskState:
        """
        Runs the pipeline and returns the state
        """
        LOGGER.info(
            f""" [*] Running {type(self).__name__} with run config: {type(self.pipeline_run_config).__name__}. """  # noqa
        )

        # Configure callback
        callback = None
        if self.observer.observers:
            callback = self.callback

        result = self.pipeline(
            **asdict(self.pipeline_run_config),
            generator=generator,
            callback=callback,
        )

        self._clean_up_threads()
        self.state.output = result

        return self.state


class StableDiffusionPipelineTask(
    StableDiffusionPipelineTaskBase[
        StableDiffusionPipeline,
        StableDiffusionPipelineRunConfig,
    ],
):
    @torch.no_grad()
    def update_image_preview(self, step: int, latents: torch.FloatTensor):
        if self.pipeline and self.pipeline_run_config:
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
                self.observer.run_observers(
                    StableDiffusionPipelineTaskState(step=step, output=output)
                )


class StableDiffusionImg2ImgPipelineTask(
    StableDiffusionPipelineTaskBase[
        StableDiffusionImg2ImgPipeline,
        StableDiffusionImg2ImgPipelineRunConfig,
    ],
):
    @torch.no_grad()
    def update_image_preview(self, step: int, latents: torch.FloatTensor):
        if self.pipeline and self.pipeline_run_config:
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
                self.observer.run_observers(
                    StableDiffusionPipelineTaskState(step=step, output=output)
                )


class StableDiffusionXLPipelineTask(
    StableDiffusionPipelineTaskBase[
        StableDiffusionXLPipeline,
        StableDiffusionXLPipelineRunConfig,
    ]
):
    @torch.no_grad()
    def update_image_preview(self, step: int, latents: torch.FloatTensor):
        if self.pipeline and self.pipeline_run_config:
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
