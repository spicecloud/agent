import json
import logging
import os
from compel import Compel, ReturnedEmbeddingsType
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union
from dataclasses import asdict

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from gql import gql
from gql.transport.exceptions import TransportQueryError

from spice_agent.inference.types import (
    CallbackOptionsBase,
    InputForStableDiffusionPipeline,
    InputForStableDiffusionXLPipeline,
    InputForStableDiffusionXLImg2ImgPipeline,
    InferenceOptionsForStableDiffusionPipeline,
    InferenceOptionsForStableDiffusionXLPipeline,
    InferenceOptionsForStableDiffusionXLImg2ImgPipeline,
    StableDiffusionPipelineInput,
    StableDiffusionXLPipelineInput,
    StableDiffusionXLImg2ImgPipelineInput,
    OutputForStableDiffusionPipeline,
)
from spice_agent.utils.config import SPICE_INFERENCE_DIRECTORY

# from torch.mps import empty_cache as mps_empty_cache ## SAVE FOR LATER

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
import transformers  # noqa
from transformers.pipelines.base import PipelineException  # noqa

LOGGER = logging.getLogger(__name__)


def _init_compel(
    device, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str = ""
) -> Compel:
    """
    Initializes compel text embedding system based on type of pipeline.
    """

    # Compel settings consistent across compel objects
    DEFAULT_COMPEL_ARGUMENTS = {
        "truncate_long_prompts": False,
        "device": device,
    }

    # Currently, these are the only supported pipelines
    if not isinstance(
        pipeline,
        (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
        ),
    ):
        message = (
            f"Prompt embeddings are not supported for pipeline {type(pipeline)}"  # noqa
        )
        LOGGER.warn(message)
        return None

    # Check if embeddings are necessary
    if pipeline.tokenizer:
        model_max_length = pipeline.tokenizer.model_max_length
        if len(prompt) <= model_max_length and len(negative_prompt) <= model_max_length:
            return None

    # Configure compel objects for StableDiffusion pipeline
    if isinstance(pipeline, StableDiffusionPipeline):
        return Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
            requires_pooled=False,
            **DEFAULT_COMPEL_ARGUMENTS,
        )

    # Configure compel objects for StableDiffusionXL pipelines
    if isinstance(pipeline, StableDiffusionXLPipeline):
        return Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            **DEFAULT_COMPEL_ARGUMENTS,
        )

    if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
        return Compel(
            tokenizer=pipeline.tokenizer_2,
            text_encoder=pipeline.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            **DEFAULT_COMPEL_ARGUMENTS,
        )

    return None


def _run_compel(
    compel: Compel, prompt: str, negative_prompt: str = ""
) -> InputForStableDiffusionXLPipeline:
    prompt_embeds = None
    pooled_prompt_embeds = None
    negative_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    if compel.requires_pooled:
        # Get pooled embeddings
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
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

    embeddings = InputForStableDiffusionXLPipeline(
        prompt_embeds=padded_prompt_embeds,
        negative_prompt_embeds=padded_negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    return embeddings


def get_input_for_stable_diffusion_pipeline(
    device, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str = ""
) -> InputForStableDiffusionPipeline | InputForStableDiffusionXLPipeline:
    """
    Returns prompt options for embeddings if a compel object exists; otherwise,
    prompt options without embeddings are returned
    """

    LOGGER.info(""" [*] Generating embeddings. (Ignore token indices complaint)""")

    without_embeddings = InputForStableDiffusionPipeline(
        prompt=prompt, negative_prompt=negative_prompt
    )
    compel = _init_compel(device, pipeline, prompt, negative_prompt)

    if compel:
        try:
            return _run_compel(compel, prompt, negative_prompt)
        except Exception as exception:
            LOGGER.error(
                f""" [*] Embedding generation failed with exception {exception}! Falling back to normal prompt."""  # noqa
            )

    return without_embeddings


class Inference:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.spice.get_device()

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()  # type: ignore
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

    def _update_inference_job(
        self,
        inference_job_id: str,
        status: str,
        was_guarded: Optional[bool] = None,
        text_output: Optional[str] = None,
        file_outputs_ids: list[str] = [],
    ):
        mutation = gql(
            """
            mutation updateInferenceJob($input: UpdateInferenceJobInput!) {
                updateInferenceJob(input: $input) {
                    ... on InferenceJob {   
                        id
                        pk
                        createdAt
                        updatedAt
                        completedAt
                        status
                        model {
                            name
                            slug
                            repoId
                            isTextInput
                            isTextOutput
                            isFileInput
                            isFileOutput
                        }
                        textInput
                        textOutput
                        wasGuarded
                        options
                    }
                }
            }
            """
        )

        input: Dict[str, str | float | list[str]] = {"inferenceJobId": inference_job_id}
        if status is not None:
            input["status"] = status
        if was_guarded is not None:
            input["wasGuarded"] = was_guarded
        if text_output is not None:
            input["textOutput"] = text_output
        if file_outputs_ids is not None or len(file_outputs_ids) > 0:
            input["fileOutputsIds"] = file_outputs_ids

        variables = {"input": input}

        try:
            result = self.spice.session.execute(mutation, variable_values=variables)
            return result
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if error.get("message", "") == "Inference Job not found.":
                        LOGGER.error(
                            f""" [*] Inference Job ID: {inference_job_id} not found.\
                                  Exiting early."""
                        )
                        return None
                raise exception
            else:
                raise exception

    def _get_filtered_options(
        self,
        options: Dict[str, Any],
        inference_options: Type[
            Union[
                InferenceOptionsForStableDiffusionPipeline,
                InferenceOptionsForStableDiffusionXLPipeline,
                InferenceOptionsForStableDiffusionXLImg2ImgPipeline,
            ]
        ],
    ) -> Dict[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key in inference_options.__annotations__
        }

    def _get_inference_options_for_stable_diffusion(
        self, pipeline: DiffusionPipeline, options: Dict[str, Any]
    ) -> Union[
        InferenceOptionsForStableDiffusionPipeline,
        InferenceOptionsForStableDiffusionXLPipeline,
        InferenceOptionsForStableDiffusionXLPipeline,
    ]:
        """
        Parses any inference options that may be defined for
        a stable diffusion pipeline.
        """

        if isinstance(pipeline, StableDiffusionPipeline):
            filtered_options = self._get_filtered_options(
                options, InferenceOptionsForStableDiffusionPipeline
            )
            return InferenceOptionsForStableDiffusionPipeline(**filtered_options)
        elif isinstance(pipeline, StableDiffusionXLPipeline):
            filtered_options = self._get_filtered_options(
                options, InferenceOptionsForStableDiffusionXLPipeline
            )
            return InferenceOptionsForStableDiffusionXLPipeline(**filtered_options)
        elif isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
            filtered_options = self._get_filtered_options(
                options, InferenceOptionsForStableDiffusionXLImg2ImgPipeline
            )
            return InferenceOptionsForStableDiffusionXLImg2ImgPipeline(
                **filtered_options
            )
        else:
            raise ValueError(
                f"Pipeline {type(pipeline).__name__} has no supported inference options!"  # noqa
            )

    def _get_generator(self, seed: int) -> torch.Generator:
        # Note, completely reproducible results are not guaranteed across
        # PyTorch releases.
        return torch.manual_seed(seed)

    def run_pipeline(
        self,
        inference_job_id: str,
    ):  # noqa
        LOGGER.info(f""" [*] Inference Job ID: {inference_job_id}.""")

        result = self._update_inference_job(
            inference_job_id=inference_job_id,
            status="RUNNING",
        )
        if not result:
            LOGGER.error(
                f""" [*] Inference Job ID: {inference_job_id} not found.\
                                  Exiting early."""
            )
            return

        inference_job_id = result["updateInferenceJob"]["id"]
        model_repo_id = result["updateInferenceJob"]["model"]["repoId"]
        text_input = result["updateInferenceJob"]["textInput"]
        is_text_input = result["updateInferenceJob"]["model"]["isTextInput"]
        is_text_output = result["updateInferenceJob"]["model"]["isTextOutput"]
        result["updateInferenceJob"]["model"]["isFileInput"]
        is_file_output = result["updateInferenceJob"]["model"]["isFileOutput"]
        options = result["updateInferenceJob"]["options"]

        LOGGER.info(f""" [*] Model: {model_repo_id}.""")
        LOGGER.info(f""" [*] Text Input: '{text_input}'""")

        variant = "fp16"
        torch_dtype = torch.float16
        if torch.backends.mps.is_available():
            variant = "fp32"
            torch_dtype = torch.float32
        #     mps_empty_cache()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        response = None
        try:
            if is_text_input and is_text_output:
                result = None
                pipe = transformers.pipeline(model=model_repo_id, device=self.device)
                result = pipe(text_input)

                response = self._update_inference_job(
                    inference_job_id=inference_job_id,
                    status="COMPLETE",
                    text_output=json.dumps(result),
                )
            elif is_text_input and is_file_output:
                SPICE_INFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                save_at = Path(SPICE_INFERENCE_DIRECTORY / f"{inference_job_id}.png")

                prompt = text_input
                negative_prompt = options.get("negative_prompt", "")
                generator = self._get_generator(int(options.get("seed", -1)))

                was_guarded = False
                if not save_at.exists():
                    pipe = DiffusionPipeline.from_pretrained(
                        model_repo_id,
                        torch_dtype=torch_dtype,
                        variant=variant,
                        use_safetensors=True,
                    )
                    pipe = pipe.to(self.device)  # type: ignore

                    # Get input for Stable Diffusion Pipelines
                    input_for_stable_diffusion_pipeline = (
                        get_input_for_stable_diffusion_pipeline(
                            self.device, pipe, prompt, negative_prompt
                        )
                    )

                    # Get callback options
                    callback_options = CallbackOptionsBase()

                    # Configure Stable Diffusion TASK
                    if isinstance(pipe, StableDiffusionPipeline):
                        # Configure inference options for stable diffusion pipeline
                        inference_options_for_stable_diffusion = (
                            self._get_inference_options_for_stable_diffusion(
                                pipe, options
                            )
                        )

                        # Configure output for stable diffusion pipeline
                        output_for_stable_diffusion_pipeline = OutputForStableDiffusionPipeline(
                            # generator=generator,
                            return_dict=False,
                        )

                        # Specify input for stable diffusion pipeline
                        stable_diffusion_pipeline_input = StableDiffusionPipelineInput(
                            input=input_for_stable_diffusion_pipeline,
                            inference_options=inference_options_for_stable_diffusion,
                            callback_options=callback_options,
                            output=output_for_stable_diffusion_pipeline,
                        )

                        pipe_result = pipe(
                            **asdict(stable_diffusion_pipeline_input.input),
                            **asdict(stable_diffusion_pipeline_input.inference_options),
                            **asdict(stable_diffusion_pipeline_input.callback_options),
                            **asdict(stable_diffusion_pipeline_input.output),
                            generator=generator,
                        )  # type:ignore
                    # Configure MOE for xl diffusion base + refinement TASK
                    elif isinstance(pipe, StableDiffusionXLPipeline):
                        # Configure input for stable diffusion xl pipeline
                        input_for_stable_diffusion_xl = (
                            InputForStableDiffusionXLPipeline(
                                **asdict(input_for_stable_diffusion_pipeline)
                            )
                        )

                        # Configure inference options for stable diffusion xl pipeline
                        inference_options_for_stable_diffusion_xl = (
                            self._get_inference_options_for_stable_diffusion(
                                pipe, options
                            )
                        )

                        if not isinstance(
                            inference_options_for_stable_diffusion_xl,
                            InferenceOptionsForStableDiffusionXLPipeline,
                        ):
                            raise ValueError(
                                f"Inference options are incorrectly configured for {StableDiffusionPipeline.__name__}"  # noqa
                            )

                        # Configure output for stable diffusion xl pipeline
                        output_for_stable_diffusion_xl_pipeline = OutputForStableDiffusionPipeline(  # noqa
                            output_type="latent",
                            # generator=generator,
                        )

                        # Specify input for stable diffusion xl pipeline
                        stable_diffusion_pipeline_xl_input = StableDiffusionXLPipelineInput(  # noqa
                            input=input_for_stable_diffusion_xl,
                            inference_options=inference_options_for_stable_diffusion_xl,
                            callback_options=callback_options,
                            output=output_for_stable_diffusion_xl_pipeline,
                        )

                        latents = pipe(
                            **asdict(stable_diffusion_pipeline_xl_input.input),
                            **asdict(
                                stable_diffusion_pipeline_xl_input.inference_options
                            ),
                            **asdict(
                                stable_diffusion_pipeline_xl_input.callback_options
                            ),
                            **asdict(stable_diffusion_pipeline_xl_input.output),
                            generator=generator,
                        ).images  # type: ignore

                        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-refiner-1.0",
                            text_encoder_2=pipe.text_encoder_2,
                            vae=pipe.vae,
                            torch_dtype=torch_dtype,
                            variant=variant,
                            use_safetensors=True,
                        )
                        refiner = refiner.to(self.device)

                        # Configure input for stable diffusion xl img2img pipeline
                        input_for_stable_diffusion_xl_img2img = (
                            InputForStableDiffusionXLImg2ImgPipeline(
                                image=latents, **asdict(input_for_stable_diffusion_xl)
                            )
                        )

                        # Configure inference options for stable diffusion xl img2img pipeline # noqa
                        inference_options_for_stable_diffusion_xl_img2img = (
                            self._get_inference_options_for_stable_diffusion(
                                refiner, options
                            )
                        )

                        if not isinstance(
                            inference_options_for_stable_diffusion_xl_img2img,
                            InferenceOptionsForStableDiffusionXLImg2ImgPipeline,
                        ):
                            raise ValueError(
                                f"Inference options are incorrectly configured for {StableDiffusionXLImg2ImgPipeline.__name__}"  # noqa
                            )

                        # Configure output for stable diffusion xl pipeline
                        output_for_stable_diffusion_xl_img2img_pipeline = OutputForStableDiffusionPipeline(  # noqa
                            # generator=generator,
                        )

                        # Specify input for stable diffusion xl pipeline
                        stable_diffusion_pipeline_xl_img2img_input = StableDiffusionXLImg2ImgPipelineInput(  # noqa
                            input=input_for_stable_diffusion_xl_img2img,
                            inference_options=inference_options_for_stable_diffusion_xl_img2img,
                            callback_options=callback_options,
                            output=output_for_stable_diffusion_xl_img2img_pipeline,
                        )

                        # TODO: Enforce better typing on derived class
                        temp_input = asdict(
                            stable_diffusion_pipeline_xl_img2img_input.inference_options
                        )
                        temp_input.pop("height")
                        temp_input.pop("width")

                        pipe_result = refiner(
                            **asdict(stable_diffusion_pipeline_xl_img2img_input.input),
                            **temp_input,
                            **asdict(
                                stable_diffusion_pipeline_xl_img2img_input.callback_options
                            ),
                            **asdict(stable_diffusion_pipeline_xl_img2img_input.output),
                            generator=generator,
                        )  # type: ignore
                    else:
                        raise ValueError(
                            f"Pipeline {type(pipe).__name__} has no supported inference options!"  # noqa
                        )

                    # pipe returns a tuple in the form the first element is a list with
                    # the generated images, and the second element is a list of `bool`s
                    # denoting whether the corresponding generated image likely
                    # represents "not-safe-for-work" (nsfw) content, according to the
                    # `safety_checker`.
                    result = pipe_result[0][0]  # type: ignore
                    if len(pipe_result) > 1 and pipe_result[1]:  # type: ignore
                        was_guarded = pipe_result[1][0]  # type: ignore
                    result.save(save_at)  # type: ignore
                else:
                    LOGGER.info(f""" [*] File already exists at: {save_at}""")

                upload_file_response = self.spice.uploader.upload_file_via_api(
                    path=save_at
                )
                file_id = upload_file_response.json()["data"]["uploadFile"]["id"]
                response = self._update_inference_job(
                    inference_job_id=inference_job_id,
                    status="COMPLETE",
                    file_outputs_ids=file_id,
                    was_guarded=was_guarded,
                )
            LOGGER.info(f""" [*] COMPLETE. Result: " {result}""")
            return response
        except PipelineException as exception:
            message = f"""Input: "{input}" threw exception: {exception}"""
            LOGGER.error(message)
            self._update_inference_job(
                inference_job_id=inference_job_id,
                status="ERROR",
                text_output=json.dumps(message),
            )
