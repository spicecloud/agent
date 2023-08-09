import json
import logging
import os
from compel import Compel, ReturnedEmbeddingsType
from pathlib import Path
from typing import Optional, Dict, Any

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from gql import gql
from gql.transport.exceptions import TransportQueryError

from spice_agent.utils.config import SPICE_INFERENCE_DIRECTORY

# from torch.mps import empty_cache as mps_empty_cache ## SAVE FOR LATER


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
import transformers  # noqa
from transformers.pipelines.base import PipelineException  # noqa

LOGGER = logging.getLogger(__name__)


class CompelEmbeddingsProvider:
    """
    Provides embeddings for a pipeline
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        device,
        prompt: str,
        negative_prompt: str = "",
    ) -> None:
        self.pipeline = pipeline
        self.device = device
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.compel = self._init_compel()

    def _init_compel(self) -> Compel:
        """
        Initializes compel text embedding system based on type of pipeline.
        """

        # Compel settings consistent across compel objects
        DEFAULT_COMPEL_ARGUMENTS = {
            "truncate_long_prompts": False,
            "device": self.device,
        }

        pipe = self.pipeline

        # Currently, these are the only supported pipelines
        if not isinstance(
            pipe,
            (
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
                StableDiffusionXLImg2ImgPipeline,
            ),
        ):
            message = f"Prompt embeddings are not supported for pipeline {type(pipe)}"
            LOGGER.warn(message)
            return None

        # Check if embeddings are necessary
        if pipe.tokenizer:
            model_max_length = pipe.tokenizer.model_max_length
            if (
                len(self.prompt) <= model_max_length
                and len(self.negative_prompt) <= model_max_length
            ):
                return None

        # Configure compel objects for StableDiffusion pipeline
        if isinstance(pipe, StableDiffusionPipeline):
            return Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                requires_pooled=False,
                **DEFAULT_COMPEL_ARGUMENTS,
            )

        # Configure compel objects for StableDiffusionXL pipelines
        if isinstance(pipe, StableDiffusionXLPipeline):
            return Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                **DEFAULT_COMPEL_ARGUMENTS,
            )

        if isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
            return Compel(
                tokenizer=pipe.tokenizer_2,
                text_encoder=pipe.text_encoder_2,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
                **DEFAULT_COMPEL_ARGUMENTS,
            )

        return None

    def _run_compel(self) -> dict:
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if self.compel.requires_pooled:
            # Get pooled embeddings
            prompt_embeds, pooled_prompt_embeds = self.compel(self.prompt)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.compel(
                self.negative_prompt
            )
        else:
            prompt_embeds = self.compel(self.prompt)
            negative_prompt_embeds = self.compel(self.negative_prompt)

        # Pad prompt embeds
        [
            padded_prompt_embeds,
            padded_negative_prompt_embeds,
        ] = self.compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )

        embeddings = {
            "prompt_embeds": padded_prompt_embeds,
            "negative_prompt_embeds": padded_negative_prompt_embeds,
        }

        if pooled_prompt_embeds is not None:
            embeddings["pooled_prompt_embeds"] = pooled_prompt_embeds

        if negative_pooled_prompt_embeds is not None:
            embeddings["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        return embeddings

    def __call__(self) -> dict:
        """
        Returns prompt embeddings if a compel object exists; otherwise,
        no embeddings are returned
        """

        LOGGER.info(""" [*] Generating embeddings. (Ignore token indices complaint)""")

        self.without_embeddings = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
        }

        if self.compel:
            try:
                return self._run_compel()
            except Exception as exception:
                LOGGER.error(
                    f""" [*] Embedding generation failed with exception {exception}! Falling back to normal prompt."""  # noqa
                )

        return self.without_embeddings


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

    def _get_stable_diffusion_options(self, options: Dict[str, Any]) -> dict:
        """
        Parses any inference options that may be defined for
        StableDiffusionPipeline
        """

        stable_diffusion_options: dict = {}

        if "negative_prompt" in options:
            stable_diffusion_options["negative_prompt"] = options["negative_prompt"]

        if "guidance_scale" in options:
            stable_diffusion_options["guidance_scale"] = options["guidance_scale"]

        if "num_inference_steps" in options:
            stable_diffusion_options["num_inference_steps"] = options[
                "num_inference_steps"
            ]

        if "seed" in options:
            # Note, completely reproducible results are not guaranteed across
            # PyTorch releases.
            stable_diffusion_options["generator"] = torch.manual_seed(
                int(options["seed"])
            )

        return stable_diffusion_options

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

                stable_diffusion_options = self._get_stable_diffusion_options(options)
                prompt = text_input
                negative_prompt = ""
                if "negative_prompt" in stable_diffusion_options:
                    negative_prompt = stable_diffusion_options["negative_prompt"]
                    # We remove the negative prompt from options since we will
                    # track it in prompt embeddings
                    stable_diffusion_options.pop("negative_prompt")

                was_guarded = False
                if not save_at.exists():
                    pipe = DiffusionPipeline.from_pretrained(
                        model_repo_id,
                        torch_dtype=torch_dtype,
                        variant=variant,
                        use_safetensors=True,
                    )
                    pipe = pipe.to(self.device)  # type: ignore

                    # Generates prompt embeddings for Stable Diffusion Pipelines
                    prompt_embeddings = CompelEmbeddingsProvider(
                        pipeline=pipe,
                        device=self.device,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                    )()

                    # Configure MOE for xl diffusion base + refinement
                    if (
                        isinstance(pipe, StableDiffusionXLPipeline)
                        and "stabilityai/stable-diffusion-xl-base-1.0"
                    ):
                        latents = pipe(
                            output_type="latent",
                            **stable_diffusion_options,
                            **prompt_embeddings,
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

                        # Generates additional embeddings for refiner
                        prompt_embeddings_for_refiner = CompelEmbeddingsProvider(
                            pipeline=refiner,
                            device=self.device,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                        )()

                        pipe_result = refiner(
                            image=latents,  # type: ignore
                            **stable_diffusion_options,
                            **prompt_embeddings_for_refiner,
                        )  # type: ignore
                    else:
                        pipe_result = pipe(
                            return_dict=False,
                            **stable_diffusion_options,
                            **prompt_embeddings,
                        )  # type:ignore

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
