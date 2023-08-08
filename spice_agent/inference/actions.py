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
from transformers import pipeline  # noqa
from transformers.pipelines.base import PipelineException  # noqa

LOGGER = logging.getLogger(__name__)


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

    def _run_compel(self, compel: Compel, prompt: str, negative_prompt: str):
        prompt_embeds = compel(prompt)
        negative_prompt_embeds = compel(negative_prompt)

        # Pad prompt embeds
        [
            padded_prompt_embeds,
            padded_negative_prompt_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )

        return (padded_prompt_embeds, padded_negative_prompt_embeds)

    def _run_compel_with_pooling(
        self, compel: Compel, prompt: str, negative_prompt: str
    ):
        # Get pooled embeddings
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, pooled_negative_prompt_embeds = compel(negative_prompt)

        # Pad prompt embeds
        [
            padded_prompt_embeds,
            padded_negative_prompt_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )

        return (
            padded_prompt_embeds,
            pooled_prompt_embeds,
            padded_negative_prompt_embeds,
            pooled_negative_prompt_embeds,
        )

    def _get_prompt_embeddings(
        self, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str = ""
    ) -> dict:
        """
        Generates prompt embeddings for prompt and negative prompt. If this is not
        possible, we simply return a dictionary containing the
        prompt and negative prompt
        """

        compel = None
        without_embeddings = {"prompt": prompt, "negative_prompt": negative_prompt}

        if not isinstance(
            pipeline,
            (
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
                StableDiffusionXLImg2ImgPipeline,
            ),
        ):
            message = (
                f"prompt embeddings are not supported for pipeline {type(pipeline)}"
            )
            LOGGER.warn(message)
            return without_embeddings

        # Check if we even need to do an embedding
        if pipeline.tokenizer:
            model_max_length = pipeline.tokenizer.model_max_length
            if (
                len(prompt) <= model_max_length
                and len(negative_prompt) <= model_max_length
            ):
                return without_embeddings

        if isinstance(pipeline, StableDiffusionPipeline):
            compel = Compel(
                truncate_long_prompts=False,
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=False,
            )

            prompt_embeds, negative_prompt_embeds = self._run_compel(
                compel, prompt, negative_prompt
            )

            return {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            }

        # Configure compel objects for StableDiffusionXL pipelines
        if isinstance(pipeline, StableDiffusionXLPipeline):
            compel = Compel(
                truncate_long_prompts=False,
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )

        if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
            compel = Compel(
                truncate_long_prompts=False,
                tokenizer=pipeline.tokenizer_2,
                text_encoder=pipeline.text_encoder_2,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
            )

        if compel and isinstance(
            pipeline, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)
        ):
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self._run_compel_with_pooling(compel, prompt, negative_prompt)

            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            }

        return without_embeddings

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
                pipe = pipeline(model=model_repo_id, device=self.device)
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

                    # Configure prompt embeddings
                    prompt_embeddings = self._get_prompt_embeddings(
                        pipe, prompt, negative_prompt=negative_prompt
                    )

                    # Configure MOE for xl diffusion base + refinement
                    if (
                        isinstance(pipe, StableDiffusionXLPipeline)
                        and "stabilityai/stable-diffusion-xl-base-1.0"
                    ):
                        # arguments
                        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-refiner-1.0",
                            text_encoder_2=pipe.text_encoder_2,
                            vae=pipe.vae,
                            torch_dtype=torch_dtype,
                            variant=variant,
                            use_safetensors=True,
                        )

                        refiner = refiner.to(self.device)

                        latents = pipe(
                            output_type="latent",
                            **stable_diffusion_options,
                            **prompt_embeddings,
                        ).images  # type: ignore

                        prompt_embeddings_for_refiner = self._get_prompt_embeddings(
                            refiner, prompt, negative_prompt=negative_prompt
                        )

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
