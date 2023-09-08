import json
import logging
import threading
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Optional, Dict, Any, List

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

from gql import gql
from gql.transport.exceptions import TransportQueryError
from spice_agent.inference.tasks import (
    StableDiffusionPipelineTask,
    StableDiffusionImg2ImgPipelineTask,
    StableDiffusionXLPipelineTask,
    StableDiffusionXLImg2ImgPipelineTask,
)

from spice_agent.inference.types import (
    TStableDiffusionPipelineState,
    StableDiffusionXLPipelineState,
    StableDiffusionPipelineState,
)

from spice_agent.utils.config import SPICE_INFERENCE_DIRECTORY, create_directory

# from torch.mps import empty_cache as mps_empty_cache ## SAVE FOR LATER

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
import transformers  # noqa
from transformers.pipelines.base import PipelineException  # noqa

LOGGER = logging.getLogger(__name__)

# TODO: Currently, we only support single image outputs for
# Stable Diffusion pipelines. This value is a placeholder for when
# multi-image outputs are added
IMAGE_GROUP_VALUE = 0

SPICE_INFERENCE_FILE_INPUTS_DIRECTORY = Path(SPICE_INFERENCE_DIRECTORY / "file-inputs")
create_directory(SPICE_INFERENCE_FILE_INPUTS_DIRECTORY)


class Inference:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.spice.get_device()
        self.pipe = None
        self.inference_job_id = None

        # Threading
        self.progress_thread = None
        self.image_preview_thread = None
        self.update_inference_job_lock = threading.Lock()

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()  # type: ignore
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

    def _get_presigned_url(self, file_id: str):
        query = gql(
            """
            query GetFile($fileId: GlobalID!) {
                getFile(fileId: $fileId) {
                    ... on File {
                        fileName
                        presignedUrl
                    }
                }
            }
            """
        )

        variables: dict[str, Any] = {"fileId": file_id}

        try:
            result = self.spice.session.execute(query, variable_values=variables)
            return result
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if error.get("message", "") == "File not found.":
                        LOGGER.error(
                            f""" [*] File ID: {file_id} not found.\
                                  Exiting early."""
                        )
                        return None
                raise exception
            else:
                raise exception

    def _update_inference_job(
        self,
        inference_job_id: str,
        status: Optional[str] = None,
        status_details: Optional[Dict[str, Any]] = None,
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
                        statusDetails
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
                        fileInputs {
                            id
                        }
                        wasGuarded
                        options
                    }
                }
            }
            """
        )

        input: Dict[str, Any] = {"inferenceJobId": inference_job_id}
        if status is not None:
            input["status"] = status
        if status_details is not None:
            input["statusDetails"] = status_details
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

    def _get_generator(self, seed: int) -> torch.Generator:
        # Note, completely reproducible results are not guaranteed across
        # PyTorch releases.
        return torch.Generator(device="cpu").manual_seed(seed)

    def download_file_inputs(self, file_inputs: List[Dict[str, str]]) -> List[Path]:
        """
        For each file input, a corresponding presigned url is fetched
        and then used to cache a file input. The locations of each file is returned.
        """

        file_input_paths: List[Path] = []

        for file in file_inputs:
            file_id = file["id"]

            get_file_result = self._get_presigned_url(file_id)
            if get_file_result:
                file_name = get_file_result["getFile"]["fileName"]
                presigned_url = get_file_result["getFile"]["presignedUrl"]
            else:
                message = f"Unable to get presigned url for {file_id}"
                raise ValueError(message)

            response = requests.get(presigned_url)
            response.raise_for_status()
            file_path = SPICE_INFERENCE_FILE_INPUTS_DIRECTORY.joinpath(file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
                file_input_paths.append(file_path)

        return file_input_paths

    def update_progress(self, state: TStableDiffusionPipelineState):
        progress = state.progress

        if state.progress:
            with self.update_inference_job_lock:
                self._update_inference_job(
                    inference_job_id=self.inference_job_id,
                    status_details={"progress": progress},
                )

    def update_image_preview_for_stable_diffusion(
        self, state: StableDiffusionPipelineState
    ):
        step = state.step
        output = state.output

        if output:
            file_name = f"{self.inference_job_id}-{step}-{IMAGE_GROUP_VALUE}.png"
            save_at = Path(SPICE_INFERENCE_DIRECTORY / file_name)

            image = output[0][0]
            image.save(save_at)
            if len(output) > 1 and output[1]:  # type: ignore
                was_guarded = output[1][0]
            else:
                was_guarded = False

            upload_file_response = self.spice.uploader.upload_file_via_api(path=save_at)
            file_id = upload_file_response.json()["data"]["uploadFile"]["id"]

            with self.update_inference_job_lock:
                self._update_inference_job(
                    inference_job_id=self.inference_job_id,
                    file_outputs_ids=file_id,
                    was_guarded=was_guarded,
                )

    def update_image_preview_for_stable_diffusion_xl(
        self, state: StableDiffusionXLPipelineState
    ):
        step = state.step
        output = state.output

        if output:
            file_name = f"{self.inference_job_id}-{step}-{IMAGE_GROUP_VALUE}.png"
            save_at = Path(SPICE_INFERENCE_DIRECTORY / file_name)

            image = output[0][0]
            image.save(save_at)

            upload_file_response = self.spice.uploader.upload_file_via_api(path=save_at)
            file_id = upload_file_response.json()["data"]["uploadFile"]["id"]

            with self.update_inference_job_lock:
                self._update_inference_job(
                    inference_job_id=self.inference_job_id,
                    file_outputs_ids=file_id,
                )

    def load_image(self, file_path: Path) -> Optional[PIL.Image.Image]:
        try:
            # Open the image file
            image = PIL.Image.open(file_path)
            return image
        except FileNotFoundError as exception:
            message = f"load_image: File not found - {exception}"
            LOGGER.error(message)
        except IOError as exception:
            message = f"load_image: Input/Output error - {exception}"
            LOGGER.error(message)
        except Exception as exception:
            message = f"load_image: An unexpected error occurred - {exception}"
            LOGGER.error(message)
            raise exception

        return None

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
        self.inference_job_id = inference_job_id

        model_repo_id = result["updateInferenceJob"]["model"]["repoId"]
        text_input = result["updateInferenceJob"]["textInput"]
        file_inputs = result["updateInferenceJob"]["fileInputs"]
        is_text_input = result["updateInferenceJob"]["model"]["isTextInput"]
        is_text_output = result["updateInferenceJob"]["model"]["isTextOutput"]
        result["updateInferenceJob"]["model"]["isFileInput"]
        is_file_output = result["updateInferenceJob"]["model"]["isFileOutput"]
        is_file_input = result["updateInferenceJob"]["model"]["isFileInput"]
        options = result["updateInferenceJob"]["options"]

        LOGGER.info(f""" [*] Model: {model_repo_id}.""")
        LOGGER.info(f""" [*] Text Input: '{text_input}'""")

        # Prepare file cache if input files are used
        file_input_paths: List[Path] = []
        if is_file_input and file_inputs:
            file_input_paths = self.download_file_inputs(file_inputs)

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
                options["prompt"] = text_input
                generator = self._get_generator(int(options.get("seed", -1)))

                max_step = options.get("num_inference_steps", 999)
                SPICE_INFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                file_name = f"{inference_job_id}-{max_step}-{IMAGE_GROUP_VALUE}.png"
                save_at = Path(SPICE_INFERENCE_DIRECTORY / file_name)
                was_guarded = False

                if not save_at.exists():
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_repo_id,
                        torch_dtype=torch_dtype,
                        variant=variant,
                        use_safetensors=True,
                    )
                    self.pipeline = pipeline

                    # Build images
                    images: List[PIL.Image.Image] = []

                    for file_input_path in file_input_paths:
                        image = self.load_image(file_input_path)
                        if image:
                            images.append(image)

                    options["image"] = images

                    # Configure Stable Diffusion TASK
                    if isinstance(self.pipeline, StableDiffusionPipeline):
                        task_options = options.copy()
                        task_options["return_dict"] = False

                        # Configure Stable Diffusion Img2Img Task
                        if options["image"]:
                            self.pipeline = (
                                StableDiffusionImg2ImgPipeline.from_pretrained(
                                    model_repo_id,
                                    torch_dtype=torch_dtype,
                                    variant=variant,
                                    use_safetensors=True,
                                )
                            )

                            task = StableDiffusionImg2ImgPipelineTask(
                                self.pipeline, self.device, task_options
                            )
                        else:
                            task = StableDiffusionPipelineTask(
                                self.pipeline, self.device, task_options
                            )

                        task.add_observer(
                            self.update_image_preview_for_stable_diffusion
                        )

                        task.add_observer(self.update_progress)

                        pipe_result = task.run(
                            generator=generator,
                        )
                        print(pipe_result)
                    # Configure StableDiffusionXLImg2ImgPipeline Task
                    elif (
                        isinstance(self.pipeline, StableDiffusionXLPipeline)
                        and options["image"]
                    ):
                        task_options = options.copy()

                        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-refiner-1.0",
                            torch_dtype=torch_dtype,
                            variant=variant,
                            use_safetensors=True,
                        )
                        self.pipeline = pipeline

                        task = StableDiffusionXLImg2ImgPipelineTask(
                            self.pipeline, self.device, task_options
                        )

                        task.add_observer(self.update_progress)

                        pipe_result = task.run(generator=generator)
                    # Configure MOE for xl diffusion base TASK + refinement TASK
                    elif isinstance(self.pipeline, StableDiffusionXLPipeline):
                        task_options = options.copy()
                        task_options["output_type"] = "latent"
                        task_options["denoising_end"] = 0.8
                        task = StableDiffusionXLPipelineTask(
                            self.pipeline, self.device, task_options
                        )

                        task.add_observer(
                            self.update_image_preview_for_stable_diffusion_xl
                        )
                        task.add_observer(self.update_progress)

                        latents = task.run(
                            generator=generator,
                        ).images

                        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-refiner-1.0",
                            text_encoder_2=self.pipeline.text_encoder_2,
                            vae=self.pipeline.vae,
                            torch_dtype=torch_dtype,
                            variant=variant,
                            use_safetensors=True,
                        )

                        refiner_task_options = options.copy()
                        refiner_task_options["image"] = latents
                        refiner_task_options["output_type"] = "pil"
                        refiner_task_options["denoising_start"] = 0.8
                        refiner_task_options["return_dict"] = False
                        refiner_task = StableDiffusionXLImg2ImgPipelineTask(
                            refiner, self.device, refiner_task_options
                        )

                        pipe_result = refiner_task.run(
                            generator=generator,
                        )
                    else:
                        raise ValueError(
                            f"Pipeline {type(self.pipeline).__name__} has no supported inference options!"  # noqa
                        )

                    # pipe returns a tuple in the form the first element is a list with
                    # the generated images, and the second element is a list of `bool`s
                    # denoting whether the corresponding generated image likely
                    # represents "not-safe-for-work" (nsfw) content, according to the
                    # `safety_checker`.
                    # TODO decode output based on output of actual pipeline
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

                # Cleanup threads
                if self.progress_thread:
                    self.progress_thread.join()

                if self.image_preview_thread:
                    self.image_preview_thread.join()

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
