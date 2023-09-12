import json
import logging
import threading
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Optional, Dict, Any, List

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from gql import gql
from gql.transport.exceptions import TransportQueryError
from spice_agent.inference.tasks import (
    StableDiffusionXLPipelineTask,
)

from spice_agent.inference.types import (
    StableDiffusionPipelineTaskState,
)

from spice_agent.inference.factory import StableDiffusionTaskFactory

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

    def update_progress(self, state: StableDiffusionPipelineTaskState):
        progress = state.progress

        if state.progress:
            with self.update_inference_job_lock:
                self._update_inference_job(
                    inference_job_id=self.inference_job_id,
                    status_details={"progress": progress},
                )

    def update_image_preview(self, state: StableDiffusionPipelineTaskState):
        step = state.step
        image, was_guarded = self._reduce_stable_diffusion_output(state)

        if image and not was_guarded:
            file_name = f"{self.inference_job_id}-{step}-{IMAGE_GROUP_VALUE}.png"
            save_at = Path(SPICE_INFERENCE_DIRECTORY / file_name)

            image.save(save_at)
            upload_file_response = self.spice.uploader.upload_file_via_api(path=save_at)
            file_id = upload_file_response.json()["data"]["uploadFile"]["id"]

            with self.update_inference_job_lock:
                self._update_inference_job(
                    inference_job_id=self.inference_job_id,
                    file_outputs_ids=file_id,
                    was_guarded=was_guarded,
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

    def _reduce_stable_diffusion_output(
        self, task_result: StableDiffusionPipelineTaskState
    ):
        """
        Interprets stable diffusion task result

        TODO: add multi-image interpretation

        Returns (image, was_guarded)
        """

        image = None
        was_guarded = False
        task_result_output = task_result.output
        if isinstance(task_result_output, StableDiffusionPipelineOutput):
            nsfw_content_detected = task_result_output.nsfw_content_detected
            if nsfw_content_detected and nsfw_content_detected[0]:
                was_guarded = True
            else:
                image = task_result_output.images[0]
        elif isinstance(task_result_output, StableDiffusionXLPipelineOutput):
            image = task_result_output.images[0]
        elif task_result_output:  # no image is fine
            message = f"No interpretation of task_result_output found: {type(task_result_output)}"  # noqa
            LOGGER.error(message)
            raise NotImplementedError(message)

        return (image, was_guarded)

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

        # TODO: Add cache cleanup logic

        # Prepare file cache if input files are used
        file_input_paths: List[Path] = []
        if is_file_input and file_inputs:
            file_input_paths = self.download_file_inputs(file_inputs)

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
                # Some of the inference options belong to task_config
                task_config: dict[str, Any] = {}
                task_config["is_control"] = options.pop("is_control", False)
                task_config["control_types"] = options.pop("control_types", [])

                inference_options = options
                inference_options["prompt"] = text_input
                generator = self._get_generator(int(options.get("seed", -1)))

                max_step = options.get("num_inference_steps", 999)
                SPICE_INFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                file_name = f"{inference_job_id}-{max_step}-{IMAGE_GROUP_VALUE}.png"
                save_at = Path(SPICE_INFERENCE_DIRECTORY / file_name)
                was_guarded = False

                if not save_at.exists():
                    # Build images
                    images: List[PIL.Image.Image] = []

                    for file_input_path in file_input_paths:
                        image = self.load_image(file_input_path)
                        if image:
                            # TODO: add some sort of image formatting
                            # constraints, channels and size influence GPU
                            # requirements
                            # image = image.convert("RGB").resize((512, 512))
                            image = image.convert("RGB")
                            images.append(image)

                    if images:
                        task_config["is_file_input_paths"] = True
                        inference_options["image"] = images

                    task_factory = StableDiffusionTaskFactory()
                    task = task_factory.create_task(
                        device=self.device,
                        model_repo_id=model_repo_id,
                        task_config=task_config,
                        inference_options=inference_options,
                    )

                    # Configure Stable Diffusion TASK
                    if task:
                        # Useful for debugging TASK
                        # print(task.serialize_pipeline_run_config())

                        task.observer.add_observer(self.update_progress)
                        task.observer.add_observer(self.update_image_preview)

                        # Reconfigure for refinement if available
                        if isinstance(task, StableDiffusionXLPipelineTask):
                            (
                                base_task,
                                refinement_task,
                            ) = task_factory.configure_ensemble_of_denoisers(
                                task,
                                prompt=inference_options["prompt"],
                                negative_prompt=inference_options["negative_prompt"],
                            )

                            refinement_task.pipeline_run_config.image = (
                                base_task.run_pipeline().output.images
                            )

                            refinement_task.observer.add_observer(self.update_progress)

                            task_result = refinement_task.run_pipeline(
                                generator=generator
                            )
                        # Run single task
                        else:
                            task_result = task.run_pipeline(generator=generator)
                    else:
                        message = f"No task found for {model_repo_id} with task configuration: {task_config}"  # noqa
                        LOGGER.error(message)
                        raise ValueError(message)

                    image, was_guarded = self._reduce_stable_diffusion_output(
                        task_result=task_result
                    )

                    if not was_guarded and image:
                        image.save(save_at)
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
