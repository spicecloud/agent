import json
import logging
import os
from typing import Dict, Optional

from gql import gql
from gql.transport.exceptions import TransportQueryError

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import transformers  # noqa
from transformers import pipeline, set_seed  # noqa
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
        text_output: Optional[str] = None,
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
                            slug
                        }
                        textInput
                        textOutput
                        isTextInput
                        isTextOutput
                        isFileInput
                        isFileOutput
                    }
                }
            }
            """
        )

        input: Dict[str, str | float] = {"inferenceJobId": inference_job_id}
        if status is not None:
            input["status"] = status
        if text_output is not None:
            input["textOutput"] = text_output

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

        model = result["updateInferenceJob"]["model"]["slug"]
        text_input = result["updateInferenceJob"]["textInput"]
        LOGGER.info(f""" [*] Model: {model}.""")
        LOGGER.info(f""" [*] Text Input: '{text_input}'""")

        try:
            result = None
            pipe = pipeline(model=model, device=self.device)
            result = pipe(text_input)

            LOGGER.info(f""" [*] SUCCESS. Result: " {result}""")
            self._update_inference_job(
                inference_job_id=inference_job_id,
                status="SUCCESS",
                text_output=json.dumps(result),
            )
        except PipelineException as exception:
            message = f"""Input: "{input}" threw exception: {exception}"""
            LOGGER.error(message)
            self._update_inference_job(
                inference_job_id=inference_job_id,
                status="ERROR",
                text_output=json.dumps(message),
            )
