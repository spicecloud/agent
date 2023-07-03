import json
import logging
import os
import ssl
import sys

import pika
from pika.exceptions import AMQPConnectionError, ConnectionClosedByBroker
from retry import retry

from spice_agent.utils.config import (
    SPICE_ROUND_VERIFICATION_FILEPATH,
    SPICE_TRAINING_FILEPATH,
    read_config_file,
)
from spice_agent.utils.version import get_current_version

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import transformers  # noqa

LOGGER = logging.getLogger(__name__)

ACTIVE_VERIFY_ROUND_STATUSES = [
    "CLAIMED",
    "DOWNLOADING_ROUND_MODEL",
    "DOWNLOADING_VERIFICATION_DATASET",
    "VERIFYING",
]
ACTIVE_TRAINING_ROUND_STEP_STATUSES = [
    "CLAIMED",
    "DOWNLOADING_MODEL",
    "DOWNLOADING_DATASET",
    "TRAINING",
    "UPLOADING_STEP_MODEL",
]
ACTIVE_TESTING_STEP_STATUSES = [
    "TRAINING_COMPLETE",
    "DOWNLOADING_TESTING_DATASET",
    "TESTING",
]
ACTIVE_UPLOADING_STEP_STATUSES = [
    "TESTING_COMPLETE",
    "REQUEST_UPLOAD",
    "UPLOADING",
]

ACTIVE_STATUSES = (
    ACTIVE_VERIFY_ROUND_STATUSES
    + ACTIVE_TRAINING_ROUND_STEP_STATUSES
    + ACTIVE_TESTING_STEP_STATUSES
    + ACTIVE_UPLOADING_STEP_STATUSES
)  # noqa


class Worker:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.channel = None
        self.device = self.spice.get_device()
        self.can_do_validation = "cuda" in self.device.type
        self.ACTIVE_STATUSES = ACTIVE_STATUSES

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

    def _claim_message_callback(self, channel, method, properties, body):
        """
        Based on if we get a run_id, training_round_step_id, training_round_id from the
        message consume
        """
        LOGGER.info(" [*] Processing message.")
        data = json.loads(body.decode("utf-8"))

        inference_job_id = data.get("inference_job_id", None)
        training_round_step_id = data.get("training_round_step_id", None)
        training_round_id = data.get("training_round_id", None)

        if (
            not inference_job_id
            and not training_round_step_id
            and not training_round_id
        ):
            raise Exception(
                f'No inference_job_id or training_round_step_id or training_round_id found in message body: {body.decode("utf-8")}'  # noqa
            )

        message_acked = False
        if inference_job_id:
            result = self.spice.inference._update_inference_job(
                inference_job_id=inference_job_id, status="CLAIMED"
            )
            # ack message at inference level so another machine does not steal the
            # message while inference is running
            channel.basic_ack(delivery_tag=method.delivery_tag)
            message_acked = True
            self.spice.inference.run_pipeline(inference_job_id=inference_job_id)
            LOGGER.info(" [*] Completed inference job.")

        if training_round_id:
            result = self.spice.training._update_training_round(
                training_round_id=training_round_id, status="CLAIMED"
            )
            if result is not None:
                LOGGER.info(" [*] Obtained training round.")

        if training_round_step_id:
            result = self.spice.training._update_training_round_step(
                training_round_step_id=training_round_step_id, status="CLAIMED"
            )
            if result is not None:
                LOGGER.info(" [*] Obtained training round step.")

        if not message_acked and channel.is_open:
            channel.basic_ack(delivery_tag=method.delivery_tag)
        else:
            LOGGER.info(" [*] Channel closed already. Cannot ack message.")

        LOGGER.info(" [*] Stopping worker...")
        if self.channel:
            self.channel.stop_consuming()
            self.channel.close()

    def _create_channel(self):
        credentials = pika.PlainCredentials(
            self.spice.host_config["fingerprint"],
            self.spice.host_config["rabbitmq_password"],
        )

        host = self.spice.host_config["rabbitmq_host"]
        ssl_options = None
        if "localhost" not in host:
            context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            context.verify_mode = ssl.CERT_NONE
            ssl_options = pika.SSLOptions(context)

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.spice.host_config["rabbitmq_host"],
                port=self.spice.host_config["rabbitmq_port"],
                virtual_host="agent",
                credentials=credentials,
                ssl_options=ssl_options,
            )
        )

        self.channel = connection.channel()
        LOGGER.info(" [*] Channel connected.")

    @retry(AMQPConnectionError, delay=0, jitter=(1, 3))
    def _consume(self):
        if self.channel is None or self.channel.is_closed:
            self._create_channel()

        if self.channel is None:
            raise Exception(" [*] Channel cannot be opened.")

        self.channel.basic_consume(
            queue="default",
            on_message_callback=self._claim_message_callback,
        )

        try:
            LOGGER.info(" [*] Waiting for messages. To exit press CTRL+C")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            LOGGER.info(" [*] Stopping RabbitMQ consumer...")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            sys.exit()
        except ConnectionClosedByBroker:
            LOGGER.info(" [*] Connection closed by Broker.")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
        except Exception as exception:
            LOGGER.info(f"Exception: {exception}")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            raise exception

    def start(self):
        LOGGER.info(" [*] ✨ spice worker")
        LOGGER.info(f" [*] Version: {get_current_version()}")
        try:
            while True:
                self.spice.hardware.check_in_http(is_available=True)

                # check if this machine picked up a training round already
                training_config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

                has_step_to_train = False
                has_step_to_test = False
                has_model_to_upload = False
                has_round_to_verify = False

                if training_config.get("id", None) is not None:
                    # check that training round step is active
                    step_status = training_config.get("status", None)
                    if step_status in ACTIVE_TRAINING_ROUND_STEP_STATUSES:
                        has_step_to_train = True
                    if step_status in ACTIVE_TESTING_STEP_STATUSES:
                        has_step_to_test = True
                    elif step_status in ACTIVE_UPLOADING_STEP_STATUSES:
                        has_model_to_upload = True

                if self.can_do_validation:
                    verification_config = read_config_file(
                        filepath=SPICE_ROUND_VERIFICATION_FILEPATH
                    )
                    if verification_config.get("id", None) is not None:
                        step_status = verification_config.get("status", None)
                        if step_status in ACTIVE_VERIFY_ROUND_STATUSES:
                            has_round_to_verify = True

                if (
                    not has_step_to_train
                    and not has_step_to_test
                    and not has_model_to_upload
                    and not has_round_to_verify
                ):
                    self._consume()

                if has_round_to_verify:
                    LOGGER.info(
                        f" [*] Validating - Using Device: {self.device} for validation"
                    )
                    self.spice.training.verify_model()
                if has_step_to_train:
                    LOGGER.info(
                        f" [*] Training - Using Device: {self.device} for training"
                    )
                    self.spice.training.train_model()
                if has_step_to_test:
                    LOGGER.info(
                        f" [*] Testing - Using Device: {self.device} for testing"
                    )
                    self.spice.training.test_model()
                if has_model_to_upload:
                    LOGGER.info(" [*] Uploading")
                    self.spice.training.upload_models()

        except KeyboardInterrupt:
            LOGGER.info(" [*] Stopping worker...")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            sys.exit()
