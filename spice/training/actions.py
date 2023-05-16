import hashlib
import json
import logging
import os
import platform
import ssl
import sys
from typing import Optional

from datasets import load_dataset
import evaluate
from gql import gql
import numpy as np
import pika
from pika.exceptions import AMQPConnectionError, ConnectionClosedByBroker
from retry import retry
import torch  # noqa
from torch.mps import empty_cache
from torch.optim import AdamW  # noqa
from torch.utils.data import DataLoader  # noqa
from transformers import (  # noqa
    AutoImageProcessor,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
    pipeline,
)

from spice.utils.config import (
    SPICE_MODEL_CACHE_FILEPATH,
    SPICE_TRAINING_FILEPATH,
    create_directory,
    read_config_file,
    update_config_file,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
from torch.optim import AdamW  # noqa
from torch.utils.data import DataLoader  # noqa
import transformers  # noqa

LOGGER = logging.getLogger(__name__)

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


MODEL_BUCKET_NAME = "spice-models"


class Training:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.get_device()
        self.channel = None

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

    def get_device(self):
        """
        First check if mps is available as a device
        Then check for a CUDA device
        Finally, fall back to CPU
        """
        device = None
        os_family = platform.system()

        # mps device enables high-performance training on GPU for macOS
        # devices with Metal programming framework
        # https://pytorch.org/docs/master/notes/mps.html
        if os_family == "Darwin" and torch.backends.mps.is_available():  # type: ignore
            device = torch.device("mps")
            empty_cache()
            if self.spice.DEBUG:
                print("Using MPS device.")
        else:
            if device is None and self.spice.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.mps.is_built():  # type: ignore
                    print(
                        "MPS not available because the current PyTorch install was not built with MPS enabled."  # noqa
                    )
                else:
                    print(
                        "MPS not available because the current macOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."  # noqa
                    )

        if device is None and torch.cuda.is_available():
            device = torch.device("cuda:0")
            if self.spice.DEBUG:
                print("Using CUDA device.")
        else:
            if device is None and self.spice.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.cuda.is_built():  # type: ignore
                    print(
                        "CUDA not available because the current PyTorch install was not built with CUDA enabled."  # noqa
                    )
                else:
                    print(
                        "CUDA not available because the current you do not have an CUDA-enabled device on this machine."  # noqa
                    )

        if device is None:
            # fallback to CPU
            device = torch.device("cpu")
            if self.spice.DEBUG:
                print("Using cpu.")

        return device

    def _update_training_round_step(
        self, training_round_step_id: str, status: str, file_id: Optional[str] = None
    ):
        mutation = gql(
            """
            mutation updateTrainingRoundStepFromHardware($trainingRoundStepId: String!, $status: String!, $fileId: String) {
                updateTrainingRoundStepFromHardware(trainingRoundStepId: $trainingRoundStepId, status: $status, fileId: $fileId) {
                    id
                    status
                    baseModel
                    baseModelRevision
                    baseDatasetRepoId
                    baseDatasetRepoRevision
                    datasetStartingRow
                    datasetEndingRow
                    trainingEpochs
                    trainingBatchSize
                    trainingRound {
                        id
                    }
                }
            }
        """  # noqa
        )
        variables = {"trainingRoundStepId": training_round_step_id}
        if status is not None:
            variables["status"] = status
        if file_id is not None:
            variables["fileId"] = file_id
        result = self.spice.session.execute(mutation, variable_values=variables)
        update_config_file(
            filepath=SPICE_TRAINING_FILEPATH,
            new_config=result["updateTrainingRoundStepFromHardware"],
        )
        return result

    def _claim_training_round_step_callback(self, channel, method, properties, body):
        print(" [*] Processing message.")
        data = json.loads(body.decode("utf-8"))
        training_round_step_id = data["training_round_step_id"]
        if not training_round_step_id:
            raise Exception(
                f'No training_round_step_id found in message body: {body.decode("utf-8")}'  # noqa
            )
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="CLAIMED"
        )
        print(" [*] Obtained training round step.")

        if channel.is_open:
            channel.basic_ack(delivery_tag=method.delivery_tag)
        else:
            print(" [*] Channel closed already. Cannot ack message.")

        print(" [*] Stopping worker...")
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
                virtual_host="training",
                credentials=credentials,
                ssl_options=ssl_options,
            )
        )

        self.channel = connection.channel()
        print(" [*] Channel connected.")

    @retry(AMQPConnectionError, delay=0, jitter=(1, 3))
    def _consume(self):
        if self.channel is None or self.channel.is_closed:
            self._create_channel()

        # self.channel.queue_declare("inference", durable=True, auto_delete=False)
        self.channel.basic_consume(
            queue="default",
            on_message_callback=self._claim_training_round_step_callback,
        )
        try:
            print(" [*] Waiting for messages. To exit press CTRL+C")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print(" [*] Stopping RabbitMQ consumer...")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            sys.exit()
        except ConnectionClosedByBroker:
            print(" [*] Connection closed by Broker.")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
        except Exception as exception:
            print(f"Exception: {exception}")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            raise exception

    def upload_models(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        training_round_id = config["trainingRound"]["id"]
        training_round_directory = f"{training_round_id}/steps/"

        # round_id_step_id will be in form: [uuid]/steps/[uuid]
        # round_id_step_id is used for finding the step model in the cache
        # and as the key in S3
        bucket_key = f"{training_round_id}/steps/{training_round_step_id}"

        # model_cache_for_training_round contains all the step models
        # for this particular training round
        model_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            training_round_directory
        )

        # step_model_path IS the trained step model
        step_model_path = model_cache_for_training_round.joinpath(
            f"{training_round_step_id}.pt"
        )

        # create the "file" and attach it to the training round step
        file_checksum = hashlib.md5(step_model_path.read_bytes()).hexdigest()
        file_id = self.spice.uploader._create_file(
            file_name=step_model_path.name,
            file_size=step_model_path.stat().st_size,
            file_checksum=file_checksum,
        )
        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="REQUEST_UPLOAD",
            file_id=file_id,
        )
        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="UPLOADING",
        )

        self.spice.uploader.upload_file(
            bucket_name=MODEL_BUCKET_NAME,
            key=bucket_key,
            filepath=step_model_path,
            file_id=file_id,
        )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="COMPLETE"
        )

    def train_model(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="CLAIMED"
        )
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

        base_model = config["baseModel"]
        base_model_revision = config["baseModelRevision"]
        dataset_repo_id = config["baseDatasetRepoId"]
        dataset_repo_revision = config["baseDatasetRepoRevision"]
        dataset_starting_row = config["datasetStartingRow"]
        dataset_ending_row = config["datasetEndingRow"]
        config["trainingEpochs"]
        training_batch_size = config["trainingBatchSize"]
        training_round_id = config["trainingRound"]["id"]

        # create the folder for the new training round
        # where the step model will be saved
        training_round_directory = f"{training_round_id}/steps/"
        model_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            training_round_directory
        )
        create_directory(filepath=model_cache_for_training_round)

        # get dataset
        print("Loading dataset...")
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="DOWNLOADING_DATASET"
        )

        split_string = f"train[{dataset_starting_row}:{dataset_ending_row}]"
        train_dataset = load_dataset(
            path=dataset_repo_id, revision=dataset_repo_revision, split=split_string
        )

        # get tokenizer from base model bert-base-cased
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_model, revision=base_model_revision
        )

        # create a tokenize function that will tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        print("Tokenizing dataset...")
        tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

        # Remove the text column because the model does not accept raw text as an input
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])

        # Rename the label column to labels because
        # the model expects the argument to be named labels
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

        # Set the format of the dataset to return PyTorch tensors instead of lists
        tokenized_datasets.set_format("torch")

        # Load your model with the number of expected labels:
        print("Loading base model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=5
        )

        training_args = TrainingArguments(
            output_dir=str(model_cache_for_training_round),
            evaluation_strategy="no",  # set to "no", set to "epoch" for eval + train
            num_train_epochs=1,
            per_device_train_batch_size=training_batch_size,
            use_mps_device=torch.backends.mps.is_available(),  # type: ignore
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(output_dir=str(model_cache_for_training_round))

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TESTING"
        )

        # clear the cache
        train_dataset.cleanup_cache_files()

    def test_model(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="CLAIMED"
        )
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

        base_model = config["baseModel"]
        base_model_revision = config["baseModelRevision"]
        dataset_repo_id = config["baseDatasetRepoId"]
        dataset_repo_revision = config["baseDatasetRepoRevision"]
        config["datasetStartingRow"]
        config["datasetEndingRow"]
        training_batch_size = config["trainingBatchSize"]
        training_round_id = config["trainingRound"]["id"]

        # create the folder for the new training round
        # where the step model will be saved
        training_round_directory = f"{training_round_id}/steps/"
        model_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            training_round_directory
        )

        # get dataset
        print("Loading dataset...")
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="DOWNLOADING_DATASET"
        )

        test_dataset = load_dataset(
            path=dataset_repo_id, revision=dataset_repo_revision, split="test"
        )

        # get tokenizer from base model bert-base-cased
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_model, revision=base_model_revision
        )

        # create a tokenize function that will tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        print("Tokenizing dataset...")
        tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

        # Remove the text column because the model does not accept raw text as an input
        tokenized_test_datasets = tokenized_test_datasets.map(
            tokenize_function, batched=True
        )

        # Rename the label column to labels because
        # the model expects the argument to be named labels
        tokenized_test_datasets = tokenized_test_datasets.rename_column(
            "label", "labels"
        )

        # Set the format of the dataset to return PyTorch tensors instead of lists
        tokenized_test_datasets.set_format("torch")

        # Load your model with the number of expected labels:
        print("Loading base model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=5
        )

        eval_args = TrainingArguments(
            output_dir=str(model_cache_for_training_round),
            evaluation_strategy="epoch",
            num_train_epochs=1,
            per_device_eval_batch_size=training_batch_size,
            use_mps_device=torch.backends.mps.is_available(),  # type: ignore
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_datasets,
            compute_metrics=compute_metrics,
        )

        trainer.evaluate()

        trainer.save_model(output_dir=str(model_cache_for_training_round))

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TESTING_COMPLETE"
        )

        # clear the cache
        test_dataset.cleanup_cache_files()

    def worker(self):
        try:
            while True:
                self.spice.hardware.check_in_http(is_available=True)

                # first check if this machine picked up a training round already
                config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
                has_step_to_train = False
                has_step_to_test = False
                has_model_to_upload = False

                if config.get("id", None) is not None:
                    step_status = config.get("status", None)
                    if step_status in ACTIVE_TRAINING_ROUND_STEP_STATUSES:
                        has_step_to_train = True
                    if step_status in ACTIVE_TESTING_STEP_STATUSES:
                        has_step_to_test = True
                    elif step_status in ACTIVE_UPLOADING_STEP_STATUSES:
                        has_model_to_upload = True

                if (
                    not has_step_to_train
                    and not has_step_to_test
                    and not has_model_to_upload
                ):
                    self._consume()
                if has_step_to_train:
                    print(f" [*] Training - Using Device: {self.device} for training")
                    self.train_model()
                if has_step_to_test:
                    print(f" [*] Testing - Using Device: {self.device} for training")
                    self.test_model()
                if has_model_to_upload:
                    print(" [*] Uploading")
                    self.upload_models()

        except KeyboardInterrupt:
            print(" [*] Stopping worker...")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            sys.exit()
