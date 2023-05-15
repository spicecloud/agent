import json
import logging
import os
import platform
import ssl
import sys

from datasets import load_dataset
from gql import gql
import pika
from pika.exceptions import AMQPConnectionError, ConnectionClosedByBroker
from retry import retry
from torch.mps import empty_cache
from tqdm.auto import tqdm
from transformers import (  # noqa
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from spice.utils.config import (
    SPICE_TRAINING_FILEPATH,
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

    def _update_training_round_step(self, training_round_step_id: str, status: str):
        mutation = gql(
            """
            mutation updateTrainingRoundStepFromHardware($trainingRoundStepId: String!, $status: String!) {
                updateTrainingRoundStepFromHardware(trainingRoundStepId: $trainingRoundStepId, status: $status) {
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
                }
            }
        """  # noqa
        )
        variables = {"trainingRoundStepId": training_round_step_id, "status": status}
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

    def train(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config.get("id")
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="CLAIMED"
        )
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

        base_model = config.get("baseModel")
        base_model_revision = config.get("baseModelRevision")
        dataset_repo_id = config.get("baseDatasetRepoId")
        dataset_repo_revision = config.get("baseDatasetRepoRevision")
        dataset_starting_row = config.get("datasetStartingRow")
        dataset_ending_row = config.get("datasetEndingRow")
        training_epochs = config.get("trainingEpochs")
        training_batch_size = config.get("trainingBatchSize")
        testing_batch_size = 32  # need to put this in config

        # get dataset
        print("Loading dataset...")
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="DOWNLOADING_DATASET"
        )

        split_string = f"train[{dataset_starting_row}:{dataset_ending_row}]"
        dataset = load_dataset(
            path=dataset_repo_id, revision=dataset_repo_revision, split=split_string
        )

        test_split_string = "test[:]"
        test_dataset = load_dataset(
            path=dataset_repo_id,
            revision=dataset_repo_revision,
            split=test_split_string,
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
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

        # Remove the text column because the model does not accept raw text as an input
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_test_datasets = tokenized_test_datasets.remove_columns(["text"])
        # Rename the label column to labels because
        # the model expects the argument to be named labels
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_test_datasets = tokenized_test_datasets.rename_column(
            "label", "labels"
        )
        # Set the format of the dataset to return PyTorch tensors instead of lists
        tokenized_datasets.set_format("torch")
        tokenized_test_datasets.set_format("torch")

        # Create a DataLoader for your training and test datasets
        # so you can iterate over batches of data
        print("Creating dataloaders...")
        train_dataloader = DataLoader(
            tokenized_datasets, shuffle=True, batch_size=training_batch_size
        )
        test_dataloader = DataLoader(
            # Need to add batch size to training config
            tokenized_test_datasets,
            shuffle=False,
            batch_size=testing_batch_size,
        )

        # Load your model with the number of expected labels:
        print("Loading base model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=5
        )
        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="DOWNLOADING_DATASET"
        )

        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = training_epochs * len(train_dataloader)
        num_testing_steps = len(test_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        model.to(self.device)

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TRAINING"
        )
        progress_bar = tqdm(range(num_training_steps), desc="Training Progress")

        model.train()
        for epoch in range(training_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        progress_bar.close()

        # Need to add
        # self._update_training_round_step(
        #     training_round_step_id=training_round_step_id, status="TESTING"
        # )
        progress_bar_testing = tqdm(range(num_testing_steps), desc="Testing Progress")

        model.eval()
        test_correct = 0
        test_wrong = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = model(**batch)
                logits = predictions["logits"]

                predictions_class = torch.argmax(logits, dim=logits.dim() - 1)
                correct_predictions = torch.eq(predictions_class, batch["labels"]).sum()
                test_correct += correct_predictions
                test_wrong += testing_batch_size - correct_predictions

                progress_bar_testing.update(1)

        progress_bar_testing.close()

        test_accuracy = (test_correct * 1.0) / (test_correct + test_wrong)

        training_round_step_dict = {
            "model_state_dict": model.state_dict(),
            "test_accuracy": test_accuracy,
        }

        torch.save(training_round_step_dict, f"split-{training_round_step_id}.pt")

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="COMPLETE"
        )
        print("Complete!")

    def worker(self):
        print(f" [*] Using Device: {self.device} for training.")
        try:
            while True:
                self.spice.hardware.check_in_http(is_available=True)

                # first check if this machine picked up a training round already
                config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
                has_step_to_complete = (
                    config.get("id", None) is not None
                    and config.get("status", None)
                    in ACTIVE_TRAINING_ROUND_STEP_STATUSES
                )
                if not has_step_to_complete:
                    self._consume()
                self.train()

        except KeyboardInterrupt:
            print(" [*] Stopping worker...")
            if self.channel:
                self.channel.stop_consuming()
                self.channel.close()
                self.channel = None
            self.spice.hardware.check_in_http(is_available=False)
            sys.exit()
