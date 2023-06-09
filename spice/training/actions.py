import hashlib
import json
import logging
import os
from typing import Dict, Optional

import evaluate
import numpy as np
import requests
import torch
import transformers
from aiohttp import client_exceptions
from datasets import load_dataset
from gql import gql
from gql.transport.exceptions import TransportQueryError
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from spice.utils.config import (
    HF_HUB_DIRECTORY,
    SPICE_MODEL_CACHE_FILEPATH,
    SPICE_ROUND_VERIFICATION_FILEPATH,
    SPICE_TRAINING_FILEPATH,
    copy_directory,
    create_directory,
    read_config_file,
    update_config_file,
)

LOGGER = logging.getLogger(__name__)
MODEL_BUCKET_NAME = "spice-models"
TOKENIZER_MAX_LENGTH = 32

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class StatusDetailsCallback(transformers.TrainerCallback):
    """
    A [`TrainerCallback`] that sends the progress of training or evaluation
    to the spice backend.
    """

    def __init__(
        self,
        training,
        status: str,
        training_round_step_id: Optional[str] = None,
        training_round_id: Optional[str] = None,
    ):
        self.training = training
        self.status = status
        self.training_round_step_id = training_round_step_id
        self.training_round_id = training_round_id

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.training_round_step_id:
                self.training._update_training_round_step(
                    training_round_step_id=self.training_round_step_id,
                    status=self.status,
                    status_details={"progress": 0},
                )
            if self.training_round_id:
                self.training._update_training_round(
                    training_round_id=self.training_round_id,
                    status=self.status,
                    status_details={"progress": 0},
                )
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            progress = round((state.global_step / state.max_steps) * 100)
            if self.training_round_step_id:
                self.training._update_training_round_step(
                    training_round_step_id=self.training_round_step_id,
                    status=self.status,
                    status_details={"progress": progress},
                )
            if self.training_round_id:
                self.training._update_training_round(
                    training_round_id=self.training_round_id,
                    status=self.status,
                    status_details={"progress": progress},
                )
            self.current_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.training_round_step_id:
                self.training._update_training_round_step(
                    training_round_step_id=self.training_round_step_id,
                    status=self.status,
                    status_details={"progress": 100},
                )
            if self.training_round_id:
                self.training._update_training_round(
                    training_round_id=self.training_round_id,
                    status=self.status,
                    status_details={"progress": 0},
                )


class Training:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.spice.get_device()

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

    def _update_training_round(
        self,
        training_round_id: str,
        status: str,
        status_details: Optional[dict] = None,
        round_accuracy: Optional[float] = None,
        round_loss: Optional[float] = None,
    ):
        mutation = gql(
            """
            mutation updateTrainingRoundFromHardware($trainingRoundId: String!, $status: String!, $statusDetails: JSON, $roundAccuracy: Float, $roundLoss: Float) {
                updateTrainingRoundFromHardware(trainingRoundId: $trainingRoundId, status: $status, statusDetails: $statusDetails, roundAccuracy: $roundAccuracy, roundLoss: $roundLoss) {
                    id
                    status
                    statusDetails
                    roundNumber
                    trainingJob {
                        id
                        baseModelRepoId
                        baseModelRepoRevision
                        baseDatasetRepoId
                        baseDatasetRepoRevision
                        hfModelRepoId
                    }
                }
            }
            """  # noqa
        )
        variables: Dict[str, str | float] = {"trainingRoundId": training_round_id}
        if status is not None:
            variables["status"] = status
        if status_details:
            variables["statusDetails"] = json.dumps(status_details)
        if round_accuracy is not None:
            variables["roundAccuracy"] = round_accuracy
        if round_loss is not None:
            variables["roundLoss"] = round_loss

        try:
            result = self.spice.session.execute(mutation, variable_values=variables)
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if error.get("message", None) == "Round not found.":
                        LOGGER.error("Round not found.")
                        return None
            else:
                raise exception

        update_config_file(
            filepath=SPICE_ROUND_VERIFICATION_FILEPATH,
            new_config=result["updateTrainingRoundFromHardware"],
        )
        return result

    def _update_training_round_step(
        self,
        training_round_step_id: str,
        status: str,
        status_details: Optional[dict] = None,
        file_id: Optional[str] = None,
        step_accuracy: Optional[float] = None,
        step_loss: Optional[float] = None,
    ):
        mutation = gql(
            """
            mutation updateTrainingRoundStepFromHardware($trainingRoundStepId: String!, $status: String!, $fileId: String, $statusDetails: JSON, $stepAccuracy: Float, $stepLoss: Float) {
                updateTrainingRoundStepFromHardware(trainingRoundStepId: $trainingRoundStepId, status: $status, fileId: $fileId, statusDetails: $statusDetails, stepAccuracy: $stepAccuracy, stepLoss: $stepLoss) {
                    id
                    status
                    hfModelRepoId
                    hfModelRepoRevision
                    hfDatasetRepoId
                    hfDatasetRepoRevision
                    datasetStartingRow
                    datasetEndingRow
                    trainingEpochs
                    trainingBatchSize
                    trainingRound {
                        id
                        roundNumber
                        trainingJob {
                            id
                        }
                    }
                    statusDetails
                }
            }
        """  # noqa
        )
        variables: Dict[str, str | float] = {
            "trainingRoundStepId": training_round_step_id
        }
        if status is not None:
            variables["status"] = status
        if status_details:
            variables["statusDetails"] = json.dumps(status_details)
        if file_id is not None:
            variables["fileId"] = file_id
        if step_accuracy is not None:
            variables["stepAccuracy"] = step_accuracy
        if step_loss is not None:
            variables["stepLoss"] = step_loss

        try:
            result = self.spice.session.execute(mutation, variable_values=variables)
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if error.get("message", None) == "Round step not found.":
                        LOGGER.error("Round step not found.")
                        return None
            else:
                raise exception
        except client_exceptions.ClientOSError:
            # the backend can be down or deploying this step, do not break the app
            if status in self.spice.worker.ACTIVE_STATUSES:
                return None

        update_config_file(
            filepath=SPICE_TRAINING_FILEPATH,
            new_config=result["updateTrainingRoundStepFromHardware"],
        )
        return result

    def _get_agent_round_presigned_urls(self, training_round_id: str):
        # todo: remove training_round_number
        query = gql(
            """
            query getAgentRoundPresignedUrls($trainingRoundId: String!) {
                getAgentRoundPresignedUrls(trainingRoundId: $trainingRoundId) {
                    roundModel
                    config
                }
            }
            """  # noqa
        )
        variables = {
            "trainingRoundId": training_round_id,
        }

        return self.spice.session.execute(query, variable_values=variables)

    def upload_models(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        hf_model_repo_id = config["hfModelRepoId"]
        training_round_id = config["trainingRound"]["id"]
        training_round_number = config["trainingRound"]["roundNumber"]
        training_job_id = config["trainingRound"]["trainingJob"]["id"]
        training_round_directory = f"{training_round_id}/steps/{training_round_step_id}"

        # round_id_step_id will be in form: [uuid]/steps/[uuid]
        # round_id_step_id is used for finding the step model in the cache
        # and as the key in S3
        bucket_dir = f"{training_job_id}/rounds/{training_round_number}/steps/{training_round_step_id}/"  # noqa

        # model_cache_for_training_round contains all the step models
        # for this particular training round
        model_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            training_round_directory
        )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="UPLOADING",
        )

        # hf_model_directory could refer to different snapshots of the model
        hf_model_repo_id_formatted = hf_model_repo_id.replace("/", "--")
        snapshot_directory = f"models--{hf_model_repo_id_formatted}/snapshots"
        hf_model_directory = HF_HUB_DIRECTORY.joinpath(snapshot_directory)
        recent_snapshot_directory = max(
            hf_model_directory.iterdir(), key=lambda d: d.stat().st_mtime
        )
        recent_hf_model_directory = hf_model_directory.joinpath(
            recent_snapshot_directory
        )
        copy_directory(recent_hf_model_directory, model_cache_for_training_round)

        for file in model_cache_for_training_round.iterdir():
            if file.is_file():
                bucket_key = bucket_dir + file.name

                # create the "file" and attach it to the training round step
                file_checksum = hashlib.md5(file.read_bytes()).hexdigest()
                file_id = self.spice.uploader._create_file(
                    file_name=file.name,
                    file_size=file.stat().st_size,
                    file_checksum=file_checksum,
                    location=f"s3://{bucket_key}",
                )
                self.spice.uploader.upload_file(
                    bucket_name=MODEL_BUCKET_NAME,
                    key=bucket_key,
                    filepath=file,
                    file_id=file_id,
                )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="COMPLETE"
        )

    def train_model(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]

        if config.get("status") == "READY_FOR_PICKUP":
            self._update_training_round_step(
                training_round_step_id=training_round_step_id, status="CLAIMED"
            )
            config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

        hf_model_repo_id = config["hfModelRepoId"]
        hf_model_repo_revision = config["hfModelRepoRevision"]
        hf_dataset_repo_id = config["hfDatasetRepoId"]
        hf_dataset_repo_revision = config["hfDatasetRepoRevision"]
        dataset_starting_row = config["datasetStartingRow"]
        dataset_ending_row = config["datasetEndingRow"]
        training_batch_size = config["trainingBatchSize"]
        training_round_id = config["trainingRound"]["id"]

        # create the folder for the new training round
        # where the step model will be saved
        training_round_directory = f"{training_round_id}/steps/{training_round_step_id}"
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
            path=hf_dataset_repo_id,
            revision=hf_dataset_repo_revision,
            split=split_string,
        )

        # get tokenizer from base model bert-base-cased
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=hf_model_repo_id,
            revision=hf_model_repo_revision,
        )

        # create a tokenize function that will tokenize the dataset
        # bert-base-uncased uses a subword tokenizer so the maximum length corresponds
        # to 512 subword tokens
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                # max_length=TOKENIZER_MAX_LENGTH,
            )

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
            hf_model_repo_id, num_labels=5
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

        status_details_callback = StatusDetailsCallback(
            training=self,
            status="TRAINING",
            training_round_step_id=training_round_step_id,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
        )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TRAINING"
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

        if config.get("status") == "READY_FOR_PICKUP":
            self._update_training_round_step(
                training_round_step_id=training_round_step_id, status="CLAIMED"
            )
            config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)

        hf_model_repo_id = config["hfModelRepoId"]
        hf_model_repo_revision = config["hfModelRepoRevision"]
        hf_dataset_repo_id = config["hfDatasetRepoId"]
        hf_dataset_repo_revision = config["hfDatasetRepoRevision"]
        training_batch_size = config["trainingBatchSize"]
        training_round_id = config["trainingRound"]["id"]

        # create the folder for the new training round
        # where the step model will be saved
        training_round_directory = f"{training_round_id}/steps/{training_round_step_id}"
        model_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            training_round_directory
        )

        # get dataset
        print("Loading test dataset...")
        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="DOWNLOADING_TESTING_DATASET",
        )

        test_dataset = load_dataset(
            path=hf_dataset_repo_id, revision=hf_dataset_repo_revision, split="test"
        )

        # get tokenizer from base model bert-base-cased
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=hf_model_repo_id,
            revision=hf_model_repo_revision,
        )

        # create a tokenize function that will tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                # max_length=TOKENIZER_MAX_LENGTH,
            )

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
            hf_model_repo_id, num_labels=5
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

        status_details_callback = StatusDetailsCallback(
            training=self,
            training_round_step_id=training_round_step_id,
            status="TESTING",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_datasets,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
        )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="TESTING",
        )

        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics, combined=False)

        self._update_training_round_step(
            training_round_step_id=training_round_step_id,
            status="TESTING_COMPLETE",
            step_accuracy=metrics["eval_accuracy"],
            step_loss=metrics["eval_loss"],
        )

        # clear the cache
        test_dataset.cleanup_cache_files()

    def _download_verify_model_file(self, url, destination_url, is_json=False):
        response = requests.get(url)
        response.raise_for_status()

        # Implement checksum if file is dated
        if is_json:
            with open(destination_url, "w") as file:  # download json file
                json_string = response.content.decode("utf-8")
                json_data = json.loads(json_string)
                # TEMPORARY FORMATTING FIX THIS SHOULD BE DONE IN UPLOADER
                json_formatted = json.dumps(json_data, indent=2) + "\n"
                file.write(json_formatted)
        else:
            with open(destination_url, "wb") as file:  # download binary file
                file.write(response.content)

    def verify_model(self):
        config = read_config_file(filepath=SPICE_ROUND_VERIFICATION_FILEPATH)
        training_round_id = config["id"]

        if config.get("status") == "READY_FOR_PICKUP":
            self._update_training_round(
                training_round_id=training_round_id, status="CLAIMED"
            )
            config = read_config_file(filepath=SPICE_ROUND_VERIFICATION_FILEPATH)

        training_round_number = config["roundNumber"]
        training_job_id = config["trainingJob"]["id"]
        hf_model_repo_id = config["trainingJob"]["hfModelRepoId"]
        base_model_repo_id = config["trainingJob"]["baseModelRepoId"]
        base_model_repo_revision = config["trainingJob"]["baseModelRepoRevision"]
        dataset_repo_id = config["trainingJob"]["baseDatasetRepoId"]
        dataset_repo_revision = config["trainingJob"]["baseDatasetRepoRevision"]
        verification_batch_size = 32

        # create the folder for the verification round
        # where the step model will be saved
        verification_round_directory = f"{training_round_id}/verification"
        verification_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            verification_round_directory
        )

        spice_model_round_cache = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            f"{training_round_id}/"
        )

        # get the presigned urls for a round's model + configs
        result = self._get_agent_round_presigned_urls(
            training_round_id=training_round_id,
        )

        agent_round_presigned_round_model_url = result["getAgentRoundPresignedUrls"][
            "roundModel"
        ]
        agent_presigned_config_url = result["getAgentRoundPresignedUrls"]["config"]

        destination_round_file_url = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            f"{training_round_id}/pytorch_model.bin"
        )

        destination_config_url = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            f"{training_round_id}/config.json"
        )

        print("Downloading round model...")
        self._update_training_round(
            training_round_id=training_round_id, status="DOWNLOADING_ROUND_MODEL"
        )
        self._download_verify_model_file(
            agent_round_presigned_round_model_url, destination_round_file_url
        )

        self._download_verify_model_file(
            agent_presigned_config_url, destination_config_url, is_json=True
        )

        # download validation dataset
        print("Loading verification dataset...")
        self._update_training_round(
            training_round_id=training_round_id,
            status="DOWNLOADING_VERIFICATION_DATASET",
        )

        # We need to actually create validation files here
        test_dataset = load_dataset(
            path=dataset_repo_id, revision=dataset_repo_revision, split="test"
        )

        # get tokenizer from base model bert-base-cased
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=hf_model_repo_id
            if hf_model_repo_id and training_round_number > 1
            else base_model_repo_id,
            revision="main"
            if hf_model_repo_id and training_round_number > 1
            else base_model_repo_revision,
        )

        # create a tokenize function that will tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                # max_length=TOKENIZER_MAX_LENGTH,
            )

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
            spice_model_round_cache, num_labels=5
        )

        eval_args = TrainingArguments(
            output_dir=str(verification_cache_for_training_round),
            evaluation_strategy="epoch",
            num_train_epochs=1,
            per_device_eval_batch_size=verification_batch_size,
            use_mps_device=torch.backends.mps.is_available(),  # type: ignore
            save_strategy="no",
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        status_details_callback = StatusDetailsCallback(
            training=self,
            training_round_id=training_round_id,
            status="VERIFYING",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_datasets,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
        )

        self._update_training_round(
            training_round_id=training_round_id,
            status="VERIFYING",
        )
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics, combined=False)

        # Upload evaluation results
        bucket_dir = f"{training_job_id}/rounds/{training_round_number}/merged/"
        for file in verification_cache_for_training_round.iterdir():
            if file.is_file():
                bucket_key = bucket_dir + file.name

                # create the "file" and attach it to the training round step
                file_checksum = hashlib.md5(file.read_bytes()).hexdigest()
                file_id = self.spice.uploader._create_file(
                    file_name=file.name,
                    file_size=file.stat().st_size,
                    file_checksum=file_checksum,
                    location=f"s3://{bucket_key}",
                )

                self.spice.uploader.upload_file(
                    bucket_name=MODEL_BUCKET_NAME,
                    key=bucket_key,
                    filepath=file,
                    file_id=file_id,
                )

        print("Complete!")

        self._update_training_round(
            training_round_id=training_round_id,
            status="VERIFYING_COMPLETE",
            round_accuracy=metrics["eval_accuracy"],
            round_loss=metrics["eval_loss"],
        )

        # clear the cache
        test_dataset.cleanup_cache_files()
