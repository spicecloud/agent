import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional

from aiohttp import client_exceptions
from datasets import load_dataset
import evaluate
from gql import gql
from gql.transport.exceptions import TransportQueryError
import numpy as np
import requests
import torch  # noqa
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (  # noqa
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from transformers.data import DefaultDataCollator

from spice_agent.utils.config import (
    HF_HUB_DIRECTORY,
    SPICE_MODEL_CACHE_FILEPATH,
    SPICE_ROUND_VERIFICATION_FILEPATH,
    SPICE_TRAINING_FILEPATH,
    copy_directory,
    create_directory,
    read_config_file,
    update_config_file,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
import transformers  # noqa

LOGGER = logging.getLogger(__name__)


MODEL_BUCKET_NAME = "spice-models"

TOKENIZER_MAX_LENGTH = 32


class ThreadedStatusDetailsCallbackDecorator:
    """
    A class decorator that enables threaded functionality for StatusDetailsCallback
    """

    def __init__(self, is_threaded: Optional[bool] = False):
        self.is_threaded = is_threaded
        self.current_thread = None

    def __call__(
        self,
        cls,
        functions_to_decorate=["on_train_begin", "on_step_end", "on_train_end"],
    ):
        if not self.is_threaded:
            return cls

        # Wrap functions with defined decorators
        # Example: on_step_end gets wrapped with on_step_end_decorator
        decorated_methods = []
        for name in functions_to_decorate:
            value = cls.__dict__[name]
            if callable(value):
                decorated_method = getattr(self, f"{name}_decorator", None)
                if decorated_method:
                    decorated_method = decorated_method(value)
                    setattr(cls, name, decorated_method)
                    decorated_methods.append(name)

        # Set wrapped class' attributes
        setattr(cls, "decorated_methods", decorated_methods)
        return cls

    def __del__(self):
        self._clean_current_thread()

    def on_train_begin_decorator(self, function):
        def wrapper(*args, **kwargs):
            try:
                if not self.current_thread:
                    self.current_thread = self._get_threaded_function(
                        function, *args, **kwargs
                    )

                    self.current_thread.start()
            except KeyboardInterrupt as exception:
                self._clean_current_thread()
                raise exception

        return wrapper

    def on_step_end_decorator(self, function):
        def wrapper(*args, **kwargs):
            try:
                # If the current thread is no longer alive, we join it with the main
                # thread and reset current thread
                if self.current_thread and not self.current_thread.is_alive():
                    self.current_thread.join()
                    del self.current_thread
                    self.current_thread = None

                # We set the current thread and start it's activity
                if not self.current_thread:
                    self.current_thread = self._get_threaded_function(
                        function, *args, **kwargs
                    )
                    self.current_thread.start()
            except KeyboardInterrupt as exception:
                self._clean_current_thread()
                raise exception

        return wrapper

    def on_train_end_decorator(self, function):
        def wrapper(*args, **kwargs):
            try:
                # Complete final on_step_end update
                if self.current_thread and self.current_thread.is_alive():
                    self.current_thread.join()
                    del self.current_thread
                    self.current_thread = None

                # We execute on_train_end on the main thread since it is the final call
                # in StatusDetailsCallback and we have to wait for it's completion.
                function(*args, **kwargs)
            except KeyboardInterrupt as exception:
                self._clean_current_thread()
                raise exception

        return wrapper

    def _get_threaded_function(self, function, *args, **kwargs) -> threading.Thread:
        def target_function():
            function(*args, **kwargs)

        thread = threading.Thread(target=target_function)
        return thread

    def _clean_current_thread(self):
        if self.current_thread:
            if self.current_thread.is_alive():
                self.current_thread.join()
            del self.current_thread


@ThreadedStatusDetailsCallbackDecorator(is_threaded=True)
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
                    status_details={"progress": 100},
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

    def _get_training_round_step(self, training_round_step_id):
        query = gql(
            """
            query trainingRoundStep($id: GlobalID!) {
                trainingRoundStep(id: $id) {
                    id
                    status
                }
            }
            """  # noqa
        )
        variables = {
            "id": training_round_step_id,
        }

        return self.spice.session.execute(query, variable_values=variables)

    def _get_training_round(self, training_round_id):
        query = gql(
            """
            query trainingRound($id: GlobalID!) {
                trainingRound(id: $id) {
                    id
                    status
                }
            }
            """  # noqa
        )
        variables = {
            "id": training_round_id,
        }

        return self.spice.session.execute(query, variable_values=variables)

    def _update_training_round(
        self,
        training_round_id: str,
        status: str,
        status_details: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ):
        mutation = gql(
            """
            mutation updateTrainingRound($input: UpdateTrainingRoundInput!) {
                updateTrainingRound(input: $input) {
                    ... on TrainingRound {
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
            }
            """  # noqa
        )
        input: Dict[str, str | float] = {"trainingRoundId": training_round_id}
        if status is not None:
            input["status"] = status
        if status_details:
            input["statusDetails"] = json.dumps(status_details)
        if metrics:
            input["metrics"] = json.dumps(metrics)

        variables = {"input": input}
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
            new_config=result["updateTrainingRound"],
        )
        return result

    def _update_training_round_step(
        self,
        training_round_step_id: str,
        status: str,
        status_details: Optional[dict] = None,
        file_id: Optional[str] = None,
        metrics: Optional[dict] = None,
    ):
        mutation = gql(
            """
            mutation updateTrainingRoundStep($input: UpdateTrainingRoundStepInput!) {
                updateTrainingRoundStep(input: $input) {
                    ... on TrainingRoundStep {
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
                                baseModelRepoId
                                baseModelRepoRevision
                            }
                        }
                        statusDetails
                    }
                }
            }
        """  # noqa
        )
        input: Dict[str, str | float] = {"trainingRoundStepId": training_round_step_id}
        if status is not None:
            input["status"] = status
        if status_details:
            input["statusDetails"] = json.dumps(status_details)
        if file_id is not None:
            input["fileId"] = file_id
        if metrics:
            input["metrics"] = json.dumps(metrics)

        variables = {"input": input}

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
            new_config=result["updateTrainingRoundStep"],
        )
        return result

    def _get_agent_round_presigned_urls(self, training_round_id: str):
        # todo: remove training_round_number
        query = gql(
            """
            query getAgentRoundPresignedUrls($trainingRoundId: GlobalID!) {
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
        if not self.check_training_round_step_exists(training_round_step_id):
            return None

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
                self.spice.uploader.upload_file_direct(
                    bucket_name=MODEL_BUCKET_NAME,
                    key=bucket_key,
                    filepath=file,
                    file_id=file_id,
                )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="COMPLETE"
        )

    def _load_dataset(self, config, task):
        # TODO: create one central training config file
        if task == "train" or task == "test":
            hf_dataset_repo_id = config["hfDatasetRepoId"]
            hf_dataset_repo_revision = config["hfDatasetRepoRevision"]
        elif task == "verify":
            hf_dataset_repo_id = config["trainingJob"]["baseDatasetRepoId"]
            hf_dataset_repo_revision = config["trainingJob"]["baseDatasetRepoRevision"]
        else:
            error_message = f"_load_dataset with task={task} does not exist!"
            LOGGER.error(error_message)
            raise ValueError(error_message)

        if task == "train":
            dataset_starting_row = config["datasetStartingRow"]
            dataset_ending_row = config["datasetEndingRow"]
            split = f"train[{dataset_starting_row}:{dataset_ending_row}]"
        else:
            split = "test"  # both test_model and verify_model uses test split

        return load_dataset(
            path=hf_dataset_repo_id, revision=hf_dataset_repo_revision, split=split
        )

    def _tokenize_dataset(self, dataset, config, task):
        # TODO: create one central training config file

        # get tokenizer
        print("Loading tokenizer...")
        if task == "train" or task == "test":
            hf_model_repo_id = config["hfModelRepoId"]
            hf_model_repo_revision = config["hfModelRepoRevision"]

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=hf_model_repo_id,
                revision=hf_model_repo_revision,
            )
        elif task == "verify":
            training_round_number = config["roundNumber"]
            hf_model_repo_id = config["trainingJob"]["hfModelRepoId"]
            base_model_repo_id = config["trainingJob"]["baseModelRepoId"]
            base_model_repo_revision = config["trainingJob"]["baseModelRepoRevision"]

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=hf_model_repo_id
                if hf_model_repo_id and training_round_number > 1
                else base_model_repo_id,
                revision="main"  # uses main tokenizer
                if hf_model_repo_id and training_round_number > 1
                else base_model_repo_revision,
            )
        else:
            error_message = f"_tokenize_dataset with task={task} does not exist!"
            LOGGER.error(error_message)
            raise ValueError(error_message)

        # create a tokenize function that will tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=TOKENIZER_MAX_LENGTH,
            )

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Remove the text column because the model does not accept raw text as an input
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])

        # Rename the label column to labels because
        # the model expects the argument to be named labels
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

        # Set the format of the dataset to return PyTorch tensors instead of lists
        tokenized_dataset.set_format("torch")

        return tokenizer, tokenized_dataset

    def _process_dataset(self, dataset, config, task):
        # get image processor
        LOGGER.info("Loading image processor...")
        if task in ["train", "test"]:
            hf_model_repo_id = config["hfModelRepoId"]
            hf_model_repo_revision = config["hfModelRepoRevision"]

            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=hf_model_repo_id,
                revision=hf_model_repo_revision,
                trust_remote_code=True,
            )
        elif task == "verify":
            training_round_number = config["roundNumber"]
            hf_model_repo_id = config["trainingJob"]["hfModelRepoId"]
            base_model_repo_id = config["trainingJob"]["baseModelRepoId"]
            base_model_repo_revision = config["trainingJob"]["baseModelRepoRevision"]

            pretrained_model_name_or_path = base_model_repo_id
            revision = "main"
            if hf_model_repo_id and training_round_number > 1:
                pretrained_model_name_or_path = base_model_repo_id
                revision = base_model_repo_revision

            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                revision=revision,
                trust_remote_code=True,
            )
        else:
            error_message = f"_process_dataset with task={task} does not exist!"
            LOGGER.error(error_message)
            raise ValueError(error_message)

        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )

        size = image_processor.size.get("shortest_edge", None)
        if (
            not size
            and image_processor.size.get("height", None)
            and image_processor.size.get("width", None)
        ):
            size = (image_processor.size["height"], image_processor.size["width"])
        else:
            message = "size was not set. image_processor did not have keys \
                shortest_edge, or height and width."
            LOGGER.error(message)
            raise Exception(message)

        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        def transforms(examples):
            examples["tensor"] = [_transforms(img) for img in examples["image"]]
            del examples["image"]
            return examples

        print("Processing dataset...")
        dataset = dataset.with_transform(transforms)
        return image_processor, dataset

    def _load_model(self, config, task):
        # Setup model cache for training/testing
        model_cache_for_training_round_step = None
        base_model_repo_id = None
        if task == "train" or task == "test":
            training_round_id = config["trainingRound"]["id"]
            training_round_step_id = config["id"]
            base_model_repo_id = config["trainingRound"]["trainingJob"][
                "baseModelRepoId"
            ]
            training_round_directory = (
                f"{training_round_id}/steps/{training_round_step_id}"
            )
            model_cache_for_training_round_step = SPICE_MODEL_CACHE_FILEPATH.joinpath(
                training_round_directory
            )

        # Select pretrained model
        if task == "train":
            pretrained_model_name_or_path = config["hfModelRepoId"]
        elif task == "test":
            if not model_cache_for_training_round_step:
                error_message = f"_get_trainer with task={task} requires model_cache_for_training_round_step!"  # noqa
                LOGGER.error(error_message)
                raise ValueError(error_message)
            pretrained_model_name_or_path = model_cache_for_training_round_step
        elif task == "verify":
            training_round_id = config["id"]
            base_model_repo_id = config["trainingJob"]["baseModelRepoId"]
            spice_model_round_cache = SPICE_MODEL_CACHE_FILEPATH.joinpath(
                f"{training_round_id}/"
            )
            pretrained_model_name_or_path = spice_model_round_cache
        else:
            error_message = f"_load_model with task={task} does not exist!"
            LOGGER.error(error_message)
            raise ValueError(error_message)

        # Load your model with the number of expected labels:
        LOGGER.info(f"Loading {task} model...")
        if base_model_repo_id == "spicecloud/spice-cnn-base":
            model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path,
                num_labels=10,
                trust_remote_code=True,
            )
        elif base_model_repo_id == "bert-base-uncased":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=5
            )
        else:
            error_message = (
                f"_load_model has unknown base model: {base_model_repo_id} type"
            )
            LOGGER.error(error_message)
            raise ValueError(error_message)

        return model

    def _get_training_arguments(self, output_dir, config, task):
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            use_mps_device=torch.backends.mps.is_available(),  # type: ignore
        )

        if task == "train":
            # set to "no", set to "epoch" for eval + train
            training_arguments.evaluation_strategy = "no"
            training_arguments.per_device_train_batch_size = config["trainingBatchSize"]
        elif task == "test":
            training_arguments.evaluation_strategy = "epoch"
            training_arguments.per_device_eval_batch_size = config["trainingBatchSize"]
        elif task == "verify":
            training_arguments.evaluation_strategy = "epoch"
            training_arguments.per_device_eval_batch_size = 32
            training_arguments.save_strategy = "no"
        else:
            error_message = f"_get_training_arguments with task={task} does not exist!"
            LOGGER.error(error_message)
            raise ValueError(error_message)

        return training_arguments

    def train_model(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        if not self.check_training_round_step_exists(training_round_step_id):
            return None

        training_round_id = config["trainingRound"]["id"]
        base_model_repo_id = config["trainingRound"]["trainingJob"]["baseModelRepoId"]

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
        train_dataset = self._load_dataset(config, "train")

        # tokenize dataset
        tokenizer = None
        if base_model_repo_id == "spicecloud/spice-cnn-base":
            tokenizer, tokenized_dataset = self._process_dataset(
                train_dataset, config, "train"
            )
        elif base_model_repo_id == "bert-base-uncased":
            tokenizer, tokenized_dataset = self._tokenize_dataset(
                train_dataset, config, "train"
            )
        else:
            error_message = (
                f"train_model has unknown base model: {base_model_repo_id} type"
            )
            LOGGER.error(error_message)
            raise ValueError(error_message)

        # load your model with the number of expected labels:
        model = self._load_model(config, "train")

        training_args = self._get_training_arguments(
            model_cache_for_training_round, config, "train"
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

        if base_model_repo_id == "spicecloud/spice-cnn-base":
            training_args.remove_unused_columns = False

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
            data_collator=DefaultDataCollator(),
        )

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TRAINING"
        )

        trainer.train()

        trainer.save_model(output_dir=str(model_cache_for_training_round))

        self._update_training_round_step(
            training_round_step_id=training_round_step_id, status="TRAINING_COMPLETE"
        )

        # clear the cache
        train_dataset.cleanup_cache_files()

    def test_model(self):
        config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
        training_round_step_id = config["id"]
        if not self.check_training_round_step_exists(training_round_step_id):
            return None

        training_round_id = config["trainingRound"]["id"]
        base_model_repo_id = config["trainingRound"]["trainingJob"]["baseModelRepoId"]

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
        test_dataset = self._load_dataset(config, "test")

        # tokenize dataset
        tokenizer = None
        if base_model_repo_id == "spicecloud/spice-cnn-base":
            tokenizer, tokenized_dataset = self._process_dataset(
                test_dataset, config, "test"
            )
        elif base_model_repo_id == "bert-base-uncased":
            tokenizer, tokenized_dataset = self._tokenize_dataset(
                test_dataset, config, "test"
            )
        else:
            error_message = (
                f"test_model has unknown base model: {base_model_repo_id} type"
            )
            LOGGER.error(error_message)
            raise ValueError(error_message)

        # load your model with the number of expected labels:
        model = self._load_model(config, "test")

        eval_args = self._get_training_arguments(
            model_cache_for_training_round, config, "test"
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

        if base_model_repo_id == "spicecloud/spice-cnn-base":
            eval_args.remove_unused_columns = False

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
            data_collator=DefaultDataCollator(),
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
            metrics=metrics,
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

    def check_training_round_exists(self, training_round_id) -> bool:
        try:
            result = self._get_training_round(training_round_id=training_round_id)
            if (
                result["trainingRound"]["status"]
                not in self.spice.worker.ACTIVE_STATUSES
            ):
                LOGGER.info(
                    f"Training round {training_round_id} is not active. Deleting {SPICE_ROUND_VERIFICATION_FILEPATH} config."  # noqa
                )
                SPICE_ROUND_VERIFICATION_FILEPATH.unlink(missing_ok=True)
                return False
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if (
                        error.get("message", "")
                        == "TrainingRound matching query does not exist."
                    ):
                        LOGGER.error(
                            f""" [*] Training Round ID: {training_round_id} not found. Deleting {SPICE_ROUND_VERIFICATION_FILEPATH} config."""  # noqa
                        )
                        SPICE_ROUND_VERIFICATION_FILEPATH.unlink(missing_ok=True)
                        return False
                raise exception
            else:
                raise exception
        return True

    def check_training_round_step_exists(self, training_round_step_id) -> bool:
        try:
            result = self._get_training_round_step(
                training_round_step_id=training_round_step_id
            )
            if (
                result["trainingRoundStep"]["status"]
                not in self.spice.worker.ACTIVE_STATUSES
            ):
                LOGGER.info(
                    f"Training round step {training_round_step_id} is not active. Deleting {SPICE_TRAINING_FILEPATH} config."  # noqa
                )
                SPICE_TRAINING_FILEPATH.unlink(missing_ok=True)
                return False
        except TransportQueryError as exception:
            if exception.errors:
                for error in exception.errors:
                    if (
                        error.get("message", "")
                        == "TrainingRoundStep matching query does not exist."
                    ):
                        LOGGER.error(
                            f""" [*] Training Round Step ID: {training_round_step_id} not found. Deleting {SPICE_TRAINING_FILEPATH}."""  # noqa
                        )
                        SPICE_TRAINING_FILEPATH.unlink(missing_ok=True)
                        return False
                raise exception
            else:
                raise exception
        return True

    def verify_model(self):
        config = read_config_file(filepath=SPICE_ROUND_VERIFICATION_FILEPATH)
        training_round_id = config["id"]

        if not self.check_training_round_exists(training_round_id=training_round_id):
            return None

        training_round_number = config["roundNumber"]
        training_job_id = config["trainingJob"]["id"]
        base_model_repo_id = config["trainingJob"]["baseModelRepoId"]

        # create the folder for the verification round
        # where the step model will be saved
        verification_round_directory = f"{training_round_id}/verification"
        verification_cache_for_training_round = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            verification_round_directory
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
        test_dataset = self._load_dataset(config, "verify")

        # tokenize dataset
        tokenizer = None
        if base_model_repo_id == "spicecloud/spice-cnn-base":
            tokenizer, tokenized_dataset = self._process_dataset(
                test_dataset, config, "verify"
            )
        elif base_model_repo_id == "bert-base-uncased":
            tokenizer, tokenized_dataset = self._tokenize_dataset(
                test_dataset, config, "verify"
            )
        else:
            error_message = (
                f"test_model has unknown base model: {base_model_repo_id} type"
            )
            LOGGER.error(error_message)
            raise ValueError(error_message)

        # load your model with the number of expected labels:
        model = self._load_model(config, "verify")

        eval_args = self._get_training_arguments(
            verification_cache_for_training_round, config, "verify"
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

        if base_model_repo_id == "spicecloud/spice-cnn-base":
            eval_args.remove_unused_columns = False

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[status_details_callback],
            data_collator=DefaultDataCollator(),
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

                self.spice.uploader.upload_file_direct(
                    bucket_name=MODEL_BUCKET_NAME,
                    key=bucket_key,
                    filepath=file,
                    file_id=file_id,
                )

        print("Complete!")

        self._update_training_round(
            training_round_id=training_round_id,
            status="VERIFYING_COMPLETE",
            metrics=metrics,
        )

        # clear the cache
        test_dataset.cleanup_cache_files()
