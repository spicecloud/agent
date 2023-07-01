import logging

from datasets import load_dataset
import evaluate
import numpy as np
from torchvision.transforms import ColorJitter, Compose, RandomResizedCrop
from tqdm.auto import tqdm

from spice_agent.client import Spice

# needs to be imported before transformers to set PYTORCH_ENABLE_MPS_FALLBACK=1


LOGGER = logging.getLogger(__name__)


spice = Spice()

import torch  # noqa
from torch.optim import AdamW  # noqa
from torch.utils.data import DataLoader  # noqa
from transformers import (  # noqa
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
    pipeline,
)

device = spice.inference.device


def example_audio_pipeline():
    # this is model="facebook/wav2vec2-large-960h-lv60-self" under the hood.
    # generator = pipeline(task="automatic-speech-recognition", device=device)
    generator = pipeline(
        task="automatic-speech-recognition", return_timestamps="word", device=device
    )
    # generator = pipeline(
    #     model="facebook/wav2vec2-large-960h-lv60-self",
    #     return_timestamps="word",
    #     device=device,
    # )
    result = generator(
        [
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
            # "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        ]
    )
    print(result)


def example_gpt2_iterator():
    def data():
        for i in range(1000):
            yield f"My example {i}"

    pipe = pipeline(
        model="gpt2",
        device=device,
    )
    generated_characters = 0
    for out in pipe(data()):
        generated_characters += len(out[0]["generated_text"])


def example_vision_classifier():
    vision_classifier = pipeline(task="image-classification", device=device)
    # vision_classifier = pipeline(model="google/vit-base-patch16-224", device=device)
    raw_predictions = vision_classifier(
        images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    )
    predictions = [
        {"score": round(pred["score"], 4), "label": pred["label"]}
        for pred in raw_predictions
    ]
    print(predictions)


def example_nlp_classifier():
    classifier = pipeline(model="facebook/bart-large-mnli", device=device)
    result = classifier(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
    print(result)


def example_preprocess_text_data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    encoded_inputs = tokenizer(
        batch_sentences, padding=True, truncation=True, return_tensors="pt"
    )
    print(encoded_inputs)


def example_preprocess_image_data():
    # load the actual images
    dataset = load_dataset("food101", split="train[:1]")

    # get a pretrained model, which is an image processor
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # determine the actual size the image processor expects
    size = image_processor.size.get("shortest_edge", None)
    if (
        not size
        and image_processor.size.get("height", None)
        and image_processor.size.get("width", None)
    ):
        size = (image_processor.size["height"], image_processor.size["width"])
    else:
        message = "size was not set. image_processor did not have keys shortest_edge, \
            or height and width."
        LOGGER.error(message)
        raise Exception(message)

    # define the transforms to apply to each image
    _transforms = Compose(
        [RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)]
    )

    def transforms(examples):
        images = [_transforms(img.convert("RGB")) for img in examples["image"]]
        examples["pixel_values"] = image_processor(
            images, do_resize=False, return_tensors="pt"
        )["pixel_values"]
        return examples

    dataset.set_transform(transforms)

    # this will show that pixel_values has been added to each image in the dataset
    dataset[0].keys()

    # the dataset is now ready to send to a model!
    return dataset


def example_fine_tune_using_trainer():
    # get dataset
    dataset = load_dataset("yelp_review_full")

    # get tokenizer from base model bert-base-cased
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # create a tokenize function that will tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # make a smaller training dataset
    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    # make a smaller eval dataset
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # create hyperparameters for training
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch"
    )

    # setup the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        device=device,
    )

    trainer.train()


def example_fine_tune_using_pytorch():
    # get dataset
    print("Loading dataset...")
    dataset = load_dataset("yelp_review_full")

    # get tokenizer from base model bert-base-cased
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # create a tokenize function that will tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove the text column because the model does not accept raw text as an input
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # Rename the label column to labels because
    # the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Set the format of the dataset to return PyTorch tensors instead of lists
    tokenized_datasets.set_format("torch")

    # make a smaller training dataset
    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    # make a smaller eval dataset
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # Create a DataLoader for your training and test datasets
    # so you can iterate over batches of data
    print("Creating dataloaders...")
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    # Load your model with the number of expected labels:
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.to(device)

    print("Start training")
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print("Start evaluation")
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    output = metric.compute()
    print(output)

    torch.save(model.state_dict(), "yelp-reviews-bert.pt")

    print("Complete")


example_fine_tune_using_pytorch()
