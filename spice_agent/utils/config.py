import json
import os
from pathlib import Path
import shutil
from typing import Dict

HOME_DIRECTORY = Path.home()
SPICE_INFERENCE_DIRECTORY = Path(HOME_DIRECTORY / ".cache" / "spice" / "inference")
SPICE_HOSTS_FILEPATH = Path(HOME_DIRECTORY / ".config" / "spice" / "hosts.json")
SPICE_TRAINING_FILEPATH = Path(HOME_DIRECTORY / ".config" / "spice" / "training.json")
SPICE_ROUND_VERIFICATION_FILEPATH = Path(
    HOME_DIRECTORY / ".config" / "spice" / "training-verification.json"
)
SPICE_MODEL_CACHE_FILEPATH = Path(HOME_DIRECTORY / ".cache" / "spice" / "models")
HF_HUB_DIRECTORY = Path(HOME_DIRECTORY / ".cache" / "huggingface" / "hub")


def create_directory(filepath: Path = SPICE_MODEL_CACHE_FILEPATH):
    filepath.mkdir(parents=True, exist_ok=True)


def create_config_file(filepath: Path = SPICE_HOSTS_FILEPATH):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.touch()
    with filepath.open("w") as json_file:
        json.dump({}, json_file)


def update_config_file(filepath: Path = SPICE_HOSTS_FILEPATH, new_config: Dict = {}):
    existing_config = read_config_file(filepath=filepath)
    merged_config = {**existing_config, **new_config}
    with filepath.open("w") as json_file:
        json.dump(merged_config, json_file, indent=2)


def read_config_file(filepath: Path = SPICE_HOSTS_FILEPATH) -> Dict:
    if not filepath.exists():
        create_config_file(filepath=filepath)
    return json.loads(filepath.read_text())


def copy_directory(src_dir, dst_dir, ignore_files=["config.json", "pytorch_model.bin"]):
    """
    Copies current training_round_model_repo config files (e.g. tokenizer, vocab)
    into model cache for upload to s3 and hf
    """
    for file in os.listdir(src_dir):
        if file in ignore_files:
            continue

        src_path = Path(src_dir / file)
        dst_path = Path(dst_dir / file)
        shutil.copy2(src_path, dst_path)
