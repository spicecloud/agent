import json
from pathlib import Path
from typing import Dict


def get_config_filepath() -> Path:
    home_directory = Path.home()
    spice_config_filepath = Path(home_directory / ".config" / "spice" / "hosts.json")
    return spice_config_filepath


def create_config_file():
    spice_config_filepath = get_config_filepath()
    spice_config_filepath.parent.mkdir(parents=True, exist_ok=True)
    spice_config_filepath.touch()
    with spice_config_filepath.open("w") as json_file:
        json.dump({}, json_file)


def update_config_file(new_config: Dict):
    existing_config = read_config_file()
    spice_config_filepath = get_config_filepath()
    merged_config = {**existing_config, **new_config}
    with spice_config_filepath.open("w") as json_file:
        json.dump(merged_config, json_file)


def read_config_file() -> Dict:
    spice_config_filepath = get_config_filepath()
    if not spice_config_filepath.exists():
        create_config_file()
    return json.loads(spice_config_filepath.read_text())
