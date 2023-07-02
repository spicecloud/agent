import configparser
import subprocess
from pathlib import Path


HOME_DIRECTORY = Path.home()

SPICE_BINARY_PATH = Path(HOME_DIRECTORY / ".pyenv" / "shims" / "spice")

SPICE_GPU_MONITOR_SERVICE = "cloud.spice.gpu-monitor.service"
SPICE_GPU_MONITOR_SERVICE_FILEPATH = Path(
    HOME_DIRECTORY / ".config" / "systemd" / "user" / SPICE_GPU_MONITOR_SERVICE
)


def stop_service():
    try:
        if is_service_active():
            stop_existing_service = f"systemctl --user stop {SPICE_GPU_MONITOR_SERVICE}"
            subprocess.check_output(stop_existing_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("stop_service: ", exception)
        return False


def is_service_active():
    try:
        is_service_active = (
            f"systemctl --user show {SPICE_GPU_MONITOR_SERVICE} -p ActiveState --value"
        )
        output = subprocess.check_output(is_service_active.split(" ")).decode().strip()
        return output == "active"
    except subprocess.CalledProcessError as exception:
        print("is_service_active: ", exception)
        return False


def disable_service():
    try:
        if is_service_enabled():
            disable_existing_service = (
                f"systemctl --user disable {SPICE_GPU_MONITOR_SERVICE}"
            )
            subprocess.check_output(disable_existing_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("disable_service: ", exception)
        return False


def is_service_enabled():
    try:
        is_service_active = f"systemctl --user show {SPICE_GPU_MONITOR_SERVICE} -p UnitFileState --value"  # noqa
        output = subprocess.check_output(is_service_active.split(" ")).decode().strip()
        return output == "enabled"
    except subprocess.CalledProcessError as exception:
        print("is_service_enabled: ", exception)
        return False


def populate_service_file():
    if SPICE_GPU_MONITOR_SERVICE_FILEPATH.exists():
        SPICE_GPU_MONITOR_SERVICE_FILEPATH.unlink()
    SPICE_GPU_MONITOR_SERVICE_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    SPICE_GPU_MONITOR_SERVICE_FILEPATH.touch()

    config = configparser.ConfigParser()
    config.optionxform = str  # type: ignore

    config["Unit"] = {
        "Description": "Spice Agent",
        "After": "network.target",
    }

    config["Service"] = {
        "ExecStart": f"{SPICE_BINARY_PATH} daemon auto",
        "Restart": "always",
    }

    config["Install"] = {
        "WantedBy": "default.target",
    }

    try:
        with open(SPICE_GPU_MONITOR_SERVICE_FILEPATH, "w") as service_file:
            config.write(service_file)
    except IOError as exception:
        print(f"populate_service_file: {exception}")

    SPICE_GPU_MONITOR_SERVICE_FILEPATH.chmod(0o644)
    verify_service_file()


def verify_service_file():
    systemctl_check = (
        f"systemctl --user show {SPICE_GPU_MONITOR_SERVICE} -p CanStart --value"
    )
    try:
        output = subprocess.check_output(systemctl_check.split(" ")).decode().strip()
        if output == "no":
            raise Exception(f"{systemctl_check} yielded {output}")
        return True
    except subprocess.CalledProcessError as exception:
        print("verify_service_file: ", exception)
        return False


def enable_service():
    try:
        enable_service = f"systemctl --user enable {SPICE_GPU_MONITOR_SERVICE}"
        subprocess.check_output(enable_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("enable_service: ", exception)
        return False


def start_service():
    try:
        start_new_service = f"systemctl --user start {SPICE_GPU_MONITOR_SERVICE}"
        subprocess.check_output(start_new_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("start_service: ", exception)
        return False


def full_service_install():
    stop_service()
    disable_service()
    populate_service_file()
    enable_service()
    start_service()


def full_service_uninstall():
    stop_service()
    disable_service()
    if SPICE_GPU_MONITOR_SERVICE_FILEPATH.exists():
        SPICE_GPU_MONITOR_SERVICE_FILEPATH.unlink()
