import os
from pathlib import Path
import subprocess

HOME_DIRECTORY = Path.home()
CURRENT_USER = str(HOME_DIRECTORY.expanduser()).lstrip("/Users/")

SPICE_BINARY_PATH = Path(HOME_DIRECTORY / ".pyenv" / "shims" / "spice")
SPICE_LAUNCH_AGENT_FILEPATH = Path(
    HOME_DIRECTORY / "Library" / "LaunchAgents" / "cloud.spice.agent.plist"
)
SPICE_LAUNCH_AGENT_LOGS = Path(
    HOME_DIRECTORY / "Library" / "Logs" / "cloud.spice.agent.log"
)
SPICE_LAUNCH_AGENT_LABEL = "cloud.spice.agent"
SPICE_LAUNCH_AGENT_WORKING_DIR = Path(HOME_DIRECTORY / "Logs" / "cloud.spice.agent.log")


def stop_launch_agent():
    try:
        stop_existing_process = f"launchctl stop {SPICE_LAUNCH_AGENT_LABEL}"
        subprocess.check_output(stop_existing_process.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("stop_launch_agent: ", exception)
        return False


def start_launch_agent():
    try:
        start_new_agent = f"launchctl start {SPICE_LAUNCH_AGENT_LABEL}"
        subprocess.check_output(start_new_agent.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("start_launch_agent: ", exception)
        return False


def unload_launch_agent():
    try:
        remove_existing_agent = f"launchctl unload -w {SPICE_LAUNCH_AGENT_FILEPATH}"
        subprocess.check_output(remove_existing_agent.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("remove_launch_agent: ", exception)
        return False


def load_launch_agent():
    try:
        load_new_plist = f"launchctl load -w {SPICE_LAUNCH_AGENT_FILEPATH}"
        subprocess.check_output(load_new_plist.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("load_launch_agent: ", exception)
        return False


def create_stdout_file():
    if not SPICE_LAUNCH_AGENT_LOGS.exists():
        SPICE_LAUNCH_AGENT_LOGS.parent.mkdir(parents=True, exist_ok=True)
        SPICE_LAUNCH_AGENT_LOGS.touch()


def delete_stdout_file():
    if SPICE_LAUNCH_AGENT_LOGS.exists():
        SPICE_LAUNCH_AGENT_LOGS.unlink()


def populate_fresh_launch_agent():
    SPICE_LAUNCH_AGENT_LOGS.parent.mkdir(parents=True, exist_ok=True)
    SPICE_LAUNCH_AGENT_LOGS.touch()

    if SPICE_LAUNCH_AGENT_FILEPATH.exists():
        SPICE_LAUNCH_AGENT_FILEPATH.unlink()
    SPICE_LAUNCH_AGENT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    SPICE_LAUNCH_AGENT_FILEPATH.touch()

    SPICE_LAUNCH_AGENT_FILEPATH.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>KeepAlive</key>
    <true/>
    <key>Label</key>
    <string>{SPICE_LAUNCH_AGENT_LABEL}</string>
    <key>LimitLoadToSessionType</key>
	<array>
		<string>Aqua</string>
		<string>Background</string>
		<string>LoginWindow</string>
		<string>StandardIO</string>
	</array>
    <key>ProgramArguments</key>
    <array>
        <string>{SPICE_BINARY_PATH}</string>
        <string>worker</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{SPICE_LAUNCH_AGENT_LOGS}</string>
    <key>StandardOutPath</key>
    <string>{SPICE_LAUNCH_AGENT_LOGS}</string>
</dict>
</plist>"""  # noqa: E501
    )
    SPICE_LAUNCH_AGENT_FILEPATH.chmod(0o644)
    verify_launch_agent()


def verify_launch_agent():
    plutil_check = f"plutil -lint {SPICE_LAUNCH_AGENT_FILEPATH}"
    try:
        subprocess.check_output(plutil_check.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("verify_launch_agent: ", exception)
        return False


def view_launch_agent_logs():
    follow_logs = f"tail -f -n +1 {SPICE_LAUNCH_AGENT_LOGS}"
    os.system(follow_logs)


def full_launch_agent_install():
    stop_launch_agent()
    unload_launch_agent()
    populate_fresh_launch_agent()
    create_stdout_file()
    load_launch_agent()
    start_launch_agent()


def full_launch_agent_uninstall():
    stop_launch_agent()
    unload_launch_agent()
    if SPICE_LAUNCH_AGENT_FILEPATH.exists():
        SPICE_LAUNCH_AGENT_FILEPATH.unlink()
    delete_stdout_file()
