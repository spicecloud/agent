from pathlib import Path
import subprocess

home_directory = Path.home()
SPICE_LAUNCH_AGENT_FILEPATH = Path(
    home_directory / "Library" / "LaunchAgents" / "cloud.spice.agent.plist"
)
SPICE_LAUNCH_AGENT_LOGS_DIR = Path(home_directory / "Logs" / "cloud.spice.agent.log")
SPICE_LAUNCH_AGENT_NAME = "cloud.spice.agent"
SPICE_LAUNCH_AGENT_WORKING_DIR = Path(home_directory / "Logs" / "cloud.spice.agent.log")


def start_launch_agent():
    try:
        stop_existing_process = f"launchctl stop {SPICE_LAUNCH_AGENT_NAME}"
        subprocess.check_output(stop_existing_process.split(" "))
    except subprocess.CalledProcessError:
        pass

    try:
        remove_existing_agent = f"launchctl remove {SPICE_LAUNCH_AGENT_NAME}"
        subprocess.check_output(remove_existing_agent.split(" "))
    except subprocess.CalledProcessError:
        pass

    SPICE_LAUNCH_AGENT_LOGS_DIR.parent.mkdir(parents=True, exist_ok=True)
    SPICE_LAUNCH_AGENT_LOGS_DIR.touch()

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
    <string>{SPICE_LAUNCH_AGENT_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/dylan/.pyenv/shims/spice</string>
        <string>inference</string>
        <string>worker</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{SPICE_LAUNCH_AGENT_LOGS_DIR}</string>
    <key>StandardOutPath</key>
    <string>{SPICE_LAUNCH_AGENT_LOGS_DIR}</string>
</dict>
</plist>"""
    )
    SPICE_LAUNCH_AGENT_FILEPATH.chmod(0o644)

    load_new_plist = f"launchctl load {SPICE_LAUNCH_AGENT_FILEPATH}"
    subprocess.check_output(load_new_plist.split(" "))

    start_new_agent = f"launchctl start {SPICE_LAUNCH_AGENT_NAME}"
    subprocess.check_output(start_new_agent.split(" "))
