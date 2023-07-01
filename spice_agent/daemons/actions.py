import platform

from spice_agent.daemons.launch_agents import (
    full_launch_agent_install,
    full_launch_agent_uninstall,
    start_launch_agent,
    stop_launch_agent,
    view_launch_agent_logs,
)


class Daemons:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.os_family = platform.system()

    def install(self):
        result = True
        if self.os_family == "Darwin":
            full_launch_agent_install()
        elif self.os_family == "Linux":
            print("Not Implemented")
            if "WSL2" in platform.platform():
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def uninstall(self):
        result = True
        if self.os_family == "Darwin":
            full_launch_agent_uninstall()
        elif self.os_family == "Linux":
            print("Not Implemented")
            if "WSL2" in platform.platform():
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def stop(self) -> bool:
        result = True
        if self.os_family == "Darwin":
            result = stop_launch_agent()
        elif self.os_family == "Linux":
            print("Not Implemented")
            if "WSL2" in platform.platform():
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def start(self) -> bool:
        result = True
        if self.os_family == "Darwin":
            result = start_launch_agent()
        elif self.os_family == "Linux":
            print("Not Implemented")
            if "WSL2" in platform.platform():
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def logs(self):
        if self.os_family == "Darwin":
            view_launch_agent_logs()
        elif self.os_family == "Linux":
            print("Not Implemented")
            if "WSL2" in platform.platform():
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")