import platform

from spice.daemons.launch_agents import (
    load_launch_agent,
    populate_fresh_launch_agent,
    remove_launch_agent,
    start_launch_agent,
    stop_launch_agent,
    view_launch_agent_logs,
)


class Daemons:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.os_family = platform.system()

    def install(self):
        stop_launch_agent()
        remove_launch_agent()
        populate_fresh_launch_agent()
        load_launch_agent()
        start_launch_agent()

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
