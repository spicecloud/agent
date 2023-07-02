import platform

from spice_agent.daemons.launch_agents import (
    full_launch_agent_install,
    full_launch_agent_uninstall,
    start_launch_agent,
    stop_launch_agent,
    view_launch_agent_logs,
)

from spice_agent.daemons import launch_service
from spice_agent.daemons.utils import launch_gpu_monitor_service
from spice_agent.daemons.utils.gpu_monitor import update_service_from_gpu_monitor


class Daemons:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.os_family = platform.system()

    def install(self):
        result = True
        if self.os_family == "Darwin":
            full_launch_agent_install()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                launch_service.full_service_install()
            else:
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def uninstall(self):
        result = True
        if self.os_family == "Darwin":
            full_launch_agent_uninstall()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                launch_service.full_service_uninstall()
            else:
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def stop(self) -> bool:
        result = True
        if self.os_family == "Darwin":
            result = stop_launch_agent()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                launch_service.stop_service()
            else:
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def start(self) -> bool:
        result = True
        if self.os_family == "Darwin":
            result = start_launch_agent()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                launch_service.start_service()
            else:
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")
        return result

    def logs(self):
        if self.os_family == "Darwin":
            view_launch_agent_logs()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                launch_service.view_service_logs()
            else:
                print("Not Implemented")
        elif self.os_family == "Windows":
            print("Not Implemented")

    def auto(self):
        """
        Launches gpu monitoring service that automatically turns on/off
        the spice agent daemon for user convenience. Thresholds can be found in
        gpu_monitor.py.
        """
        if self.os_family == "Darwin":
            raise NotImplementedError()
        elif self.os_family == "Linux":
            if "WSL2" in platform.platform():
                pass
            else:
                raise NotImplementedError()
        elif self.os_family == "Windows":
            raise NotImplementedError()

        # Install gpu_monitor service
        if not launch_gpu_monitor_service.is_service_enabled():
            launch_gpu_monitor_service.full_service_install()

        # Updates spice daemon using gpu monitor
        update_service_from_gpu_monitor()
