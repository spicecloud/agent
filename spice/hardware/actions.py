import json
import platform
import subprocess
from typing import Dict

from gql import gql



class Hardware:
    def __init__(self, spice) -> None:
        self.spice = spice

    def get_darwin_system_profiler_values(self) -> Dict[str, str]:
        system_profiler_hardware_data_type = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"]
        )
        system_profiler_hardware_data = json.loads(system_profiler_hardware_data_type)
        data_type = system_profiler_hardware_data.get("SPHardwareDataType")[0]
        return {
            "apple_model_name": data_type.get("machine_model"),
            "apple_model_identifier": data_type.get("machine_name"),
            "apple_model_number": data_type.get("model_number"),
            "physical_memory": data_type.get("physical_memory"),
            "apple_serial_number": data_type.get("serial_number"),
        }

    def get_system_info(self):
        os_family = platform.system()
        system_info = {}
        if os_family == "Darwin":
            system_info = {**system_info, **self.get_darwin_system_profiler_values()}
            system_info["cpu_brand"] = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .strip()
                .decode("utf-8")
            )
            system_info["apple_mac_os_version"] = platform.mac_ver()[0]

        system_info["name"] = platform.node()
        system_info["os_family"] = os_family
        system_info["os_release"] = platform.release()
        system_info["os_version"] = platform.version()
        system_info["platform"] = platform.platform()
        system_info["processor"] = platform.processor()
        system_info["machine"] = platform.machine()
        system_info["architecture"] = platform.architecture()[0]

        system_info["cpu_cores"] = str(platform.os.cpu_count())
        return system_info

    def register(self):
        system_info = self.get_system_info()
        mutation = gql(
            """
            mutation registerHardware($systemInfo: JSON!) {
                registerHardware(systemInfo: $systemInfo) {
                    systemInfo
                    createdAt
                    updatedAt
                    fingerPrint
                }
            }
        """
        )
        params = {"systemInfo": system_info}
        result = self.spice.session.execute(mutation, variable_values=params)
        result.get("registerHardware").get("fingerPrint")
        return result
