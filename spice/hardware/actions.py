import io
import csv
import json
import platform
import subprocess
from typing import Dict

from gql import gql

from spice.utils.config import update_config_file


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

    def get_windows_computer_product_values(self) -> Dict[str, str]:
        windows_computer_product_csv = subprocess.check_output(
            [
                "cmd.exe",
                "/C",
                "wmic",
                "csproduct",
                "get",
                "Name,",
                "Vendor,",
                "Version,",
                "UUID",
                "/format:csv",
            ],
            stderr=subprocess.DEVNULL,
        )
        windows_computer_product_csv = (
            windows_computer_product_csv.decode("utf-8").replace("\r", "").lstrip("\n")
        )
        windows_computer_product_csv = csv.DictReader(
            io.StringIO(windows_computer_product_csv)
        )
        csp_info = list(windows_computer_product_csv)[0]
        return {
            "windows_model_name": csp_info.get("Name"),
            "windows_model_vendor": csp_info.get("Vendor"),
            "windows_model_version": csp_info.get("Version"),
            "windows_model_uuid": csp_info.get("UUID"),
        }

    def get_windows_cpu_values(self) -> Dict[str, str]:
        windows_cpu_csv = subprocess.check_output(
            [
                "cmd.exe",
                "/C",
                "wmic",
                "cpu",
                "get",
                "Name,",
                "MaxClockSpeed",
                "/format:csv",
            ],
            stderr=subprocess.DEVNULL,
        )
        windows_cpu_csv = windows_cpu_csv.decode("utf-8").replace("\r", "").lstrip("\n")
        windows_cpu_csv = csv.DictReader(io.StringIO(windows_cpu_csv))
        cpu_info = list(windows_cpu_csv)[0]
        return {
            "cpu_brand": cpu_info.get("Name").strip(),
            "cpu_max_clock_speed": cpu_info.get("MaxClockSpeed"),
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
        elif os_family == "Linux":
            # Support for Linux-based VMs in Windows
            if "WSL2" in platform.platform():
                system_info = {
                    **system_info,
                    **self.get_windows_computer_product_values(),
                    **self.get_windows_cpu_values(),
                }
        elif os_family == "Windows":
            system_info = {
                **system_info,
                **self.get_windows_computer_product_values(),
                **self.get_windows_cpu_values(),
            }

    def register(self):
        system_info = self.get_system_info()
        mutation = gql(
            """
            mutation registerHardware($systemInfo: JSON!) {
                registerHardware(systemInfo: $systemInfo) {
                    fingerprint
                    rabbitmqPassword
                    rabbitmqHost
                    rabbitmqPort
                }
            }
        """
        )
        variables = {"systemInfo": system_info}
        result = self.spice.session.execute(mutation, variable_values=variables)
        fingerprint = result.get("registerHardware").get("fingerprint")
        rabbitmq_password = result.get("registerHardware").get("rabbitmqPassword")
        rabbitmq_host = result.get("registerHardware").get("rabbitmqHost")
        rabbitmq_port = result.get("registerHardware").get("rabbitmqPort")

        self.spice.host_config["fingerprint"] = fingerprint
        self.spice.host_config["rabbitmq_password"] = rabbitmq_password
        self.spice.host_config["rabbitmq_host"] = rabbitmq_host
        self.spice.host_config["rabbitmq_port"] = rabbitmq_port
        self.spice.full_config[self.spice.host] = self.spice.host_config
        update_config_file(self.spice.full_config)
        return result

    def check_in_http(
        self,
        is_healthy: bool = True,
        is_quarantined: bool = False,
        is_available: bool = True,
    ):
        mutation = gql(
            """
            mutation checkIn($isHealthy: Boolean!, $isQuarantined: Boolean!, $isAvailable: Boolean!) {
                checkIn(isHealthy: $isHealthy, isQuarantined: $isQuarantined, isAvailable: $isAvailable) {
                    createdAt
                    updatedAt
                    lastCheckIn
                }
            }
        """  # noqa
        )
        variables = {
            "isHealthy": is_healthy,
            "isQuarantined": is_quarantined,
            "isAvailable": is_available,
        }
        result = self.spice.session.execute(mutation, variable_values=variables)
        return result
