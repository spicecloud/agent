import csv
import io
import json
import logging
import platform
from shutil import which
import subprocess
from typing import Dict, List

from aiohttp import client_exceptions
import click
from gql import gql

from spice_agent.utils.config import (
    SPICE_TRAINING_FILEPATH,
    read_config_file,
    update_config_file,
)
from spice_agent.utils.memory_size import MemorySize

LOGGER = logging.getLogger(__name__)


class Hardware:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.previous_state = {
            "isHealthy": False,
            "isQuarantined": False,
            "isAvailable": False,
            "isOnline": False,
        }

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

    def _get_supported_metal_device(self) -> int | None:
        """
        Checks if metal hardware is supported. If so, the index
        of the supported metal device is returned
        """
        supported_metal_device = None
        is_system_profiler_available = bool(which("system_profiler"))
        if is_system_profiler_available:
            system_profiler_display_data_type_command = (
                "system_profiler SPDisplaysDataType -json"
            )
            try:
                system_profiler_display_data_type_output = subprocess.check_output(
                    system_profiler_display_data_type_command.split(" ")
                )
            except subprocess.CalledProcessError as exception:
                message = f"Error running system_profiler: {exception}"
                LOGGER.error(message)
                return supported_metal_device

            try:
                system_profiler_display_data_type_json = json.loads(
                    system_profiler_display_data_type_output
                )
            except json.JSONDecodeError as exception:
                message = f"Error decoding JSON: {exception}"
                LOGGER.error(message)
                return supported_metal_device

            # Checks if any attached displays have metal support
            # Note, other devices here could be AMD GPUs or unconfigured Nvidia GPUs
            for index, display in enumerate(
                system_profiler_display_data_type_json["SPDisplaysDataType"]
            ):
                if "spdisplays_mtlgpufamilysupport" in display:
                    supported_metal_device = index
                    return supported_metal_device

        return supported_metal_device

    def get_gpu_config(self) -> List:
        """
        For Nvidia based systems, nvidia-smi will be used to profile the gpu/s.
        For Metal based systems, we will gather information from SPDisplaysDataType.
        """
        gpu_config = []

        # Check nvidia gpu availability
        is_nvidia_smi_available = bool(which("nvidia-smi"))
        if is_nvidia_smi_available:
            nvidia_smi_query_gpu_csv_command = "nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv"  # noqa
            try:
                nvidia_smi_query_gpu_csv_output = subprocess.check_output(
                    nvidia_smi_query_gpu_csv_command.split(" "),
                )
            except subprocess.CalledProcessError as exception:
                message = f"Command {nvidia_smi_query_gpu_csv_command} failed with exception: {exception}"  # noqa
                LOGGER.error(message)
                raise exception

            try:
                nvidia_smi_query_gpu_csv_decoded = (
                    nvidia_smi_query_gpu_csv_output.decode("utf-8")
                    .replace("\r", "")
                    .replace(", ", ",")
                    .lstrip("\n")
                )
            except UnicodeDecodeError as exception:
                message = f"Error decoding: {exception}"
                LOGGER.error(message)
                raise exception

            nvidia_smi_query_gpu_csv_dict_reader = csv.DictReader(
                io.StringIO(nvidia_smi_query_gpu_csv_decoded)
            )

            for gpu_info in nvidia_smi_query_gpu_csv_dict_reader:
                # Refactor key into B
                memory_total_in_mebibytes = gpu_info.pop("memory.total [MiB]")
                memory_size = MemorySize(memory_total_in_mebibytes)
                gpu_info["memory_total"] = memory_size.to_bytes()

                gpu_config.append(gpu_info)

        if platform.system() == "Darwin":
            # Check Metal gpu availability
            supported_metal_device = self._get_supported_metal_device()
            if supported_metal_device is not None:
                # Since Apple's SoC contains Metal,
                # we query the system itself for total memory
                system_profiler_hardware_data_type_command = (
                    "system_profiler SPHardwareDataType -json"
                )

                try:
                    system_profiler_hardware_data_type_output = subprocess.check_output(
                        system_profiler_hardware_data_type_command.split(" ")
                    )
                except subprocess.CalledProcessError as exception:
                    message = f"Error running {system_profiler_hardware_data_type_command}: {exception}"  # noqa
                    LOGGER.error(message)
                    raise exception

                try:
                    system_profiler_hardware_data_type_json = json.loads(
                        system_profiler_hardware_data_type_output
                    )
                except json.JSONDecodeError as exception:
                    message = f"Error decoding JSON: {exception}"  # noqa
                    LOGGER.error(message)
                    raise exception

                metal_device_json = system_profiler_hardware_data_type_json[
                    "SPHardwareDataType"
                ][supported_metal_device]

                gpu_info = {}
                gpu_info["name"] = metal_device_json.get("chip_type")

                # Refactor key into B
                physical_memory = metal_device_json.get("physical_memory")
                memory_size = MemorySize(physical_memory)
                gpu_info["memory_total"] = memory_size.to_bytes()

                gpu_config.append(gpu_info)

        # Raise an error if there is no valid gpu config
        if not gpu_config:
            message = "No valid gpu configuration"
            LOGGER.error(message)
            raise NotImplementedError(message)

        return gpu_config

    def get_windows_computer_service_product_values(self) -> Dict[str, str]:
        windows_computer_service_product_csv_command = (
            "cmd.exe /C wmic csproduct get Name, Vendor, Version, UUID /format:csv"
        )
        windows_computer_service_product_csv_output = subprocess.check_output(
            windows_computer_service_product_csv_command.split(" "),
            stderr=subprocess.DEVNULL,
        )
        windows_computer_service_product_csv_decoded = (
            windows_computer_service_product_csv_output.decode("utf-8")
            .replace("\r", "")
            .lstrip("\n")
        )
        windows_computer_service_product_dict = csv.DictReader(
            io.StringIO(windows_computer_service_product_csv_decoded)
        )
        csp_info = list(windows_computer_service_product_dict)[0]
        return {
            "windows_model_name": csp_info.get("Name", ""),
            "windows_model_vendor": csp_info.get("Vendor", ""),
            "windows_model_version": csp_info.get("Version", ""),
            "windows_model_uuid": csp_info.get("UUID", ""),
        }

    def get_windows_cpu_values(self) -> Dict[str, str]:
        windows_cpu_csv_command = (
            "cmd.exe /C wmic cpu get Name, MaxClockSpeed /format:csv"  # noqa
        )
        windows_cpu_csv_output = subprocess.check_output(
            windows_cpu_csv_command.split(" "),
            stderr=subprocess.DEVNULL,
        )
        windows_cpu_csv_decoded = (
            windows_cpu_csv_output.decode("utf-8").replace("\r", "").lstrip("\n")
        )
        windows_cpu_dict = csv.DictReader(io.StringIO(windows_cpu_csv_decoded))
        cpu_info = list(windows_cpu_dict)[0]
        return {
            "cpu_brand": cpu_info.get("Name", "").strip(),
            "cpu_max_clock_speed": cpu_info.get("MaxClockSpeed", ""),
        }

    def get_ubuntu_values(self) -> Dict[str, str]:
        get_machine_id_command = "cat /etc/machine-id"
        machine_id = subprocess.check_output(
            get_machine_id_command.split(" "),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        if machine_id:
            return {"linux_machine_id": machine_id}
        return {}

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
                    **self.get_windows_computer_service_product_values(),
                    **self.get_windows_cpu_values(),
                }
            else:
                system_info = {**system_info, **self.get_ubuntu_values()}
        elif os_family == "Windows":
            system_info = {
                **system_info,
                **self.get_windows_computer_service_product_values(),
                **self.get_windows_cpu_values(),
            }

        system_info["name"] = platform.node()
        system_info["os_family"] = os_family
        system_info["os_release"] = platform.release()
        system_info["os_version"] = platform.version()
        system_info["platform"] = platform.platform()
        system_info["processor"] = platform.processor()
        system_info["machine"] = platform.machine()
        system_info["architecture"] = platform.architecture()[0]
        system_info["cpu_cores"] = str(platform.os.cpu_count())  # type: ignore exits
        system_info["gpu_config"] = self.get_gpu_config()
        return system_info

    def register(self):
        system_info = self.get_system_info()
        mutation = gql(
            """
            mutation registerHardware($input: RegisterHardwareInput!) {
                registerHardware(input: $input) {
                    ... on RegisterHardware {
                        fingerprint
                        rabbitmqPassword
                        rabbitmqHost
                        rabbitmqPort
                    }
                }
            }
        """
        )
        input = {"systemInfo": system_info}
        variables = {"input": input}
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
        update_config_file(new_config=self.spice.full_config)

        # then check in that the hardware, validate that it is saved correctly
        # and headers are
        self.spice.create_session()
        self.check_in_http(is_healthy=True)
        return result

    def check_in_http(
        self,
        is_healthy: bool = True,
        is_quarantined: bool = False,
        is_available: bool = False,
        is_online: bool = True,
    ):
        fingerprint = self.spice.host_config.get("fingerprint", None)
        if not fingerprint:
            message = "No fingerprint found. Please register: spice hardware register"
            raise Exception(message)

        new_state = {
            "isHealthy": is_healthy,
            "isQuarantined": is_quarantined,
            "isAvailable": is_available,
            "isOnline": is_online,
        }

        mutation = gql(
            """
            mutation checkIn($input: CheckInInput!) {
                checkIn(input: $input) {
                    ... on Hardware {
                        createdAt
                        updatedAt
                        lastCheckIn
                    }
                }
            }
        """  # noqa
        )
        input = new_state
        variables = {"input": input}
        try:
            result = self.spice.session.execute(mutation, variable_values=variables)
            previous_state = self.previous_state
            self.previous_state = new_state.copy()

            message = " [*] Checked in successfully: "
            for key, value in new_state.items():
                if value != previous_state.get(key, None):
                    if value is True:
                        message = (
                            message
                            + click.style(f"{key}: ")
                            + click.style("💤")
                            + click.style(" ==> ✅ ", fg="green")
                        )
                    else:
                        message = (
                            message
                            + click.style(f"{key}: ")
                            + click.style("✅")
                            + click.style(" ==> 💤 ", fg="yellow")
                        )
            LOGGER.info(message)

            return result
        except client_exceptions.ClientConnectorError:
            config = read_config_file(filepath=SPICE_TRAINING_FILEPATH)
            if config.get("status") in self.spice.worker.ACTIVE_STATUSES:
                return None
