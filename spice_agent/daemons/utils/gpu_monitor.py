import csv
import subprocess

from spice_agent.daemons.launch_service import (
    start_service,
    stop_service,
    is_service_active,
)

PRE_MEMORY_UTILIZATION_THRESHOLD = 50  # prior to starting launch_service
POST_MEMORY_UTILIZATION_THRESHOLD = 95  # after starting launch_service


# TODO: mps_utilization
def get_gpu_utilization():
    try:
        output = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                    "--format=csv,nounits",
                ]
            )
            .decode()
            .strip()
        )

        gpu_stats = csv.reader(output.splitlines(), delimiter=",")
        headers = next(gpu_stats)
        info = {}

        for data in gpu_stats:
            for j, item in enumerate(data):
                # print(f"{headers[j].strip()}: {item.strip()}")
                header = headers[j].strip().split(" ")[0]
                value = item.strip()
                info[header] = value

        return info

    except subprocess.CalledProcessError:
        print("Error retrieving GPU utilization.")


def get_memory_utilization():
    gpu_utilization = get_gpu_utilization()
    if gpu_utilization:
        memory_total = float(gpu_utilization["memory.total"])
        memory_used = float(gpu_utilization["memory.used"])
        memory_utilization = (memory_used / memory_total) * 100
        return memory_utilization
    else:
        raise ValueError(f"get_memory_utilization: {gpu_utilization}")


def update_service_from_gpu_monitor():
    memory_utilization = get_memory_utilization()

    # TODO: remove spice agent calls - these should be purely systemctl

    if is_service_active() and memory_utilization >= POST_MEMORY_UTILIZATION_THRESHOLD:
        stop_service()
    elif memory_utilization < PRE_MEMORY_UTILIZATION_THRESHOLD:
        start_service()
