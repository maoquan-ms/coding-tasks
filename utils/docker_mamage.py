#!/usr/bin/env python3
# filename: docker_manager.py

"""Utility functions for starting and stopping vLLM CPU Docker containers."""

from __future__ import annotations

import sys
import time
from typing import List

import docker
from docker.errors import APIError, NotFound

from .safe_print import safe_print

# ---------------------------------------------------------------------------
# Docker management helpers
# ---------------------------------------------------------------------------

def start_containers(
    image: str,
    host_model: str,
    container_model: str,
    total_cpu: int,
    instances: int,
    base_port: int,
) -> List[int]:
    r"""Launch multiple vLLM CPU containers and return the list of exposed host ports.

    Args:
        image: Docker image name.
        host_model: Path to the model directory on the host.
        container_model: Mount point inside the container.
        total_cpu: Total number of CPU cores available on the host.
        instances: Number of containers to launch.
        base_port: Port for the first container; each subsequent container uses +1.

    Returns:
        A list of host ports used by the newly started containers.
    """
    client = docker.from_env()
    threads_per_instance = max(total_cpu // instances, 1)
    env_threads = str(threads_per_instance)
    safe_print(f"[Docker] {threads_per_instance} OMP/MKL threads per instance")

    ports: List[int] = []
    for i in range(instances):
        name = f"vllm_cpu_{base_port + i}"
        port = base_port + i
        ports.append(port)

        # Remove any stale container with the same name.
        try:
            old = client.containers.get(name)
            safe_print(f"[Docker] Removing existing container {name}")
            old.remove(force=True)
        except NotFound:
            pass

        safe_print(f"[Docker] Starting {name} → host port {port}")
        try:
            client.containers.run(
                image=image,
                name=name,
                detach=True,
                environment={
                    "OMP_NUM_THREADS": env_threads,
                    "MKL_NUM_THREADS": env_threads,
                },
                restart_policy={"Name": "unless-stopped"},
                ipc_mode="host",
                volumes={host_model: {"bind": container_model, "mode": "ro"}},
                ports={"8000/tcp": port},
                command=[
                    "--model",
                    container_model,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
            )
        except APIError as exc:
            safe_print(f"[Docker][ERROR] Failed to start {name}: {exc.explanation}")
            sys.exit(1)

    safe_print("[Docker] Waiting for services to become ready…")
    time.sleep(120)
    return ports


def stop_containers(instances: int, base_port: int) -> None:
    r"""Stop and remove the vLLM CPU containers that were previously launched.

    Args:
        instances: Number of containers to stop.
        base_port: Port of the first container.
    """
    client = docker.from_env()
    for i in range(instances):
        name = f"vllm_cpu_{base_port + i}"
        try:
            container = client.containers.get(name)
            container.remove(force=True)
            safe_print(f"[Docker] Removed container {name}")
        except NotFound:
            safe_print(f"[Docker] Container {name} does not exist")
