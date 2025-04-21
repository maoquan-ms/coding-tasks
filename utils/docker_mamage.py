import sys
import time
import docker
from docker.errors import NotFound, APIError
from queue import Queue
from multiprocessing import Process, Manager
from .safe_print import safe_print
from typing import List, Dict

# --------------------------- Docker 管理 --------------------------- #
def start_containers(image: str, host_model: str, container_model: str,
                     total_cpu: int, instances: int, base_port: int) -> List[int]:
    r"""启动 containers，返回端口列表
        args:
            image: Docker 镜像名
            host_model: 主机模型路径
            container_model: 容器模型路径
            total_cpu: 总 CPU 核心数
            instances: 实例数
            base_port: 基础端口号
        return:
            ports: 启动的端口列表
    """
    client = docker.from_env()
    threads_per_instance = total_cpu // instances
    env_threads = str(threads_per_instance)
    safe_print(f"[Docker] 每个实例分配 {threads_per_instance} OMP/MKL 线程")

    ports = []
    for i in range(instances):
        name = f"vllm_cpu_{base_port + i}"
        port = base_port + i
        ports.append(port)

        # 清理旧容器
        try:
            old = client.containers.get(name)
            safe_print(f"[Docker] 移除已存在 {name}")
            old.remove(force=True)
        except NotFound:
            pass

        safe_print(f"[Docker] 启动 {name} → 本地 {port}")
        try:
            client.containers.run(
                image=image, name=name, detach=True,
                environment={
                    "OMP_NUM_THREADS": env_threads,
                    "MKL_NUM_THREADS": env_threads
                },
                restart_policy={"Name":"unless-stopped"},
                ipc_mode="host",
                volumes={host_model: {"bind":container_model, "mode":"ro"}},
                ports={"8000/tcp": port},
                command=["--model", container_model,
                         "--host", "0.0.0.0", "--port", "8000"]
            )
        except APIError as e:
            safe_print(f"[Docker][ERROR] 启动{name}失败：{e.explanation}")
            sys.exit(1)

    safe_print("[Docker] 等待服务就绪 …")
    time.sleep(120)
    return ports

def stop_containers(instances: int, base_port: int):
    r"""停止 containers
        args:
            instances: 实例数
            base_port: 基础端口号
    """
    client = docker.from_env()
    for i in range(instances):
        name = f"vllm_cpu_{base_port + i}"
        try:
            c = client.containers.get(name)
            c.remove(force=True)
            safe_print(f"[Docker] 已移除 {name}")
        except NotFound:
            safe_print(f"[Docker] 容器 {name} 不存在")
