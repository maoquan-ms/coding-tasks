'''
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server "
    "--model-path Qwen/Qwen2.5-Coder-0.5B-Instruct "
    "--host 0.0.0.0 "
    "--mem-fraction-static 0.85 "
    "--context-length 32768 "
    "--max-total-tokens 196608 "
    "--max-prefill-tokens 49152 "
    "--max-running-requests 1024 "
    "--attention-backend flashinfer "
    "--trust-remote-code "
    "--prefill-attention-backend flashinfer "
    "--decode-attention-backend flashinfer "
    "--chunked-prefill-size 4096 "
    "--watchdog-timeout 600 "
    "--enable-torch-compile"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")

'''

from __future__ import annotations
import dataclasses
import subprocess
import time
import typing as T

def _import_launcher():
    from sglang.test.test_utils import is_in_ci
    if is_in_ci():
        from patch import launch_server_cmd 
    else:
        from sglang.utils import launch_server_cmd
    return launch_server_cmd

@dataclasses.dataclass
class ServerConfig:
    model_path: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    host: str = "0.0.0.0"
    mem_fraction_static: float = 0.85
    context_length: int = 32_768
    max_total_tokens: int = 196_608
    max_prefill_tokens: int = 49_152
    max_running_requests: int = 1024
    attention_backend: str = "flashinfer"
    trust_remote_code: bool = True
    prefill_attention_backend: str = "flashinfer"
    decode_attention_backend: str = "flashinfer"
    chunked_prefill_size: int = 4096
    watchdog_timeout: int = 600
    enable_torch_compile: bool = True

    def to_cli_args(self) -> str:
        flags = [
            f"--model-path {self.model_path}",
            f"--host {self.host}",
            f"--mem-fraction-static {self.mem_fraction_static}",
            f"--context-length {self.context_length}",
            f"--max-total-tokens {self.max_total_tokens}",
            f"--max-prefill-tokens {self.max_prefill_tokens}",
            f"--max-running-requests {self.max_running_requests}",
            f"--attention-backend {self.attention_backend}",
            f"--prefill-attention-backend {self.prefill_attention_backend}",
            f"--decode-attention-backend {self.decode_attention_backend}",
            f"--chunked-prefill-size {self.chunked_prefill_size}",
            f"--watchdog-timeout {self.watchdog_timeout}",
        ]
        if self.trust_remote_code:
            flags.append("--trust-remote-code")
        if self.enable_torch_compile:
            flags.append("--enable-torch-compile")
        return " ".join(flags)

class LLMServer:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._proc: T.Optional[subprocess.Popen] = None
        self._port: T.Optional[int] = None

    def start(self) -> str:
        launch_server_cmd = _import_launcher()
        from sglang.utils import wait_for_server, print_highlight

        cmd = f"python3 -m sglang.launch_server {self.cfg.to_cli_args()}"
        self._proc, self._port = launch_server_cmd(cmd)
        endpoint = f"http://localhost:{self._port}"

        try:
            wait_for_server(endpoint)
            print_highlight(f"✅ Server ready at {endpoint}")
            return endpoint
        except Exception:
            self.stop()
            raise RuntimeError("Server failed to start.")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            from sglang.utils import terminate_process
            terminate_process(self._proc)
            self._proc.wait()
            print("🛑 Server terminated.")
        self._proc = None
        self._port = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

if __name__ == "__main__":
    cfg = ServerConfig() 
    server = LLMServer(cfg)
    try:
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()