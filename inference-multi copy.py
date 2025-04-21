#!/usr/bin/env python3
# filename: inference_docker_dynamic.py
import argparse, json, os, re, sys, time
from typing import List, Dict
from multiprocessing import Process, Queue, Manager
import requests
from tqdm.auto import tqdm
import concurrent.futures
from concurrent.futures import FIRST_COMPLETED, wait
import docker
from docker.errors import NotFound, APIError
import collections 

from utils import (
    safe_print,
    get_samples,
    start_containers,
    stop_containers,
)

# --------------------------- Docker 配置 --------------------------- #
IMAGE                = "konstantinvernermaif/vllm-cpu:latest"
HOST_MODEL_PATH      = r"D:\subway\project\qwen-serve\Qwen2.5-Coder-0.5B-Instruct"
CONTAINER_MODEL_PATH = "/models/qwen2"
LANG_TEMPLATES = {
    "python": {
        "name": "Python",
        "target": "function implementation",
        "requirements": (
            "You are an expert Python developer. "
            "Generate ONLY the complete, runnable Python function (including signature), "
            "with correct indentation and no explanatory text, comments, or markdown. "
        )
    },
    "java": {
        "name": "Java",
        "target": "method implementation",
        "requirements": (
            "You are an expert Java developer. "
            "Generate ONLY the complete Java method (including signature and closing brace), "
            "with correct indentation and no explanatory text, comments, or markdown. "
        )
    }
}
# --------------------------- Chat Completion 调用 --------------------------- #
def generate_chat_completion(
    messages: List[Dict[str, str]],
    host: str,
    port: int,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int
) -> str:
    """调用 Chat Completion 接口"""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------- 辅助函数 ---------------------------- #
def strip_code_fence(text: str) -> str:
    """去掉代码块围栏"""
    m = re.search(r"```(?:\w+)?\s*?\n(.*?)```", text, re.S)
    return m.group(1).rstrip() if m else text

def extract_indented_body(code: str) -> str:
    """提取 Python 或 Java 的代码主体"""
    lines = code.splitlines()
    # Python 函数体提取
    if any(ln.startswith("def ") for ln in lines):
        saw = False
        out = []
        for ln in lines:
            if not saw:
                if ln.startswith("def "):
                    saw = True
                continue
            out.append(ln)
        return "\n".join(out).rstrip()
    # Java 方法体提取
    # 判断是否包含 public/private 修饰符，作为 Java 方法的标志
    if any(ln.strip().startswith(("public ", "private ", "protected ")) for ln in lines):
        start_idx = None
        # 寻找方法签名后的第一行 '{' 之后的内容
        for i, ln in enumerate(lines):
            if "{" in ln:
                start_idx = i + 1
                break
        # 寻找最后一个 '}' 的行
        end_idx = None
        for j in range(len(lines) - 1, -1, -1):
            if "}" in lines[j]:
                end_idx = j
                break
        if start_idx is not None and end_idx is not None and start_idx < end_idx:
            return "\n".join(lines[start_idx:end_idx]).rstrip()
    # 默认按缩进提取
    return "\n".join([ln for ln in lines if ln.startswith("    ") or ln.startswith("\t")]).rstrip()

# -------------------------- 写入进程 (全局 tqdm) -------------------------- #
def writer(queue: Queue, out_path: str, total_samples: int):
    """写入结果并展示全局进度条"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, ext = os.path.splitext(out_path)
    py_file = f"{base}_python{ext}"
    ja_file = f"{base}_java{ext}"

    # 根据文件是否存在决定打开模式 (w/a)
    py_mode = "a" if os.path.exists(py_file) else "w"
    ja_mode = "a" if os.path.exists(ja_file) else "w"
    f_py = open(py_file, py_mode, encoding="utf-8")
    f_ja = open(ja_file, ja_mode, encoding="utf-8")

    bar = tqdm(total=total_samples, desc="Overall", unit="task")
    while True:
        item = queue.get()
        if item is None:
            break
        line = json.dumps(item, ensure_ascii=False) + "\n"
        task_id = item.get("task_id", "").lower()
        if task_id.startswith("python/"):
            f_py.write(line); f_py.flush()
        elif task_id.startswith("java/"):
            f_ja.write(line); f_ja.flush()
        bar.update(1)
    bar.close()
    f_py.close(); f_ja.close()

# ---------------------- 动态并发 Worker ---------------------- #

def plan_methods(item, port, model, args, template) -> List[str]:
    r"""
    First stage: generate distinct solution outlines.
    Returns a brief description of each solution approach.
    """
    system = (
        f"You are an expert {template['name']} algorithm designer. "
        f"List exactly {args.pass_k} solution approaches as bullet points. "
        "Each approach must use a different core algorithmic paradigm or data structure, "
        "and should be described in one concise sentence (no numbering, no method names)."
    )


    # 辅助：将数字转成序数，如 1->1st, 2->2nd, 3->3rd, 4->4th ...
    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    # 构造动态占位符列表
    placeholders = "\n".join(
        f"- Brief description of the {ordinal(i+1)} approach"
        for i in range(args.pass_k)
    )

    # 最终 user prompt
    user = (
        f"{item['prompt']}\n\n"
        f"List exactly {args.pass_k} distinct approaches:\n"
        f"{placeholders}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    response = generate_chat_completion(
        messages, "localhost", port, model,
        max_tokens=512, temperature=0.7, timeout=args.timeout
    )

    methods = re.findall(r"^\s*[-•*]\s+(.*)$", response, re.MULTILINE)
    methods = [method.strip() for method in methods if method.strip()]

    if len(methods) < args.pass_k:
        safe_print(f"[警告] 模型返回了 {len(methods)} 个方法描述，期望是 {args.pass_k} 个，已补充默认描述。")
        methods += [f"Default description {len(methods)+i+1}" for i in range(args.pass_k - len(methods))]
    
    return methods[:args.pass_k]


def solve_one_task(
    item: Dict[str,str],
    port: int,
    model: str,
    args
) -> Dict[str, str]:
    """
    对单个问题执行：规划 → 生成 → 自检 → 提取 → 返回最终结果字典
    返回值包含 task_id、prompt、algorithm、generation、error。
    """
    lang = item["id"].split("/")[0].lower()
    template = LANG_TEMPLATES.get(lang, {
        "name": lang.capitalize(),
        "target": "code snippet",
        "requirements": "valid code only—no extras."
    })

    # 1) 规划算法
    algos = plan_methods(item, port, model, args, template)

    final_best = []
    error = None
    # 2) 针对每个算法执行多次生成 + 自检
    for algo in algos:
        for _ in range(args.runs_per_method):
            # 2a) 生成
            system = (
                f"You are a focused {template['name']} code generator. "
                f"Respond only with {template['requirements']}"
            )
            user = (
                f"Problem:\n{item['prompt']}\n\n"
                f"Approach: {algo}\n\n"
                f"Please output only the {template['target']} including closing braces."
            )
            raw = generate_chat_completion(
                [{"role":"system","content":system},
                 {"role":"user","content":user}],
                "localhost", port, model,
                args.max_tokens, args.temperature, args.timeout
            )
            # 2b) 自我审查
            snippet = strip_code_fence(raw)
            fixed = self_debug(
                snippet, algo, template,
                port, model,
                args.max_tokens, args.temperature, args.timeout,
                test_result=""
            )
            # 2c) 提取主体
            body = extract_indented_body(fixed)
            if item["id"].startswith("Java/"):
                body = body.rstrip() + "\n    }\n}"
            final_best.append({
                "algorithm": algo,
                "generation": body
            })

    # 3) 选一个或者全部返回，看你如何聚合
    # 这里我们只返回第一条作为示例
    chosen = final_best[0] if final_best else {"algorithm": "", "generation": ""}
    return {
        "task_id":    item["id"],
        "prompt":     item["prompt"],
        "algorithm":  chosen["algorithm"],
        "generation": chosen["generation"],
        "error":      error
    }


def dynamic_worker(
    port: int,
    model: str,
    queue_in,
    queue_out: Queue,
    args,
    concurrency: int
):
    """
    使用一个线程池并发地调用 solve_one_task，
    每当一个任务完成，就把结果丢到 queue_out，并补上新的。
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures: Dict[concurrent.futures.Future, None] = {}

    def submit_one():
        try:
            item = queue_in.get_nowait()
        except Exception:
            return
        fut = executor.submit(solve_one_task, item, port, model, args)
        futures[fut] = None

    # 初始并发填充
    for _ in range(concurrency):
        submit_one()

    # 动态调度
    while futures:
        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            futures.pop(fut)
            results = fut.result()
            for result in results:
                queue_out.put(result)
            submit_one()

    executor.shutdown(wait=True)


# ---------------------------- 主流程 ---------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="HumanEval‑X vLLM Chat 推理脚本 (Docker 动态并发版)"
    )
    p.add_argument("--input", help="（可选）自定义输入文件路径，支持 JSON 或 JSONL；不指定则按 --langs 下载数据集")
    p.add_argument("--download_url", help="当 --input 文件不存在时，从该 URL 下载并保存到 --input")
    p.add_argument("--output", required=True, help="结果输出前缀，会生成 <prefix>_python.jsonl 和 <prefix>_java.jsonl")

    p.add_argument("--langs", default="python,java", help="语言列表，逗号分隔，例如 python,java")
    p.add_argument("--cache_dir", default="humaneval_cache", help="数据集缓存路径")
    p.add_argument("--cpu-cores", type=int, default=16, help="宿主机总 CPU 核心数，用于分配容器线程数")
    p.add_argument("--instances", type=int, default=4, help="启动的 Docker 实例数量")
    p.add_argument("--base-port", type=int, default=8010, help="第一个实例映射端口，后续+1")
    p.add_argument("--concurrency", type=int, default=2, help="每个实例保持的并发请求数")
    p.add_argument("--pass-k", type=int, default=1,
                   help="计算 pass@k,默认为 pass@1")
    p.add_argument("--repeat", type=int, default=1,
                     help="每个算法重复生成的次数，默认为 1")
    p.add_argument("--model", default="qwen2", help="与 vLLM --served-model-name 保持一致")
    p.add_argument("--max_tokens", type=int, default=512, help="生成时 max_tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="生成时 temperature")
    p.add_argument("--timeout", type=int, default=2000, help="HTTP 请求超时（秒）")
    p.add_argument("--resume", action='store_true', help="如果指定，则从输出文件恢复进度，否则覆盖输出文件重新开始")
    args = p.parse_args()

    ports = start_containers(IMAGE, HOST_MODEL_PATH, CONTAINER_MODEL_PATH,
                              args.cpu_cores, args.instances, args.base_port)

    try:
        # get_samples 现在处理加载、下载、缓存、恢复、文件清理等逻辑
        samples_to_process, total_task = get_samples(
            input_path=args.input, 
            download_url=args.download_url,
            langs_str=args.langs,
            cache_dir=args.cache_dir,
            output_prefix=args.output, 
            resume=args.resume,
            pass_k=args.pass_k,
            repeat=args.repeat
        )

        # 如果 get_samples 返回空列表和 0 total_task，说明无需运行
        if not samples_to_process:
             safe_print("[*] 没有需要处理的任务，程序即将退出。")
             return 

        manager = Manager()
        q_in = manager.Queue()
        for s in samples_to_process: 
             q_in.put(s)

        q_out = Queue()
        # 使用 get_samples 返回的 total_task 初始化 writer
        # writer 内部会根据文件是否存在判断是 'w' 还是 'a' 模式
        wp = Process(target=writer, args=(q_out, args.output, total_task))
        wp.start()

        workers = []
        for idx, port in enumerate(ports):
            # dynamic_worker 内部会处理 pass_k 和 repeat，所以这里不需要改动
            pw = Process(target=dynamic_worker,
                         args=(port, args.model, q_in, q_out, args, args.concurrency))
            pw.start(); workers.append(pw)
        
        # 等待所有 worker 完成
        for pw in workers: 
            pw.join()

        # 所有 worker 完成后，向 writer 发送结束信号
        q_out.put(None)
        # 等待 writer 完成写入
        wp.join() 
        safe_print("[*] 所有任务处理和写入完成。")

    finally:
        safe_print("[*] 清理 Docker ...")
        stop_containers(args.instances, args.base_port)

if __name__ == "__main__":
    main()
