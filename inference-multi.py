#!/usr/bin/env python3
# filename: inference_docker_dynamic.py

"""HumanEval‑X inference script using vLLM inside Docker containers with dynamic concurrency."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Process, Queue, Manager
from typing import Dict, List

import concurrent.futures
from concurrent.futures import FIRST_COMPLETED, wait

import requests
from tqdm.auto import tqdm

import docker
from docker.errors import APIError, NotFound

from utils import (
    safe_print,
    get_samples,
    start_containers,
    stop_containers,
)

# ---------------------------------------------------------------------------
# Docker Configuration
# ---------------------------------------------------------------------------
IMAGE = "konstantinvernermaif/vllm-cpu:latest"
HOST_MODEL_PATH = r"D:\subway\project\qwen-serve\Qwen2.5-Coder-0.5B-Instruct"
CONTAINER_MODEL_PATH = "/models/qwen2"

LANG_TEMPLATES = {
    "python": {
        "name": "Python",
        "target": "function implementation",
        "requirements": (
            "You are an expert Python developer. "
            "Generate ONLY the complete, runnable Python function (including signature) "
            "with correct indentation and no explanatory text, comments, or markdown."
        ),
    },
    "java": {
        "name": "Java",
        "target": "method implementation",
        "requirements": (
            "You are an expert Java developer. "
            "Generate ONLY the complete Java method (including signature and closing brace) "
            "with correct indentation and no explanatory text, comments, or markdown."
        ),
    },
}

# ---------------------------------------------------------------------------
# Chat Completion Wrapper
# ---------------------------------------------------------------------------

def generate_chat_completion(
    messages: List[Dict[str, str]],
    host: str,
    port: int,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    r"""Call the Chat Completion endpoint and return the generated content.
    Args:
        messages: List of messages for the chat model.
        host: Hostname of the vLLM server.
        port: Port of the vLLM server.
        model: Model name to use for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        timeout: Timeout for the HTTP request in seconds.
    Returns:
        The generated content from the model.
    
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    r"""Remove Markdown code fences, if present.
    Args:
        text: Input text that may contain Markdown code fences.
    Returns:
        The text without code fences.
    """
    
    match = re.search(r"```(?:\w+)?\s*?\n(.*?)```", text, re.S)
    return match.group(1).rstrip() if match else text


def extract_indented_body(code: str) -> str:
    r"""Extract the body of a Python function or Java method.
    Args:
        code: The input code snippet.
    Returns:
        The extracted body of the function or method.
    """
    lines = code.splitlines()

    # Python
    if any(line.startswith("def ") for line in lines):
        body_started = False
        body_lines: List[str] = []
        for line in lines:
            if not body_started:
                if line.startswith("def "):
                    body_started = True
                continue
            body_lines.append(line)
        return "\n".join(body_lines).rstrip()

    # Java
    if any(line.strip().startswith(("public ", "private ", "protected ")) for line in lines):
        start_idx = next((i + 1 for i, ln in enumerate(lines) if "{" in ln), None)
        end_idx = next((i for i in range(len(lines) - 1, -1, -1) if "}" in lines[i]), None)
        if start_idx is not None and end_idx is not None and start_idx < end_idx:
            return "\n".join(lines[start_idx:end_idx]).rstrip()

    # Fallback: return indented lines
    return "\n".join(line for line in lines if line.startswith(("    ", "\t"))).rstrip()


def postprocess_body(task_id: str, code: str) -> str:
    r"""Normalize the extracted body and restore Java braces if needed.
    Args:
        task_id: The task ID indicating the programming language.
        code: The extracted code snippet.
    Returns:
        The normalized code snippet.
    """
    body = extract_indented_body(code)
    if task_id.startswith("Java/"):
        body = extract_indented_body(body)
        body = body.rstrip() + "\n    }\n}"
    return body


# ---------------------------------------------------------------------------
# Writer Process
# ---------------------------------------------------------------------------

def writer(queue: Queue, out_path: str, total_samples: int):
    r"""Write results to disk while displaying a global progress bar.
    Args:
        queue: Queue containing the results to write.
        out_path: Output file path.
        total_samples: Total number of samples to process.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, ext = os.path.splitext(out_path)
    f_py = open(f"{base}_python{ext}", "w", encoding="utf-8")
    f_ja = open(f"{base}_java{ext}", "w", encoding="utf-8")

    bar = tqdm(total=total_samples, desc="Overall", unit="task")
    while True:
        item = queue.get()
        if item is None:
            break
        line = json.dumps(item, ensure_ascii=False) + "\n"
        task_id = item.get("task_id", "").lower()
        if task_id.startswith("python/"):
            f_py.write(line)
            f_py.flush()
        elif task_id.startswith("java/"):
            f_ja.write(line)
            f_ja.flush()
        bar.update(1)
    bar.close()
    f_py.close()
    f_ja.close()


# ---------------------------------------------------------------------------
# Planning, Generation, Self-Debugging
# ---------------------------------------------------------------------------

def _plan_methods(
    item: Dict[str, str],
    port: int,
    model: str,
    args,
    template: Dict[str, str],
) -> List[str]:
    r"""Stage 1 – generate distinct solution outlines.
    Args:
        item: The task item containing the prompt.
        port: Port of the vLLM server.
        model: Model name to use for generation.
        args: Command line arguments.
        template: Language template for the task.
    Returns:
        A list of distinct solution approaches.
    
    """
    system_prompt = (
        f"You are an expert {template['name']} algorithm designer. "
        f"List exactly {args.pass_k} solution approaches as bullet points. "
        "Each approach must use a different core algorithmic paradigm or data structure "
        "and be expressed in a single concise sentence (no numbering, no method names)."
    )

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    placeholders = "\n".join(
        f"- Brief description of the {ordinal(i + 1)} approach" for i in range(args.pass_k)
    )

    user_prompt = (
        f"{item['prompt']}\n\n"
        f"List exactly {args.pass_k} distinct approaches:\n"
        f"{placeholders}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = generate_chat_completion(
        messages,
        "localhost",
        port,
        model,
        max_tokens=512,
        temperature=0.7,
        timeout=args.timeout,
    )

    methods = [m.strip() for m in re.findall(r"^\s*[-•*]\s+(.*)$", response, re.MULTILINE) if m.strip()]
    if len(methods) < args.pass_k:
        safe_print(
            f"[Warning] Model returned {len(methods)} methods, expected {args.pass_k}. Padding with defaults."
        )
        methods += [f"Default description {len(methods) + i + 1}" for i in range(args.pass_k - len(methods))]
    return methods[:args.pass_k]


def _generate_code(item, algo, template, port, model, args):
    r"""Single code generation attempt.
    Args:
        item: The task item containing the prompt.
        algo: The selected algorithmic approach.
        template: Language template for the task.
        port: Port of the vLLM server.
        model: Model name to use for generation.
        args: Command line arguments.
    Returns:
        The generated code, the algorithm used, and any error message.
    
    """
    try:
        system_prompt = (
            f"You are a focused {template['name']} code generator. "
            f"Respond only with {template['requirements']}"
        )
        user_prompt = (
            f"{item['prompt']}\n\n"
            f"Please implement using ONLY the following algorithmic approach: {algo}.\n"
            f"Output ONLY the {template['target']}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = generate_chat_completion(
            messages,
            "localhost",
            port,
            model,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
        return strip_code_fence(resp), algo, None
    except Exception as exc:
        return "", algo, str(exc)


def _self_debug(
    code: str,
    algo: str,
    template: Dict[str, str],
    port: int,
    model: str,
    args,
    test_result: str = "",
) -> tuple[str, str, str | None]:
    r"""Stage 3 – let the LLM debug and fix the code.
    Args:
        code: The generated code to debug.
        algo: The algorithm used for the code.
        template: Language template for the task.
        port: Port of the vLLM server.
        model: Model name to use for generation.
        args: Command line arguments.
        test_result: Test result string (optional).
    Returns:
        A tuple containing the corrected code, the algorithm used, and any error message.
    
    """
    try:
        system_prompt = (
            f"You are an expert {template['name']} code debugging assistant. "
            f"Review the provided implementation for the given approach. "
            f"If bugs or syntax errors exist, correct them. "
            f"Respond ONLY with the complete, corrected {template['target']} code, "
            f"preserving the original formatting as much as possible. "
            f"Do not add explanations, comments, or conversational text."
        )

        user_parts = [
            f"Approach: {algo}",
            "Implementation to review and correct:",
            "```",
            code.rstrip(),
            "```",
        ]
        if test_result:
            user_parts.extend(["Test result:", test_result])
        user_parts.append(f"Output ONLY the corrected {template['target']} code below:")
        user_prompt = "\n".join(user_parts)

        resp = generate_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "localhost",
            port,
            model,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
        return resp, algo, None
    except Exception as exc:
        return code, algo, str(exc)


# ---------------------------------------------------------------------------
# Per‑task Pipeline
# ---------------------------------------------------------------------------

def solve_one_task(item: Dict[str, str], port: int, model: str, args) -> List[Dict]:
    r"""Process a single HumanEval‑X task and return results ready for pass@k evaluation.
    Args:
        item: The task item containing the prompt.
        port: Port of the vLLM server.
        model: Model name to use for generation.
        args: Command line arguments.   
    Returns:
        A list of dictionaries containing the task ID, prompt, algorithm, generated code, and any error message.
    
    """
    lang = item["id"].split("/")[0].lower()
    template = LANG_TEMPLATES.get(
        lang,
        {
            "name": lang.capitalize(),
            "target": "code snippet",
            "requirements": "valid code only—no extras.",
        },
    )

    # Planning
    try:
        algos = _plan_methods(item, port, model, args, template)
    except Exception as exc:
        safe_print(f"[Error] Planning failed for task {item.get('id', 'N/A')}: {exc}")
        return [
            {
                "task_id": item.get("id", "N/A"),
                "prompt": item["prompt"],
                "algorithm": "PLANNING_FAILED",
                "generation": "",
                "error": str(exc),
            }
        ]

    results: List[Dict] = []
    errors: List[str] = []

    gen_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.gen_workers)
    fix_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.fix_workers)

    pending_gen: Dict[concurrent.futures.Future, None] = {}
    pending_fix: Dict[concurrent.futures.Future, None] = {}

    for algo in algos:
        for _ in range(args.repeat):
            fut = gen_pool.submit(_generate_code, item, algo, template, port, model, args)
            pending_gen[fut] = None

    while pending_gen or pending_fix:
        done, _ = wait(pending_gen | pending_fix, return_when=FIRST_COMPLETED)
        for fut in done:
            if fut in pending_gen:
                pending_gen.pop(fut)
                raw_code, algo, err = fut.result()
                if err:
                    errors.append(err)
                    continue
                fix_fut = fix_pool.submit(
                    _self_debug,
                    raw_code,
                    algo,
                    template,
                    port,
                    model,
                    args,
                    test_result="",
                )
                pending_fix[fix_fut] = None
            elif fut in pending_fix:
                pending_fix.pop(fut)
                resp, algo, err = fut.result()
                body_code = postprocess_body(item["id"], resp)
                results.append(
                    {
                        "task_id": item["id"],
                        "prompt": item["prompt"],
                        "algorithm": algo,
                        "generation": body_code,
                        "error": err,
                    }
                )

    gen_pool.shutdown(wait=True)
    fix_pool.shutdown(wait=True)

    if errors:
        safe_print(f"[Info] Task {item.get('id', 'N/A')} encountered {len(errors)} errors: {errors}")

    return results


# ---------------------------------------------------------------------------
# Dynamic Worker
# ---------------------------------------------------------------------------

def dynamic_worker(
    port: int,
    model: str,
    queue_in,
    queue_out: Queue,
    args,
    concurrency: int,
):
    r"""Worker that dynamically schedules generation and debugging jobs.
    Args:
        port: Port of the vLLM server.
        model: Model name to use for generation.
        queue_in: Input queue containing tasks to process.
        queue_out: Output queue for results.
        args: Command line arguments.
        concurrency: Number of concurrent requests per instance.
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

    for _ in range(concurrency):
        submit_one()

    while futures:
        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            futures.pop(fut)
            try:
                results = fut.result()
                for result in results:
                    queue_out.put(result)
                submit_one()
            except Exception as exc:
                safe_print(f"[Error] Exception while processing request: {exc}")

    executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HumanEval‑X vLLM chat inference script (Docker, dynamic concurrency)."
    )
    parser.add_argument(
        "--input",
        help="Custom input file (JSON/JSONL). If omitted, the dataset will be downloaded based on --langs.",
    )
    parser.add_argument("--download_url", help="Download URL used when --input is missing.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output prefix; creates <prefix>_python.jsonl and <prefix>_java.jsonl.",
    )

    parser.add_argument("--langs", default="python,java", help="Comma‑separated language list, e.g., python,java.")
    parser.add_argument("--cache_dir", default="humaneval_cache", help="Dataset cache directory.")
    parser.add_argument("--cpu-cores", type=int, default=16, help="Total host CPU cores to allocate to containers.")
    parser.add_argument("--instances", type=int, default=4, help="Number of Docker instances to launch.")
    parser.add_argument("--base-port", type=int, default=8010, help="Port of the first instance; subsequent instances increment by 1.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent requests per instance.")
    parser.add_argument("--pass-k", type=int, default=1, help="pass@k (default: 1).")
    parser.add_argument("--repeat", type=int, default=1, help="Number of generations per algorithm (default: 1).")
    parser.add_argument("--model", default="qwen2", help="Must match the vLLM --served-model-name.")
    parser.add_argument("--max_tokens", type=int, default=512, help="max_tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature for generation.")
    parser.add_argument("--timeout", type=int, default=2000, help="HTTP request timeout (seconds).")
    parser.add_argument("--gen_workers", type=int, default=4, help="Number of threads for code generation.")
    parser.add_argument("--fix_workers", type=int, default=4, help="Number of threads for self-debugging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output files instead of overwriting.",
    )
    args = parser.parse_args()

    ports = start_containers(
        IMAGE,
        HOST_MODEL_PATH,
        CONTAINER_MODEL_PATH,
        args.cpu_cores,
        args.instances,
        args.base_port,
    )

    try:
        samples_to_process, total_task = get_samples(
            input_path=args.input,
            download_url=args.download_url,
            langs_str=args.langs,
            cache_dir=args.cache_dir,
            output_prefix=args.output,
            resume=args.resume,
            pass_k=args.pass_k,
            repeat=args.repeat,
        )

        if not samples_to_process:
            safe_print("[*] No tasks to process; exiting.")
            return

        manager = Manager()
        q_in = manager.Queue()
        for sample in samples_to_process:
            q_in.put(sample)

        q_out = Queue()
        writer_proc = Process(target=writer, args=(q_out, args.output, total_task))
        writer_proc.start()

        workers: List[Process] = []
        for port in ports:
            proc = Process(
                target=dynamic_worker,
                args=(port, args.model, q_in, q_out, args, args.concurrency),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        q_out.put(None)
        writer_proc.join()
        safe_print("[*] All tasks processed and written to disk.")

    finally:
        safe_print("[*] Cleaning up Docker containers…")
        stop_containers(args.instances, args.base_port)


if __name__ == "__main__":
    main()
