#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py  ·  自动缓存 HumanEval‑X 数据集 + 多进程推理 + tqdm 进度条
             方案 B：写入进程 + Queue；各子进程独立 tqdm(position) 显示
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict
from multiprocessing import Process, Queue

import requests
from tqdm.auto import tqdm

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# --------------------------- 打印工具 --------------------------- #
def safe_print(*args, **kwargs):
    enc = (sys.stdout.encoding or "").lower()
    if enc.startswith("utf"):
        print(*args, **kwargs)
    else:
        text = " ".join(map(str, args))
        print(text.encode(enc or "gbk", errors="replace").decode(enc or "gbk"), **kwargs)

# --------------------------- 网络下载工具 --------------------------- #
def download_file(url: str, dest: str) -> None:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    bar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    bar.close()

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
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# --------------------------- 样本缓存获取 --------------------------- #
def _load_cached_lang(cache_path: str) -> List[Dict]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_cached_lang(cache_path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def download_and_cache_lang(lang: str, cache_path: str) -> List[Dict]:
    if load_dataset is None:
        sys.exit("缺少 datasets 库，请先 `pip install datasets`")
    safe_print(f"[Info] 正在下载 HumanEval‑X:{lang} …")
    ds = load_dataset("THUDM/humaneval-x", lang, split="test")
    rows = [{"id": r["task_id"], "prompt": r["prompt"]} for r in ds]
    _save_cached_lang(cache_path, rows)
    safe_print(f"[Info] {lang} 下载完成，已缓存至 {cache_path}")
    return rows

def get_samples(args) -> List[Dict]:
    if args.input:
        if not os.path.isfile(args.input):
            if args.download_url:
                safe_print(f"[Info] {args.input} 不存在，尝试下载…")
                download_file(args.download_url, args.input)
            else:
                sys.exit(f"错误：找不到输入文件 {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            return json.load(f)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    samples: List[Dict] = []
    for lang in langs:
        cache_path = os.path.join(args.cache_dir, f"humaneval-{lang}.json")
        if os.path.isfile(cache_path):
            safe_print(f"[Info] 发现缓存 {cache_path}，直接加载。")
            rows = _load_cached_lang(cache_path)
        else:
            rows = download_and_cache_lang(lang, cache_path)
        samples.extend(rows)

    if not samples:
        sys.exit("未能获取任何样本，请检查参数。")
    safe_print(f"[Info] 共加载 {len(samples)} 道题目。")
    return samples


# --------------------------- 写入进程（区分 Python / Java） --------------------------- #
def writer(queue: Queue, out_path: str):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 根据 out_path 拆分出文件名和后缀
    base, ext = os.path.splitext(out_path)
    # 例如: out_path="results.jsonl" → base="results", ext=".jsonl"
    python_path = f"{base}_python{ext}"
    java_path   = f"{base}_java{ext}"

    # 打开两个文件句柄
    f_py = open(python_path, "w", encoding="utf-8")
    f_ja = open(java_path,   "w", encoding="utf-8")

    while True:
        item = queue.get()
        if item is None:    # 结束信号
            break

        # 根据 task_id 首部判断语言
        task_id = item.get("task_id", "")
        lang = task_id.split("/")[0].lower()

        line = json.dumps(item, ensure_ascii=False) + "\n"
        if lang == "python":
            f_py.write(line)
            f_py.flush()
        elif lang == "java":
            f_ja.write(line)
            f_ja.flush()
        else:
            # 如果还有其它语言，也可以按需追加到它们各自的文件
            pass
    
    # 关闭文件句柄
    f_py.close()
    f_ja.close()

# --------------------------- 推理进程 --------------------------- #
def strip_code_fence(text: str) -> str:
    """
    如果 text 中包含 ```python … ```，提取中间纯代码，否则原样返回。
    """
    m = re.search(r"```(?:python)?\s*?\n(.*?)```", text, re.S)
    if m:
        return m.group(1).rstrip()
    return text

def extract_indented_body(code: str) -> str:
    """
    如果代码中包含以 `def ` 开头的函数定义行，则丢掉该行及其以上所有内容，
    保留其下方所有行（通常都是缩进的函数体）。
    如果没有找到任何 `def ` 行，则提取所有以四个空格或一个制表符开头的行，作为函数体。
    """
    lines = code.splitlines()
    # 先检查是否存在 def 行
    has_def = any(ln.startswith("def ") for ln in lines)

    body_lines = []
    if has_def:
        saw_def = False
        for ln in lines:
            if not saw_def:
                if ln.startswith("def "):
                    saw_def = True
                continue
            body_lines.append(ln)
    else:
        # 无 def 行时提取所有缩进行
        for ln in lines:
            if ln.startswith("    ") or ln.startswith("\t"):
                body_lines.append(ln)
    return "\n".join(body_lines).rstrip()
def worker(port: int, samples: List[Dict], args, queue: Queue, position: int):
    bar = tqdm(samples, desc=f"Inf@{port}", unit="sample", position=position)
    for item in bar:
        lang = item["id"].split("/")[0].lower()
        if lang == "python":    
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a focused Python code generator. "
                        "When given a function signature and docstring, "
                        "respond _only_ with valid Python code—no explanations, no markdown fences, "
                        "no comments, no whitespace beyond code, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        item["prompt"]
                        + "\n\n"
                        + "Please output only the Python function implementation, nothing else."
                    )
                }
            ]
        elif lang == "java":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a focused Java code generator. "
                        "When given a function signature and docstring, "
                        "respond _only_ with valid Java code—no explanations, no markdown fences, "
                        "no comments, no whitespace beyond code, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        item["prompt"]
                        + "\n\n"
                        + "Please output only the Java function implementation, nothing else."
                    )
                }
            ]
        try:
            reply = generate_chat_completion(
                messages,
                host=args.host,
                port=port,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout
            )
            error = None
        except Exception as e:
            reply, error = "", str(e)

        cleaned = strip_code_fence(reply)
        body = extract_indented_body(cleaned)
        
        queue.put({
            "task_id": item["id"],
            "prompt":    item["prompt"],
            "generation": body,
            "error": error
        })
        bar.set_postfix(err=("✓" if error is None else "ERR"))
    bar.close()

# --------------------------- 主流程 --------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="HumanEval‑X vLLM Chat 推理脚本 (多进程写入同文件, tqdm position)"
    )
    p.add_argument("--input", help="自定义 JSON 输入文件（跳过缓存/下载）")
    p.add_argument("--download_url", help="input 不存在时备用下载 URL")
    p.add_argument("--output", required=True,
                   help="最终结果文件路径")
    p.add_argument("--langs", default="python,java",
                   help="指定语言，逗号分隔")
    p.add_argument("--cache_dir", default="humaneval_cache",
                   help="数据集本地缓存目录")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000,
                   help="单进程模式端口")
    p.add_argument("--ports",
                   help="多进程模式端口列表，如 8001,8002")
    p.add_argument("--model", default="qwen2",
                   help="与 vLLM --served-model-name 一致")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=int, default=2000)
    args = p.parse_args()

    samples = get_samples(args)

    queue = Queue()
    writer_proc = Process(target=writer, args=(queue, args.output))
    writer_proc.start()

    if args.ports:
        ports = [int(x) for x in args.ports.split(",") if x.strip()]
    else:
        ports = [args.port]

    n = len(ports)
    chunk_size = (len(samples) + n - 1) // n
    procs = []
    for i, port in enumerate(ports):
        chunk = samples[i*chunk_size : (i+1)*chunk_size]
        p = Process(target=worker, args=(port, chunk, args, queue, i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    queue.put(None)
    writer_proc.join()

    safe_print(f"[Info] 推理结束，所有结果写入 {args.output}")

if __name__ == "__main__":
    main()
