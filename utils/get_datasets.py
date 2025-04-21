import re
import os
import sys
import json
import requests
from typing import List, Dict, Tuple
import collections 
from tqdm import tqdm
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None 

from .safe_print import safe_print


# --------------------------- 网络下载工具 --------------------------- #
def download_file(url: str, dest: str) -> None:
    r"""下载文件，显示进度条
        args:
            url: 下载链接
            dest: 保存路径
    """
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

# --- 修改 get_samples 函数 ---
def get_samples(
    input_path: str | None,
    download_url: str | None,
    langs_str: str,
    cache_dir: str,
    output_prefix: str,
    resume: bool,
    pass_k: int,
    repeat: int
) -> Tuple[List[Dict], int]:
    """
    获取样本列表，并根据 resume 参数处理现有输出文件，返回待处理样本和总任务数。

    Args:
        input_path: 自定义输入文件路径。
        download_url: 当 input_path 文件不存在时，从该 URL 下载。
        langs_str: 语言列表字符串，逗号分隔。
        cache_dir: 数据集缓存路径。
        output_prefix: 输出文件前缀。
        resume: 是否从检查点恢复。
        pass_k: pass@k 的 k 值。
        repeat: 每个算法重复次数。

    Returns:
        Tuple[List[Dict], int]: (待处理样本列表, 本次运行预计生成的总数)
    """
    initial_samples: List[Dict] = []
    if input_path:
        if not os.path.isfile(input_path):
            if download_url:
                safe_print(f"[Info] {input_path} 不存在，尝试下载…")
                download_file(download_url, input_path)
            else:
                sys.exit(f"错误：找不到输入文件 {input_path}")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                # 尝试处理 JSON 和 JSONL 格式
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[': # 可能是 JSON 列表
                    initial_samples = json.load(f)
                else: # 可能是 JSONL
                    initial_samples = [json.loads(line) for line in f if line.strip()]
            safe_print(f"[Info] 从 {input_path} 加载了 {len(initial_samples)} 个样本。")
        except Exception as e:
             sys.exit(f"错误：读取或解析输入文件 {input_path} 失败: {e}")

    else:
        langs = [l.strip() for l in langs_str.split(",") if l.strip()]
        for lang in langs:
            cache_path = os.path.join(cache_dir, f"humaneval-{lang}.json")
            if os.path.isfile(cache_path):
                safe_print(f"[Info] 发现缓存 {cache_path}，直接加载。")
                try:
                    rows = _load_cached_lang(cache_path)
                except Exception as e:
                    safe_print(f"[警告] 加载缓存文件 {cache_path} 失败: {e}，尝试重新下载...")
                    rows = download_and_cache_lang(lang, cache_path) # 出错时尝试重新下载
            else:
                rows = download_and_cache_lang(lang, cache_path)
            initial_samples.extend(rows)
        safe_print(f"[Info] 共加载 {len(initial_samples)} 道题目。")

    if not initial_samples:
        sys.exit("未能获取任何样本，请检查参数。")

    # --- 开始处理 resume 逻辑 ---
    target_completions_per_task = pass_k * repeat
    samples_to_process: List[Dict] = []
    total_task: int = 0

    base, ext = os.path.splitext(output_prefix)
    py_file = f"{base}_python{ext}"
    ja_file = f"{base}_java{ext}"
    files_to_check = [py_file, ja_file]

    if resume:
        safe_print("[*] 启用恢复模式，检查并清理现有输出文件...")
        completed_counts = collections.Counter()
        lines_by_task_id = collections.defaultdict(list)

        for fpath in files_to_check:
            if os.path.exists(fpath):
                safe_print(f"[*] 读取文件: {fpath}")
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                task_id = data.get("task_id")
                                if task_id:
                                    completed_counts[task_id] += 1
                                    lines_by_task_id[task_id].append(line.strip())
                            except json.JSONDecodeError:
                                safe_print(f"[警告] 文件 {fpath} 中发现无效的 JSON 行，已跳过: {line.strip()}")
                except Exception as e:
                     safe_print(f"[错误] 读取文件 {fpath} 时出错: {e}")

        fully_completed_task_ids = {
            task_id for task_id, count in completed_counts.items()
            if count >= target_completions_per_task
        }
        safe_print(f"[*] 发现 {len(fully_completed_task_ids)} 个任务已完全完成。")

        safe_print("[*] 重写输出文件，仅保留已完全完成任务的结果...")
        try:
            with open(py_file, "w", encoding="utf-8") as f_py, \
                 open(ja_file, "w", encoding="utf-8") as f_ja:
                kept_py_lines = 0
                kept_ja_lines = 0
                for task_id in fully_completed_task_ids:
                    if task_id in lines_by_task_id:
                        lines_to_write = lines_by_task_id[task_id]
                        output_line = "\n".join(lines_to_write) + "\n"
                        if task_id.startswith("Python/"):
                            f_py.write(output_line)
                            kept_py_lines += len(lines_to_write)
                        elif task_id.startswith("Java/"):
                            f_ja.write(output_line)
                            kept_ja_lines += len(lines_to_write)
            safe_print(f"[*] Python 文件重写完成，保留 {kept_py_lines} 行。")
            safe_print(f"[*] Java 文件重写完成，保留 {kept_ja_lines} 行。")
        except Exception as e:
            safe_print(f"[错误] 重写输出文件时出错: {e}")

        for s in initial_samples:
            task_id = s.get('id') # 使用 .get 以防万一
            if task_id and completed_counts.get(task_id, 0) < target_completions_per_task:
                samples_to_process.append(s)

        if not samples_to_process:
             safe_print("[*] 所有任务均已完成（根据现有文件），无需执行。")
             total_task = 0 
        else:
            safe_print(f"[*] 原始任务数: {len(initial_samples)}")
            safe_print(f"[*] 需要处理的任务数 (未完成或部分完成): {len(samples_to_process)}")
            total_task = len(samples_to_process) * target_completions_per_task

    else: # 非 resume 模式
        safe_print("[*] 未启用恢复模式，将处理所有任务并覆盖旧文件（如果存在）。")
        samples_to_process = initial_samples
        for fpath in files_to_check:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    safe_print(f"[*] 已删除旧输出文件: {fpath}")
                except OSError as e:
                    safe_print(f"[警告] 删除旧输出文件 {fpath} 失败: {e}")
        
        if not samples_to_process:
            safe_print("[*] 没有需要处理的任务。")
            total_task = 0
        else:
            total_task = len(samples_to_process) * target_completions_per_task

    safe_print(f"[*] 每个任务需要生成次数: {target_completions_per_task}")
    safe_print(f"[*] 本次运行预计生成总数 (用于进度条): {total_task}")

    return samples_to_process, total_task