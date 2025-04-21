#!/usr/bin/env python3
# filename: data_utils.py

"""Utilities for downloading HumanEval‑X data, caching per‑language subsets, and
preparing sample lists for inference runs with optional checkpoint resumption."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import collections
import re

import requests
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:  # Lazy handling when the user does not need automatic download
    load_dataset = None  # type: ignore

from .safe_print import safe_print

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str) -> None:
    """Download *url* to *dest* with a progress bar."""
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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cached_lang(cache_path: str) -> List[Dict]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cached_lang(cache_path: str, data: List[Dict]) -> None:
    r"""Save the *data* to *cache_path* in JSON format.
    Args:
        cache_path: Path to save the cached data.
        data: List of dictionaries containing the task ID and prompt.  
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def download_and_cache_lang(lang: str, cache_path: str) -> List[Dict]:
    r"""Download the *lang* split of HumanEval‑X via `datasets` and cache to *cache_path*.
    Args:
        lang: Language to download (e.g., "Python", "Java").
        cache_path: Path to save the cached data.
    Returns:
        A list of dictionaries containing the task ID and prompt.
    """
    if load_dataset is None:
        sys.exit("The `datasets` package is required; run `pip install datasets`.")
    safe_print(f"[Data] Downloading HumanEval‑X – {lang} …")
    ds = load_dataset("THUDM/humaneval-x", lang, split="test")
    rows = [{"id": r["task_id"], "prompt": r["prompt"]} for r in ds]
    _save_cached_lang(cache_path, rows)
    safe_print(f"[Data] Cached {lang} to {cache_path}.")
    return rows


# ---------------------------------------------------------------------------
# Sample preparation
# ---------------------------------------------------------------------------

def get_samples(
    input_path: str | None,
    download_url: str | None,
    langs_str: str,
    cache_dir: str,
    output_prefix: str,
    resume: bool,
    pass_k: int,
    repeat: int,
) -> Tuple[List[Dict], int]:
    r"""Return the list of samples to process and the expected total generations.

    If *resume* is True, existing output files are scanned and already‑completed
    tasks are skipped; partial tasks are re‑processed until they reach
    `pass_k * repeat` completions.
    Args:
        input_path: Path to the input file (JSON or JSONL) containing initial samples.
        download_url: URL to download the initial sample list if *input_path* is not found.
        langs_str: Comma-separated list of languages to download from HumanEval‑X.
        cache_dir: Directory for caching downloaded datasets.
        output_prefix: Prefix for output files (Python and Java).
        resume: Whether to resume from existing output files.
        pass_k: Number of completions required for each task.
        repeat: Number of times to repeat each task.
    Returns:
        A tuple containing:
            - List of samples to process.
            - Total expected generations for this run.
    """

    # ------------------------------------------------------------------
    # 1. Load or download the initial sample list
    # ------------------------------------------------------------------
    initial_samples: List[Dict] = []

    if input_path:
        if not os.path.isfile(input_path):
            if download_url:
                safe_print(f"[Data] {input_path} missing – downloading…")
                download_file(download_url, input_path)
            else:
                sys.exit(f"Input file {input_path} not found and no download URL provided.")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":  # JSON array
                    initial_samples = json.load(f)
                else:  # JSONL
                    initial_samples = [json.loads(line) for line in f if line.strip()]
            safe_print(f"[Data] Loaded {len(initial_samples)} samples from {input_path}.")
        except Exception as exc:
            sys.exit(f"Failed to read or parse {input_path}: {exc}")
    else:
        langs = [l.strip() for l in langs_str.split(",") if l.strip()]
        for lang in langs:
            cache_path = os.path.join(cache_dir, f"humaneval-{lang}.json")
            if os.path.isfile(cache_path):
                safe_print(f"[Data] Loading cache {cache_path}…")
                try:
                    rows = _load_cached_lang(cache_path)
                except Exception as exc:
                    safe_print(f"[Warn] Failed to load cache {cache_path}: {exc}; redownloading.")
                    rows = download_and_cache_lang(lang, cache_path)
            else:
                rows = download_and_cache_lang(lang, cache_path)
            initial_samples.extend(rows)
        safe_print(f"[Data] Collected {len(initial_samples)} total tasks.")

    if not initial_samples:
        sys.exit("No samples available – check your parameters.")

    # ------------------------------------------------------------------
    # 2. Handle resume logic
    # ------------------------------------------------------------------
    target_completions_per_task = pass_k * repeat
    samples_to_process: List[Dict] = []
    total_task: int = 0

    base, ext = os.path.splitext(output_prefix)
    py_file = f"{base}_python{ext}"
    ja_file = f"{base}_java{ext}"
    out_files = [py_file, ja_file]

    if resume:
        safe_print("[*] Resume mode enabled – scanning existing output files…")
        completed_counts = collections.Counter()
        lines_by_task_id: Dict[str, List[str]] = collections.defaultdict(list)

        for path in out_files:
            if os.path.exists(path):
                safe_print(f"[*] Reading {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                task_id = data.get("task_id")
                                if task_id:
                                    completed_counts[task_id] += 1
                                    lines_by_task_id[task_id].append(line.rstrip())
                            except json.JSONDecodeError:
                                safe_print(f"[Warn] Invalid JSON skipped in {path}: {line.rstrip()}")
                except Exception as exc:
                    safe_print(f"[Error] Failed to read {path}: {exc}")

        fully_completed = {
            tid for tid, cnt in completed_counts.items() if cnt >= target_completions_per_task
        }
        safe_print(f"[*] {len(fully_completed)} tasks fully completed.")

        # Rewrite output files with only the fully‑completed tasks to keep them clean.
        try:
            with open(py_file, "w", encoding="utf-8") as f_py, open(ja_file, "w", encoding="utf-8") as f_ja:
                py_kept = ja_kept = 0
                for tid in fully_completed:
                    if tid in lines_by_task_id:
                        joined = "\n".join(lines_by_task_id[tid]) + "\n"
                        if tid.startswith("Python/"):
                            f_py.write(joined)
                            py_kept += len(lines_by_task_id[tid])
                        elif tid.startswith("Java/"):
                            f_ja.write(joined)
                            ja_kept += len(lines_by_task_id[tid])
                safe_print(f"[*] Rewrote Python file – kept {py_kept} lines.")
                safe_print(f"[*] Rewrote Java file   – kept {ja_kept} lines.")
        except Exception as exc:
            safe_print(f"[Error] Failed to rewrite output files: {exc}")

        for sample in initial_samples:
            tid = sample.get("id")
            if tid and completed_counts.get(tid, 0) < target_completions_per_task:
                samples_to_process.append(sample)

        if samples_to_process:
            safe_print(f"[*] Original tasks: {len(initial_samples)}")
            safe_print(f"[*] Pending tasks:  {len(samples_to_process)}")
            total_task = len(samples_to_process) * target_completions_per_task
        else:
            safe_print("[*] All tasks already complete; nothing to do.")
            total_task = 0
    else:
        safe_print("[*] Fresh run – previous output will be overwritten if found.")
        samples_to_process = initial_samples
        for path in out_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    safe_print(f"[*] Removed old output {path}")
                except OSError as exc:
                    safe_print(f"[Warn] Could not remove {path}: {exc}")
        total_task = len(samples_to_process) * target_completions_per_task if samples_to_process else 0

    safe_print(f"[*] Required completions per task: {target_completions_per_task}")
    safe_print(f"[*] Expected total generations this run: {total_task}")

    return samples_to_process, total_task
