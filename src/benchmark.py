import asyncio, aiohttp, argparse, time, json, sys, tiktoken
from typing import List, Tuple
from tqdm import tqdm

ENC = tiktoken.get_encoding("cl100k_base")

def build_payload(max_tokens: int = 128):
    prompt = "Implement a Python function that reverses a linked list."
    return {
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True
    }

async def send_one(session, url, payload, stats: List):
    start = time.perf_counter()
    output_tokens = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                stats.append(("http_error", 0, 0, 0, 0))
                return
            ttft = None
            async for line in resp.content:
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data.strip() == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except:
                        continue
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            output_tokens += 1
            total = time.perf_counter() - start
            stats.append(("ok", ttft or 0, total, output_tokens, 1))
    except Exception as e:
        stats.append(("exception", 0, 0, 0, 0))
        print(e, file=sys.stderr)

async def main(args):
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = build_payload(args.max_tokens)

    stats: List[Tuple[str, float, float, int, int]] = []

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=args.concurrent)
    ) as session:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(send_one(session, url, payload, stats))
            for _ in range(args.requests)
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=args.requests):
            await f
        total_wall = time.perf_counter() - t0

    ok = [s for s in stats if s[0] == "ok"]
    total_out_tokens = sum(s[3] for s in ok)
    total_success = len(ok)

    if total_success == 0:
        print("All requests failed.")
        return

    # 指标
    qps = total_success / total_wall
    token_qps = total_out_tokens / total_wall
    avg_ttft = sum(s[1] for s in ok) / total_success
    avg_tpot = (sum(s[2] for s in ok) - sum(s[1] for s in ok)) / total_out_tokens

    print("="*60)
    print(f"并发数        : {args.concurrent}")
    print(f"请求数        : {args.requests}")
    print(f"成功请求      : {total_success}")
    print(f"总输出 tokens : {total_out_tokens}")
    print(f"QPS (req/s)   : {qps:.2f}")
    print(f"Token QPS     : {token_qps:.2f} tokens/s")
    print(f"平均 TTFT     : {avg_ttft*1000:.1f} ms")
    print(f"平均 TPOT     : {avg_tpot*1000:.1f} ms")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38468)
    parser.add_argument("--concurrent", type=int, default=512)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    asyncio.run(main(args))