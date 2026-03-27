#!/usr/bin/env python3
"""vLLM Baseline Benchmark Script.

Measures TTFT, TPOT (ITL), TPS, E2E latency across various
concurrency levels and input/output length combinations.
Results are saved as JSON to results/baseline/.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

VLLM_BASE_URL = "http://localhost:30800/v1"
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

PROMPTS = [
    "Explain the concept of PagedAttention in LLM serving and why it matters for memory efficiency.",
    "What are the key differences between prefill and decode phases in transformer inference?",
    "Describe how KV-cache works in autoregressive language models.",
    "Write a Python function that implements binary search on a sorted array.",
    "Explain the memory bandwidth bottleneck in LLM inference and how batching helps.",
    "What is continuous batching and how does it improve GPU utilization?",
    "Describe the architecture of the NVIDIA Blackwell GPU and its key innovations.",
    "How does quantization (AWQ, GPTQ) reduce memory usage while maintaining model quality?",
    "Explain tensor parallelism vs pipeline parallelism for distributed LLM serving.",
    "What are the tradeoffs between latency and throughput in LLM serving systems?",
]


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    itl_ms: list[float] = field(default_factory=list)
    e2e_latency_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkResult:
    timestamp: str = ""
    model: str = MODEL_NAME
    concurrency: int = 1
    max_tokens: int = 128
    num_requests: int = 10
    total_duration_s: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    # TTFT
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    ttft_mean_ms: float = 0.0
    # ITL (Inter-Token Latency) = TPOT
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    itl_mean_ms: float = 0.0
    # E2E Latency
    e2e_p50_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    e2e_p99_ms: float = 0.0
    e2e_mean_ms: float = 0.0
    # Throughput
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    requests_per_second: float = 0.0


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def send_streaming_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int,
    base_url: str = VLLM_BASE_URL,
    model_name: str = MODEL_NAME,
) -> RequestResult:
    result = RequestResult()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    token_times: list[float] = []
    start = time.perf_counter()

    try:
        async with session.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                result.success = False
                result.error = f"HTTP {resp.status}"
                return result

            first_token_received = False
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                if content and not first_token_received:
                    result.ttft_ms = (time.perf_counter() - start) * 1000
                    first_token_received = True
                    token_times.append(time.perf_counter())
                elif content:
                    token_times.append(time.perf_counter())

                usage = data.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.completion_tokens = usage.get("completion_tokens", 0)

    except Exception as e:
        result.success = False
        result.error = str(e)
        return result

    result.e2e_latency_ms = (time.perf_counter() - start) * 1000

    # Calculate ITL (inter-token latency)
    if len(token_times) > 1:
        result.itl_ms = [
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        ]

    # Estimate tokens from chunk count if usage not provided
    if result.completion_tokens == 0 and token_times:
        result.completion_tokens = len(token_times)

    return result


async def run_benchmark(
    concurrency: int,
    max_tokens: int,
    num_requests: int,
    base_url: str = VLLM_BASE_URL,
    model_name: str = MODEL_NAME,
) -> BenchmarkResult:
    bench = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model_name,
        concurrency=concurrency,
        max_tokens=max_tokens,
        num_requests=num_requests,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt: str) -> RequestResult:
        async with semaphore:
            return await send_streaming_request(session, prompt, max_tokens, base_url, model_name)

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup: 2 requests
        print(f"  Warming up...")
        warmup_tasks = [
            send_streaming_request(session, PROMPTS[0], 16, base_url, model_name),
            send_streaming_request(session, PROMPTS[1], 16, base_url, model_name),
        ]
        await asyncio.gather(*warmup_tasks)

        # Benchmark
        print(f"  Running {num_requests} requests (concurrency={concurrency}, max_tokens={max_tokens})...")
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

        start = time.perf_counter()
        tasks = [bounded_request(p) for p in prompts]
        results: list[RequestResult] = await asyncio.gather(*tasks)
        bench.total_duration_s = time.perf_counter() - start

    # Aggregate
    successes = [r for r in results if r.success]
    bench.successful_requests = len(successes)
    bench.failed_requests = len(results) - len(successes)

    if not successes:
        print("  ❌ All requests failed!")
        return bench

    ttfts = [r.ttft_ms for r in successes]
    e2es = [r.e2e_latency_ms for r in successes]
    all_itls: list[float] = []
    for r in successes:
        all_itls.extend(r.itl_ms)

    bench.ttft_mean_ms = statistics.mean(ttfts)
    bench.ttft_p50_ms = percentile(ttfts, 50)
    bench.ttft_p95_ms = percentile(ttfts, 95)
    bench.ttft_p99_ms = percentile(ttfts, 99)

    if all_itls:
        bench.itl_mean_ms = statistics.mean(all_itls)
        bench.itl_p50_ms = percentile(all_itls, 50)
        bench.itl_p95_ms = percentile(all_itls, 95)
        bench.itl_p99_ms = percentile(all_itls, 99)

    bench.e2e_mean_ms = statistics.mean(e2es)
    bench.e2e_p50_ms = percentile(e2es, 50)
    bench.e2e_p95_ms = percentile(e2es, 95)
    bench.e2e_p99_ms = percentile(e2es, 99)

    bench.total_prompt_tokens = sum(r.prompt_tokens for r in successes)
    bench.total_completion_tokens = sum(r.completion_tokens for r in successes)
    bench.prompt_tps = bench.total_prompt_tokens / bench.total_duration_s
    bench.generation_tps = bench.total_completion_tokens / bench.total_duration_s
    bench.requests_per_second = bench.successful_requests / bench.total_duration_s

    return bench


def print_result(bench: BenchmarkResult) -> None:
    print(f"\n{'='*60}")
    print(f"  Concurrency: {bench.concurrency} | Max Tokens: {bench.max_tokens}")
    print(f"  Requests: {bench.successful_requests}/{bench.num_requests} succeeded")
    print(f"  Duration: {bench.total_duration_s:.2f}s")
    print(f"{'='*60}")
    print(f"  TTFT    - p50: {bench.ttft_p50_ms:>8.1f}ms  p95: {bench.ttft_p95_ms:>8.1f}ms  p99: {bench.ttft_p99_ms:>8.1f}ms  mean: {bench.ttft_mean_ms:>8.1f}ms")
    print(f"  ITL     - p50: {bench.itl_p50_ms:>8.1f}ms  p95: {bench.itl_p95_ms:>8.1f}ms  p99: {bench.itl_p99_ms:>8.1f}ms  mean: {bench.itl_mean_ms:>8.1f}ms")
    print(f"  E2E     - p50: {bench.e2e_p50_ms:>8.1f}ms  p95: {bench.e2e_p95_ms:>8.1f}ms  p99: {bench.e2e_p99_ms:>8.1f}ms  mean: {bench.e2e_mean_ms:>8.1f}ms")
    print(f"  Throughput - Prefill: {bench.prompt_tps:>8.1f} tok/s  Decode: {bench.generation_tps:>8.1f} tok/s  RPS: {bench.requests_per_second:.2f}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM Baseline Benchmark")
    parser.add_argument("--url", default=VLLM_BASE_URL, help="API base URL")
    parser.add_argument("--output-dir", default="results/baseline", help="Output directory")
    parser.add_argument("--tag", default="baseline", help="Tag for the result file")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8], help="Concurrency levels")
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[128, 256], help="Max tokens per request")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests per config")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    args = parser.parse_args()

    base_url = args.url
    model_name = args.model

    # Check connectivity
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                models = await resp.json()
                print(f"Connected to {base_url}. Models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        print(f"Cannot connect to {base_url}: {e}")
        return

    all_results: list[dict] = []

    for max_tokens in args.max_tokens:
        for concurrency in args.concurrency:
            print(f"\nBenchmark: concurrency={concurrency}, max_tokens={max_tokens}")
            bench = await run_benchmark(concurrency, max_tokens, args.num_requests, base_url, model_name)
            print_result(bench)
            all_results.append(asdict(bench))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{args.tag}_{ts}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results saved to: {output_file}")

    # Print summary comparison table
    print(f"\n{'='*80}")
    print(f"{'Concurrency':>12} {'MaxTok':>8} {'TTFT p50':>10} {'ITL p50':>10} {'E2E p50':>10} {'Gen TPS':>10} {'RPS':>8}")
    print(f"{'='*80}")
    for r in all_results:
        print(f"{r['concurrency']:>12} {r['max_tokens']:>8} {r['ttft_p50_ms']:>9.1f}ms {r['itl_p50_ms']:>9.1f}ms {r['e2e_p50_ms']:>9.1f}ms {r['generation_tps']:>9.1f} {r['requests_per_second']:>7.2f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
