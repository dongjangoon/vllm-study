#!/usr/bin/env python3
"""SGLang vs vLLM Comparison Benchmark Suite.

Three focused benchmarks designed to reveal architectural differences:

1. Concurrency Scaling: How throughput and latency change from 1 to 128 concurrent requests
   - Tests scheduler efficiency and batching strategy
   - SGLang: zero-overhead batch scheduler + overlap scheduling
   - vLLM: continuous batching with iteration-level scheduling

2. Prefix Cache Effectiveness: Same system prompt repeated vs unique prompts
   - Tests KV cache reuse strategy
   - SGLang: RadixAttention (tree-based automatic prefix sharing)
   - vLLM: Automatic Prefix Caching (hash-based block matching)

3. Sustained Load (Tail Latency): Steady high-concurrency traffic over many requests
   - Tests scheduling stability under pressure
   - p99/p50 ratio reveals how well the engine handles worst-case scenarios
"""

import argparse
import asyncio
import json
import random
import statistics
import string
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ---------- Prompt templates ----------

SHARED_SYSTEM_PROMPT = (
    "You are a senior software engineer specializing in distributed systems "
    "and GPU programming. Answer concisely and technically. "
    "Always include code examples when relevant. "
    "Focus on practical, production-ready solutions."
)

DIVERSE_PROMPTS = [
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
    "How does FlashAttention reduce memory usage compared to standard attention?",
    "What is speculative decoding and when should you use it?",
    "Explain the difference between FP16, BF16, and FP8 for LLM inference.",
    "How do prefix caching strategies differ between vLLM and SGLang?",
    "What is RadixAttention and how does it enable automatic KV cache reuse?",
    "Describe the CUDA memory hierarchy and its impact on kernel optimization.",
    "How does NCCL optimize collective communication for multi-GPU training?",
    "What are the key considerations when deploying LLMs on Kubernetes?",
    "Explain how structured output generation (JSON mode) works in LLM serving.",
    "What is the difference between online and offline LLM serving workloads?",
]

# Prefix cache test: same prefix, different suffix questions
PREFIX_CACHE_SUFFIXES = [
    "How does this affect TTFT in production?",
    "What metrics should I monitor?",
    "How does this scale with model size?",
    "What are the failure modes?",
    "Compare the memory overhead vs CPU overhead.",
    "How would you implement this in PyTorch?",
    "What hardware considerations matter most?",
    "How does this interact with quantization?",
    "What batch sizes work best?",
    "How does this affect tail latency?",
    "What are common misconceptions about this?",
    "How do you debug performance issues?",
    "What changed in the latest versions?",
    "How does this work with MoE models?",
    "What are the tradeoffs for real-time applications?",
    "How would you benchmark this effectively?",
    "What are the security implications?",
    "How does this interact with LoRA adapters?",
    "What is the cold start impact?",
    "How do cloud providers handle this differently?",
]

UNIQUE_PREFIXES = [
    "As a database expert, ",
    "As a frontend developer, ",
    "As a security researcher, ",
    "As a DevOps engineer, ",
    "As a data scientist, ",
    "As a mobile developer, ",
    "As a game developer, ",
    "As a ML researcher, ",
    "As a systems programmer, ",
    "As a cloud architect, ",
    "As a compiler engineer, ",
    "As a networking specialist, ",
    "As a performance engineer, ",
    "As a site reliability engineer, ",
    "As an embedded systems developer, ",
    "As a cryptography expert, ",
    "As a distributed systems engineer, ",
    "As a real-time systems developer, ",
    "As a quantum computing researcher, ",
    "As a robotics engineer, ",
]


# ---------- Data classes ----------

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
    engine: str = ""
    test_name: str = ""
    model: str = ""
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
    # ITL
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    itl_mean_ms: float = 0.0
    # E2E
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
    # Tail latency ratio
    ttft_p99_p50_ratio: float = 0.0
    itl_p99_p50_ratio: float = 0.0
    e2e_p99_p50_ratio: float = 0.0


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


# ---------- Core request function ----------

async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict],
    max_tokens: int,
    base_url: str,
    model_name: str,
) -> RequestResult:
    result = RequestResult()
    payload = {
        "model": model_name,
        "messages": messages,
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
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                result.success = False
                result.error = f"HTTP {resp.status}: {await resp.text()}"
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

    if len(token_times) > 1:
        result.itl_ms = [
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        ]

    if result.completion_tokens == 0 and token_times:
        result.completion_tokens = len(token_times)

    return result


# ---------- Benchmark runner ----------

def aggregate_results(
    results: list[RequestResult],
    engine: str,
    test_name: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    num_requests: int,
    duration: float,
) -> BenchmarkResult:
    bench = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        engine=engine,
        test_name=test_name,
        model=model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        num_requests=num_requests,
        total_duration_s=duration,
    )

    successes = [r for r in results if r.success]
    bench.successful_requests = len(successes)
    bench.failed_requests = len(results) - len(successes)

    if not successes:
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
    if duration > 0:
        bench.prompt_tps = bench.total_prompt_tokens / duration
        bench.generation_tps = bench.total_completion_tokens / duration
        bench.requests_per_second = bench.successful_requests / duration

    # Tail latency ratios
    if bench.ttft_p50_ms > 0:
        bench.ttft_p99_p50_ratio = bench.ttft_p99_ms / bench.ttft_p50_ms
    if bench.itl_p50_ms > 0:
        bench.itl_p99_p50_ratio = bench.itl_p99_ms / bench.itl_p50_ms
    if bench.e2e_p50_ms > 0:
        bench.e2e_p99_p50_ratio = bench.e2e_p99_ms / bench.e2e_p50_ms

    return bench


async def run_test(
    session: aiohttp.ClientSession,
    messages_list: list[list[dict]],
    concurrency: int,
    max_tokens: int,
    base_url: str,
    model_name: str,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(messages: list[dict]) -> RequestResult:
        async with semaphore:
            return await send_request(session, messages, max_tokens, base_url, model_name)

    start = time.perf_counter()
    tasks = [bounded_request(m) for m in messages_list]
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    return list(results), duration


def print_result(bench: BenchmarkResult) -> None:
    print(f"\n{'─'*70}")
    print(f"  [{bench.engine}] {bench.test_name}")
    print(f"  Concurrency: {bench.concurrency} | Max Tokens: {bench.max_tokens}")
    print(f"  Requests: {bench.successful_requests}/{bench.num_requests} succeeded | Duration: {bench.total_duration_s:.2f}s")
    print(f"{'─'*70}")
    print(f"  TTFT  p50: {bench.ttft_p50_ms:>7.1f}ms  p95: {bench.ttft_p95_ms:>7.1f}ms  p99: {bench.ttft_p99_ms:>7.1f}ms  (p99/p50: {bench.ttft_p99_p50_ratio:.2f}x)")
    print(f"  ITL   p50: {bench.itl_p50_ms:>7.1f}ms  p95: {bench.itl_p95_ms:>7.1f}ms  p99: {bench.itl_p99_ms:>7.1f}ms  (p99/p50: {bench.itl_p99_p50_ratio:.2f}x)")
    print(f"  E2E   p50: {bench.e2e_p50_ms:>7.1f}ms  p95: {bench.e2e_p95_ms:>7.1f}ms  p99: {bench.e2e_p99_ms:>7.1f}ms  (p99/p50: {bench.e2e_p99_p50_ratio:.2f}x)")
    print(f"  Throughput  Gen: {bench.generation_tps:>7.1f} tok/s  RPS: {bench.requests_per_second:.2f}")


# ---------- Benchmark 1: Concurrency Scaling ----------

async def bench_concurrency_scaling(
    base_url: str, model_name: str, engine: str, max_tokens: int = 128, num_requests: int = 30,
) -> list[BenchmarkResult]:
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 1: Concurrency Scaling ({engine})")
    print(f"  Goal: Measure throughput/latency curve as concurrency increases")
    print(f"  Why: Tests scheduler efficiency and batching strategy")
    print(f"{'='*70}")

    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    all_results = []

    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print("  Warming up (4 requests)...")
        warmup_msgs = [[{"role": "user", "content": "Hello"}]] * 4
        await run_test(session, warmup_msgs, 4, 16, base_url, model_name)

        for conc in concurrency_levels:
            actual_requests = max(num_requests, conc * 2)
            messages_list = [
                [{"role": "user", "content": DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)]}]
                for i in range(actual_requests)
            ]

            print(f"\n  Running concurrency={conc}, requests={actual_requests}...")
            results, duration = await run_test(
                session, messages_list, conc, max_tokens, base_url, model_name
            )

            bench = aggregate_results(
                results, engine, "concurrency_scaling", model_name,
                conc, max_tokens, actual_requests, duration,
            )
            print_result(bench)
            all_results.append(bench)

            failed = sum(1 for r in results if not r.success)
            if failed > actual_requests * 0.5:
                print(f"  WARNING: >50% failures at concurrency={conc}, stopping scaling test")
                break

    return all_results


# ---------- Benchmark 2: Prefix Cache Effectiveness ----------

async def bench_prefix_cache(
    base_url: str, model_name: str, engine: str, max_tokens: int = 128, num_requests: int = 40,
) -> list[BenchmarkResult]:
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 2: Prefix Cache Effectiveness ({engine})")
    print(f"  Goal: Compare shared-prefix vs unique-prefix performance")
    print(f"  Why: SGLang RadixAttention vs vLLM APC cache reuse strategy")
    print(f"{'='*70}")

    concurrency = 8
    all_results = []

    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print("  Warming up...")
        warmup_msgs = [[{"role": "user", "content": "Hello"}]] * 4
        await run_test(session, warmup_msgs, 4, 16, base_url, model_name)

        # Test A: Shared prefix (same system prompt + different questions)
        print(f"\n  [A] Shared prefix: same system prompt, {num_requests} different questions")
        shared_messages = []
        for i in range(num_requests):
            suffix = PREFIX_CACHE_SUFFIXES[i % len(PREFIX_CACHE_SUFFIXES)]
            shared_messages.append([
                {"role": "user", "content": f"{SHARED_SYSTEM_PROMPT}\n\nQuestion: {suffix}"},
            ])

        results_shared, dur_shared = await run_test(
            session, shared_messages, concurrency, max_tokens, base_url, model_name
        )
        bench_shared = aggregate_results(
            results_shared, engine, "prefix_cache_shared", model_name,
            concurrency, max_tokens, num_requests, dur_shared,
        )
        print_result(bench_shared)
        all_results.append(bench_shared)

        # Test B: Unique prefixes (different system prompts + different questions)
        print(f"\n  [B] Unique prefix: different system prompts, {num_requests} different questions")
        unique_messages = []
        for i in range(num_requests):
            prefix = UNIQUE_PREFIXES[i % len(UNIQUE_PREFIXES)]
            suffix = PREFIX_CACHE_SUFFIXES[i % len(PREFIX_CACHE_SUFFIXES)]
            # Add random suffix to prefix to ensure no caching
            rand_id = ''.join(random.choices(string.ascii_lowercase, k=8))
            unique_messages.append([
                {"role": "user", "content": f"{prefix} (session {rand_id}) {suffix}"},
            ])

        results_unique, dur_unique = await run_test(
            session, unique_messages, concurrency, max_tokens, base_url, model_name
        )
        bench_unique = aggregate_results(
            results_unique, engine, "prefix_cache_unique", model_name,
            concurrency, max_tokens, num_requests, dur_unique,
        )
        print_result(bench_unique)
        all_results.append(bench_unique)

        # Summary
        if bench_shared.generation_tps > 0 and bench_unique.generation_tps > 0:
            tps_diff = (bench_shared.generation_tps / bench_unique.generation_tps - 1) * 100
            ttft_diff = (bench_unique.ttft_p50_ms / bench_shared.ttft_p50_ms - 1) * 100 if bench_shared.ttft_p50_ms > 0 else 0
            print(f"\n  >>> Prefix cache effect: Gen TPS {tps_diff:+.1f}%, TTFT {ttft_diff:+.1f}% faster with shared prefix")

    return all_results


# ---------- Benchmark 3: Sustained Load (Tail Latency) ----------

async def bench_sustained_load(
    base_url: str, model_name: str, engine: str, max_tokens: int = 128,
) -> list[BenchmarkResult]:
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 3: Sustained Load - Tail Latency Stability ({engine})")
    print(f"  Goal: Measure p99/p50 ratio under sustained high concurrency")
    print(f"  Why: Reveals scheduler stability under pressure")
    print(f"{'='*70}")

    concurrency = 16
    num_requests = 100
    all_results = []

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print("  Warming up...")
        warmup_msgs = [[{"role": "user", "content": "Hello"}]] * 4
        await run_test(session, warmup_msgs, 4, 16, base_url, model_name)

        messages_list = [
            [{"role": "user", "content": DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)]}]
            for i in range(num_requests)
        ]

        print(f"\n  Running {num_requests} requests at concurrency={concurrency}...")
        results, duration = await run_test(
            session, messages_list, concurrency, max_tokens, base_url, model_name
        )

        bench = aggregate_results(
            results, engine, "sustained_load", model_name,
            concurrency, max_tokens, num_requests, duration,
        )
        print_result(bench)
        all_results.append(bench)

        print(f"\n  >>> Tail latency ratios (lower = more stable):")
        print(f"      TTFT p99/p50: {bench.ttft_p99_p50_ratio:.2f}x")
        print(f"      ITL  p99/p50: {bench.itl_p99_p50_ratio:.2f}x")
        print(f"      E2E  p99/p50: {bench.e2e_p99_p50_ratio:.2f}x")

    return all_results


# ---------- Main ----------

async def main():
    parser = argparse.ArgumentParser(description="SGLang vs vLLM Comparison Benchmark")
    parser.add_argument("--url", required=True, help="API base URL (e.g., http://localhost:30900/v1)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--engine", required=True, choices=["sglang", "vllm"], help="Engine name")
    parser.add_argument("--output-dir", default="results/comparison", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument(
        "--tests", nargs="+",
        default=["scaling", "prefix", "sustained"],
        choices=["scaling", "prefix", "sustained"],
        help="Which tests to run",
    )
    args = parser.parse_args()

    # Check connectivity
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                models = await resp.json()
                print(f"Connected to {args.url} ({args.engine})")
                print(f"Models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        print(f"Cannot connect to {args.url}: {e}")
        return

    all_results = []

    if "scaling" in args.tests:
        results = await bench_concurrency_scaling(args.url, args.model, args.engine, args.max_tokens)
        all_results.extend(results)

    if "prefix" in args.tests:
        results = await bench_prefix_cache(args.url, args.model, args.engine, args.max_tokens)
        all_results.extend(results)

    if "sustained" in args.tests:
        results = await bench_sustained_load(args.url, args.model, args.engine, args.max_tokens)
        all_results.extend(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{args.engine}_{ts}.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*90}")
    print(f"  SUMMARY: {args.engine.upper()}")
    print(f"{'='*90}")
    print(f"{'Test':<25} {'Conc':>5} {'TTFT p50':>9} {'ITL p50':>9} {'E2E p50':>10} {'Gen TPS':>9} {'p99/p50':>8}")
    print(f"{'─'*90}")
    for r in all_results:
        label = r.test_name
        print(f"{label:<25} {r.concurrency:>5} {r.ttft_p50_ms:>8.1f}ms {r.itl_p50_ms:>8.1f}ms {r.e2e_p50_ms:>9.1f}ms {r.generation_tps:>8.1f} {r.ttft_p99_p50_ratio:>7.2f}x")
    print()


if __name__ == "__main__":
    asyncio.run(main())
