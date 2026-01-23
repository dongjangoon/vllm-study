#!/usr/bin/env python3
"""
토큰 길이별 Memory Bandwidth 분석

목표: Decode 단계 비중이 클수록 memory-bound 특성이 뚜렷해지는지 확인
"""

import requests
import time
import json

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

def benchmark_token_length(max_tokens: int, num_requests: int = 5) -> dict:
    """특정 max_tokens로 벤치마크"""
    results = []

    for i in range(num_requests):
        start = time.time()
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Write a detailed explanation about machine learning."}],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = response.json()

        if "usage" in data:
            usage = data["usage"]
            completion_tokens = usage.get("completion_tokens", 0)
            # Decode 처리량 = 생성된 토큰 / 전체 시간 (단순화)
            decode_throughput = completion_tokens / elapsed if elapsed > 0 else 0

            results.append({
                "completion_tokens": completion_tokens,
                "elapsed": elapsed,
                "decode_throughput": decode_throughput,
            })

    avg_throughput = sum(r["decode_throughput"] for r in results) / len(results)
    avg_tokens = sum(r["completion_tokens"] for r in results) / len(results)

    return {
        "max_tokens": max_tokens,
        "avg_completion_tokens": avg_tokens,
        "avg_decode_throughput": avg_throughput,
    }

def main():
    print("=" * 60)
    print("토큰 길이별 Memory Bandwidth 분석")
    print("=" * 60)

    token_lengths = [50, 100, 200, 300, 500]
    results = []

    for max_tokens in token_lengths:
        print(f"\n[Max Tokens: {max_tokens}] 테스트 중...")
        result = benchmark_token_length(max_tokens, num_requests=3)
        results.append(result)

        est_bw = result["avg_decode_throughput"] * 6.0  # 6GB 모델 기준
        print(f"  -> 평균 생성 토큰: {result['avg_completion_tokens']:.0f}, ")
        print(f"  -> Decode 처리량: {result['avg_decode_throughput']:.1f} tokens/s")
        print(f"  -> 추정 BW: {est_bw:.1f} GB/s")

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"{'Max Tokens':>12} | {'Actual':>8} | {'Throughput':>12} | {'Est. BW':>10}")
    print("-" * 50)

    for r in results:
        est_bw = r["avg_decode_throughput"] * 6.0
        print(f"{r['max_tokens']:>12} | {r['avg_completion_tokens']:>8.0f} | {r['avg_decode_throughput']:>10.1f} t/s | {est_bw:>8.1f} GB/s")

    with open("/tmp/token_length_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: /tmp/token_length_benchmark.json")

if __name__ == "__main__":
    main()