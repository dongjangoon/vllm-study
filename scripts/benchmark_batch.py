#!/usr/bin/env python3
"""
Memory Bandwidth 분석을 위한 Batch Size별 벤치마크

가설: Batch size 증가 -> 가중치 로딩 공유 -> 처리량 증가 (memory bandwidth 효율 향상)
"""

import requests
import time
import concurrent.futures
from datetime import datetime
import json

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# 동일한 프롬프트로 일관된 테스트
PROMPT = "Python으로 피보나치 수열을 구현하는 방법을 설명해주세요."

def send_request(request_id: int, max_tokens: int = 100) -> dict:
    """단일 요청 전송"""
    start = time.time()
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        elapsed = time.time() - start
        data = response.json()

        if "error" in data:
            return {"id": request_id, "success": False, "error": data["error"], "time": elapsed}

        usage = data.get("usage", {})
        return {
            "id": request_id,
            "success": True,
            "time": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"id": request_id, "success": False, "error": str(e), "time": time.time() - start}
    
def run_batch_benchmark(batch_size: int, num_iterations: int = 3) -> dict:
    """특정 배치 크기로 벤치마크 실행"""
    all_results = []

    for iteration in range(num_iterations):
        start_time = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(send_request, i + 1) for i in range(batch_size)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        total_time = time.time() - start_time
        successful = [r for r in results if r["success"]]

        if successful:
            total_completion_tokens = sum(r["completion_tokens"] for r in successful)
            total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)

            # 토큰 생성 처리량 (decode 단계)
            generation_throughput = total_completion_tokens / total_time

            all_results.append({
                "iteration": iteration + 1,
                "total_time": total_time,
                "generation_throughput": generation_throughput,
                "total_tokens": total_completion_tokens + total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            })
    
    # 평균 계산
    avg_throughput = sum(r["generation_throughput"] for r in all_results) / len(all_results)
    avg_time = sum(r["total_time"] for r in all_results) / len(all_results)

    return {
        "batch_size": batch_size,
        "avg_generation_throughput": avg_throughput,
        "avg_total_time": avg_time,
        "iterations": all_results,
    }

def estimate_memory_bandwidth(throughput_tokens_per_sec: float, model_size_gb: float = 6.0) -> float:
    """
    메모리 대역폭 활용률 추정

    가정: Decode 단계에서 토큰당 모델 가중치 전체를 읽어야 함
    실제 Memory BW = throughput × model_size
    """
    return throughput_tokens_per_sec * model_size_gb

def main():
    print("=" * 70)
    print("Memory Bandwidth 분석: Batch Size별 처리량 벤치마크")
    print("=" * 70)
    print(f"모델: {MODEL}")
    print(f"RTX 5070 Ti 이론 대역폭: ~504 GB/s")
    print(f"Qwen2.5-3B 모델 크기: ~6GB (FP16)")
    print("=" * 70)
    print()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []

    for batch_size in batch_sizes:
        print(f"\n[Batch Size: {batch_size}] 테스트 중...")
        result = run_batch_benchmark(batch_size, num_iterations=3)
        results.append(result)

        est_bw = estimate_memory_bandwidth(result["avg_generation_throughput"])
        utilization = (est_bw / 504) * 100

        print(f"  -> 평균 생성 처리량: {result['avg_generation_throughput']:.2f} tokens/s")
        print(f"  -> 추정 메모리 대역폭: {est_bw:.2f} GB/s ({utilization:.2f}% 활용률)")

    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"{'Batch':>8} | {'Throughput':>12} | {'Est. BW':>10} | {'Utilization':>12}")
    print("-" * 50)

    for r in results:
        est_bw = estimate_memory_bandwidth(r["avg_generation_throughput"])
        utilization = (est_bw / 504) * 100
        print(f"{r['batch_size']:>8} | {r['avg_generation_throughput']:>10.1f} t/s | {est_bw:>8.1f} GB/s | {utilization:>10.1f}%")
    
    print("=" * 70)

    # 결과 JSON 저장
    with open("/tmp/bandwidth_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n결과 저장: /tmp/bandwidth_benchmark.json")

if __name__ == "__main__":
    main()
