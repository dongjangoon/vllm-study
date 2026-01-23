#!/usr/bin/env python3
"""
PagedAttention 메모리 효율성 분석

vLLM의 블록 기반 메모리 관리 효율성을 측정합니다.
- 동시 요청 수별 블록 사용률
- 시퀀스 길이별 블록 할당 패턴
- Prefix Caching 효과
"""

import requests
import time
import json
import concurrent.futures
from typing import List, Dict

VLLM_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def get_vllm_metrics() -> Dict:
    """vLLM Prometheus 메트릭 수집"""
    try:
        response = requests.get(METRICS_URL, timeout=5)
        metrics = {}
        for line in response.text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    # 메트릭 이름에서 라벨 제거
                    name = parts[0].split('{')[0]
                    try:
                        metrics[name] = float(parts[1])
                    except ValueError:
                        pass
        return metrics
    except Exception as e:
        return {"error": str(e)}


def generate_prompt(length: str = "short") -> str:
    """다양한 길이의 프롬프트 생성"""
    base = "마이크로서비스 아키텍처에 대해 설명해주세요. "

    if length == "short":
        return base  # ~20 토큰
    elif length == "medium":
        return base * 10  # ~200 토큰
    elif length == "long":
        return base * 50  # ~1000 토큰
    elif length == "very_long":
        return base * 100  # ~2000 토큰
    return base


def send_request_and_hold(prompt: str, max_tokens: int, hold_time: float = 0) -> Dict:
    """요청 전송 (hold_time 동안 연결 유지)"""
    start = time.time()
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = response.json()

        if hold_time > 0:
            time.sleep(hold_time)

        usage = data.get("usage", {})
        return {
            "success": True,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "time": elapsed,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def measure_block_usage(num_concurrent: int, prompt_length: str, max_tokens: int) -> Dict:
    """동시 요청 시 블록 사용량 측정"""
    prompt = generate_prompt(prompt_length)

    # 요청 전 메트릭
    before_metrics = get_vllm_metrics()

    # 동시 요청 전송
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [
            executor.submit(send_request_and_hold, prompt, max_tokens, 0)
            for _ in range(num_concurrent)
        ]

        # 요청 진행 중 메트릭 (첫 번째 완료 전에 캡처 시도)
        time.sleep(0.5)
        during_metrics = get_vllm_metrics()

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 요청 후 메트릭
    after_metrics = get_vllm_metrics()

    successful = [r for r in results if r.get("success")]

    return {
        "num_concurrent": num_concurrent,
        "prompt_length": prompt_length,
        "max_tokens": max_tokens,
        "successful_requests": len(successful),
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in successful) / len(successful) if successful else 0,
        "avg_completion_tokens": sum(r["completion_tokens"] for r in successful) / len(successful) if successful else 0,
        "metrics_before": before_metrics,
        "metrics_during": during_metrics,
        "metrics_after": after_metrics,
    }


def analyze_block_efficiency():
    """블록 사용 효율성 분석"""
    print("=" * 70)
    print("PagedAttention 메모리 효율성 분석")
    print("=" * 70)

    # vLLM 설정 확인
    initial_metrics = get_vllm_metrics()
    num_gpu_blocks = initial_metrics.get("vllm:num_gpu_blocks", 0)
    block_size = 16  # vLLM 기본값

    print(f"\n[vLLM 블록 설정]")
    print(f"  총 GPU 블록 수: {num_gpu_blocks:.0f}")
    print(f"  블록 크기: {block_size} 토큰/블록")
    print(f"  총 KV Cache 용량: {num_gpu_blocks * block_size:.0f} 토큰")
    print()

    results = []

    # 실험 1: 동시 요청 수별 블록 사용률
    print("=" * 70)
    print("[실험 1] 동시 요청 수별 블록 사용률")
    print("=" * 70)

    concurrent_tests = [1, 5, 10, 20, 30]

    for num_concurrent in concurrent_tests:
        print(f"\n동시 요청 {num_concurrent}개 테스트 중...")
        result = measure_block_usage(num_concurrent, "medium", 100)
        results.append(result)

        during = result["metrics_during"]
        cache_usage = during.get("vllm:gpu_cache_usage_perc", 0) * 100
        running = during.get("vllm:num_requests_running", 0)
        waiting = during.get("vllm:num_requests_waiting", 0)

        print(f"  실행 중: {running:.0f}, 대기 중: {waiting:.0f}")
        print(f"  KV Cache 사용률: {cache_usage:.1f}%")
        print(f"  평균 프롬프트: {result['avg_prompt_tokens']:.0f} 토큰")
        print(f"  평균 출력: {result['avg_completion_tokens']:.0f} 토큰")

    # 실험 2: 시퀀스 길이별 블록 할당
    print("\n" + "=" * 70)
    print("[실험 2] 시퀀스 길이별 블록 할당")
    print("=" * 70)

    length_tests = [
        ("short", 50),
        ("medium", 100),
        ("long", 150),
        ("very_long", 200),
    ]

    for length, max_tokens in length_tests:
        print(f"\n프롬프트 길이: {length}, 최대 출력: {max_tokens} 토큰...")
        result = measure_block_usage(5, length, max_tokens)

        during = result["metrics_during"]
        cache_usage = during.get("vllm:gpu_cache_usage_perc", 0) * 100

        total_tokens = result["avg_prompt_tokens"] + result["avg_completion_tokens"]
        expected_blocks = total_tokens / block_size

        print(f"  평균 총 토큰: {total_tokens:.0f}")
        print(f"  예상 블록 수 (per request): {expected_blocks:.1f}")
        print(f"  KV Cache 사용률: {cache_usage:.1f}%")

    # 실험 3: Prefix Caching 효과
    print("\n" + "=" * 70)
    print("[실험 3] Prefix Caching 효과")
    print("=" * 70)

    # 동일한 프롬프트로 연속 요청
    same_prompt = generate_prompt("long")

    print("\n동일 프롬프트로 10회 연속 요청...")
    prefix_results = []

    for i in range(10):
        before = get_vllm_metrics()
        result = send_request_and_hold(same_prompt, 50, 0)
        after = get_vllm_metrics()

        prefix_hits_before = before.get("vllm:prefix_cache_hit_rate", 0)
        prefix_hits_after = after.get("vllm:prefix_cache_hit_rate", 0)

        prefix_results.append({
            "iteration": i + 1,
            "time": result.get("time", 0),
            "prefix_hit_rate": prefix_hits_after,
        })

        print(f"  요청 {i+1}: {result.get('time', 0):.3f}s, Prefix Hit Rate: {prefix_hits_after:.2%}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    print("\n[동시 요청 수 vs KV Cache 사용률]")
    print(f"{'동시 요청':>10} | {'Cache 사용률':>12} | {'실행 중':>8} | {'대기 중':>8}")
    print("-" * 50)

    for r in results[:len(concurrent_tests)]:
        during = r["metrics_during"]
        cache = during.get("vllm:gpu_cache_usage_perc", 0) * 100
        running = during.get("vllm:num_requests_running", 0)
        waiting = during.get("vllm:num_requests_waiting", 0)
        print(f"{r['num_concurrent']:>10} | {cache:>10.1f}% | {running:>8.0f} | {waiting:>8.0f}")

    print("\n[PagedAttention 효율성 분석]")
    if num_gpu_blocks > 0:
        # 이론적 최대 동시 요청 계산
        avg_tokens_per_request = 300  # 프롬프트 200 + 출력 100
        blocks_per_request = avg_tokens_per_request / block_size
        theoretical_max_concurrent = num_gpu_blocks / blocks_per_request

        print(f"  총 블록 수: {num_gpu_blocks:.0f}")
        print(f"  요청당 평균 블록: {blocks_per_request:.1f}")
        print(f"  이론적 최대 동시 요청: {theoretical_max_concurrent:.0f}")
        print(f"  실제 테스트 최대 동시: {max(concurrent_tests)}")

    # JSON 저장
    with open("/tmp/paged_attention_benchmark.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n결과 저장: /tmp/paged_attention_benchmark.json")


if __name__ == "__main__":
    analyze_block_efficiency()
