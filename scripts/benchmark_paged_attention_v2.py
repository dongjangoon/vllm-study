#!/usr/bin/env python3
"""
PagedAttention 메모리 효율성 분석 v2

VRAM이 빠듯한 환경(Mistral 7B AWQ)에서 PagedAttention 효과 측정:
- 긴 시퀀스로 KV Cache 압박
- 높은 동시성으로 블록 경쟁 유발
- 실시간 메트릭 모니터링
"""

import requests
import time
import json
import concurrent.futures
import threading
from typing import Dict, List

VLLM_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"
MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"


def get_vllm_metrics() -> Dict:
    """vLLM Prometheus 메트릭 수집"""
    try:
        response = requests.get(METRICS_URL, timeout=5)
        metrics = {}
        for line in response.text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split('{')[0]
                    try:
                        metrics[name] = float(parts[1])
                    except ValueError:
                        pass
        return metrics
    except Exception as e:
        return {"error": str(e)}


def generate_long_prompt(tokens: int = 2000) -> str:
    """긴 프롬프트 생성"""
    base = """마이크로서비스 아키텍처는 애플리케이션을 작고 독립적인 서비스들의 집합으로 구성하는 방식입니다.
각 서비스는 특정 비즈니스 기능을 담당하며 독립적으로 배포할 수 있습니다.
Kubernetes는 컨테이너 오케스트레이션 플랫폼으로 Pod, Service, Deployment 개념을 제공합니다.
"""
    repeats = max(1, tokens // 60)
    return (base * repeats).strip()


def send_long_request(request_id: int, prompt: str, max_tokens: int) -> Dict:
    """긴 요청 전송"""
    start = time.time()
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        elapsed = time.time() - start
        data = response.json()

        if "error" in data:
            return {"id": request_id, "success": False, "error": str(data["error"]), "time": elapsed}

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


def monitor_metrics(duration: float, interval: float = 0.5) -> List[Dict]:
    """메트릭을 주기적으로 수집"""
    metrics_history = []
    start = time.time()

    while time.time() - start < duration:
        m = get_vllm_metrics()
        m["timestamp"] = time.time() - start
        metrics_history.append(m)
        time.sleep(interval)

    return metrics_history


def run_stress_test(num_requests: int, prompt_tokens: int, max_output: int) -> Dict:
    """스트레스 테스트 실행 + 메트릭 모니터링"""
    prompt = generate_long_prompt(prompt_tokens)

    # 메트릭 모니터링 스레드 시작
    metrics_history = []
    monitor_done = threading.Event()

    def monitor_thread():
        while not monitor_done.is_set():
            m = get_vllm_metrics()
            m["timestamp"] = time.time()
            metrics_history.append(m)
            time.sleep(0.3)

    monitor = threading.Thread(target=monitor_thread)
    monitor.start()

    # 동시 요청 전송
    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(send_long_request, i + 1, prompt, max_output)
            for i in range(num_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start_time
    monitor_done.set()
    monitor.join()

    # 메트릭 분석
    max_cache_usage = 0
    max_running = 0
    max_waiting = 0

    for m in metrics_history:
        cache = m.get("vllm:gpu_cache_usage_perc", 0) * 100
        running = m.get("vllm:num_requests_running", 0)
        waiting = m.get("vllm:num_requests_waiting", 0)

        max_cache_usage = max(max_cache_usage, cache)
        max_running = max(max_running, running)
        max_waiting = max(max_waiting, waiting)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    return {
        "num_requests": num_requests,
        "prompt_tokens": prompt_tokens,
        "max_output": max_output,
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "max_cache_usage": max_cache_usage,
        "max_running": max_running,
        "max_waiting": max_waiting,
        "avg_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in successful) / len(successful) if successful else 0,
        "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in successful) / len(successful) if successful else 0,
        "errors": [r.get("error") for r in failed],
    }


def main():
    print("=" * 70)
    print("PagedAttention 스트레스 테스트 (Mistral 7B AWQ)")
    print("=" * 70)
    print(f"모델: {MODEL}")
    print(f"GPU 메모리: ~97% 사용 중 (VRAM 빠듯한 환경)")
    print("=" * 70)

    # 초기 상태 확인
    initial = get_vllm_metrics()
    num_blocks = initial.get("vllm:num_gpu_blocks", 0)
    print(f"\n[초기 상태]")
    print(f"  GPU 블록 수: {num_blocks:.0f}")
    print(f"  블록당 토큰: 16")
    print(f"  총 KV Cache 용량: {num_blocks * 16:.0f} 토큰")

    results = []

    # 테스트 1: 동시 요청 수 증가 (긴 시퀀스)
    print("\n" + "=" * 70)
    print("[테스트 1] 동시 요청 수 증가 (긴 시퀀스)")
    print("=" * 70)

    test_configs = [
        (5, 1000, 200),   # 5개 요청, 1000 토큰 프롬프트, 200 출력
        (10, 1000, 200),
        (15, 1000, 200),
        (20, 1000, 200),
        (25, 1000, 200),
        (30, 1000, 200),
    ]

    for num_req, prompt_tok, max_out in test_configs:
        print(f"\n[동시 {num_req}개, 프롬프트 ~{prompt_tok}, 출력 {max_out}] 테스트 중...")
        result = run_stress_test(num_req, prompt_tok, max_out)
        results.append(result)

        print(f"  성공/실패: {result['successful']}/{result['failed']}")
        print(f"  최대 KV Cache 사용률: {result['max_cache_usage']:.1f}%")
        print(f"  최대 실행 중: {result['max_running']:.0f}, 최대 대기: {result['max_waiting']:.0f}")
        print(f"  총 소요 시간: {result['total_time']:.1f}s")

        if result['errors']:
            print(f"  에러: {result['errors'][0][:50]}...")

        # 잠시 대기 (블록 해제)
        time.sleep(2)

    # 테스트 2: 시퀀스 길이 증가 (고정 동시성)
    print("\n" + "=" * 70)
    print("[테스트 2] 시퀀스 길이 증가 (동시 10개)")
    print("=" * 70)

    length_configs = [
        (10, 500, 100),
        (10, 1000, 200),
        (10, 2000, 300),
        (10, 3000, 400),
        (10, 4000, 500),
    ]

    for num_req, prompt_tok, max_out in length_configs:
        print(f"\n[동시 {num_req}개, 프롬프트 ~{prompt_tok}, 출력 {max_out}] 테스트 중...")
        result = run_stress_test(num_req, prompt_tok, max_out)
        results.append(result)

        total_tokens = result['avg_prompt_tokens'] + result['avg_completion_tokens']
        print(f"  성공/실패: {result['successful']}/{result['failed']}")
        print(f"  평균 총 토큰: {total_tokens:.0f}")
        print(f"  최대 KV Cache 사용률: {result['max_cache_usage']:.1f}%")
        print(f"  최대 실행 중: {result['max_running']:.0f}, 최대 대기: {result['max_waiting']:.0f}")

        if result['failed'] > 0:
            print(f"  ⚠️  메모리 부족으로 일부 실패!")

        time.sleep(2)

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    print("\n[동시 요청 수 vs KV Cache 사용률]")
    print(f"{'동시요청':>8} | {'Cache%':>8} | {'실행중':>6} | {'대기':>6} | {'성공':>6} | {'실패':>6}")
    print("-" * 55)

    for r in results[:6]:
        print(f"{r['num_requests']:>8} | {r['max_cache_usage']:>7.1f}% | {r['max_running']:>6.0f} | {r['max_waiting']:>6.0f} | {r['successful']:>6} | {r['failed']:>6}")

    print("\n[시퀀스 길이 vs KV Cache 사용률]")
    print(f"{'토큰수':>8} | {'Cache%':>8} | {'실행중':>6} | {'대기':>6} | {'성공':>6} | {'실패':>6}")
    print("-" * 55)

    for r in results[6:]:
        total = r['avg_prompt_tokens'] + r['avg_completion_tokens']
        print(f"{total:>8.0f} | {r['max_cache_usage']:>7.1f}% | {r['max_running']:>6.0f} | {r['max_waiting']:>6.0f} | {r['successful']:>6} | {r['failed']:>6}")

    # JSON 저장
    with open("/tmp/paged_attention_stress.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n결과 저장: /tmp/paged_attention_stress.json")


if __name__ == "__main__":
    main()
