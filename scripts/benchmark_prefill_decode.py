#!/usr/bin/env python3
"""
Prefill vs Decode 단계 분석

TTFT (Time To First Token)를 측정하여 Prefill 시간을 분리하고,
프롬프트 길이별 두 단계의 특성을 비교합니다.
"""

import requests
import time
import json

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 다양한 길이의 프롬프트 생성
def generate_prompt(target_tokens: int) -> str:
    """대략 target_tokens 길이의 프롬프트 생성"""
    # 한글 기준 약 1.5 토큰/글자, 영어 기준 약 1.3 토큰/단어
    base_text = """
마이크로서비스 아키텍처는 애플리케이션을 작고 독립적인 서비스들의 집합으로 구성하는 방식입니다.
각 서비스는 특정 비즈니스 기능을 담당하며, 독립적으로 배포하고 확장할 수 있습니다.
Kubernetes는 컨테이너 오케스트레이션 플랫폼으로, Pod, Service, Deployment 등의 개념을 제공합니다.
Docker는 애플리케이션을 컨테이너로 패키징하여 일관된 실행 환경을 보장합니다.
"""
    # 목표 토큰 수에 맞게 반복
    repeats = max(1, target_tokens // 80)  # base_text가 약 80 토큰
    return (base_text * repeats).strip()


def measure_prefill_decode(prompt: str, max_tokens: int = 100) -> dict:
    """
    Streaming API로 TTFT와 전체 시간 측정
    """
    start_time = time.time()
    ttft = None
    total_tokens = 0

    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt + "\n\n위 내용을 요약해주세요."}],
                "max_tokens": max_tokens,
                "stream": True,  # Streaming 활성화
            },
            stream=True,
            timeout=120,
        )

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        # 첫 번째 토큰 시간 기록
                        if ttft is None and chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            ttft = time.time() - start_time
                        # 토큰 카운트 (delta에 cotent가 있는 경우)
                        if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            total_tokens += 1
                    except json.JSONDecodeError:
                        pass
        
        total_time = time.time() - start_time
        decode_time = total_time - ttft if ttft else total_time

        return {
            "success": True,
            "ttft": ttft,   # Prefill 시간
            "total_time": total_time,
            "decode_time": decode_time,
            "completion_tokens": total_tokens,
            "decode_throughput": total_tokens / decode_time if decode_time > 0 else 0,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    

def measure_with_usage_api(prompt: str, max_tokens: int = 100) -> dict:
    """
    Non-streaming API로 정확한 토큰 수 측정 (비교용)
    """
    start_time = time.time()

    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt + "\n\n위 내용을 요약해주세요."}],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )

        total_time = time.time() - start_time
        data = response.json()
        usage = data.get("usage", {})

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_time": total_time,
        }
    
    except Exception as e:
        return { "error": str(e) }
    

def run_benchmark():
    """
    프롬프트 길이별 Prefill vs Decode 벤치마크 실행
    """
    print("=" * 70)
    print("Prefill vs Decode 단계 분석")
    print("=" * 70)
    print(f"모델: {MODEL}")
    print("측정 방법: Streaming API로 TTFT (Time To First Token) 측정")
    print("=" * 70)
    print()

    # 프롬프트 길이 변화, 출력 고정
    prompt_lengths = [100, 300, 500, 1000, 1500, 2000]
    max_tokens = 100  # 출력 토큰 고정
    num_iterations = 3

    results = []

    for target_len in prompt_lengths:
        prompt = generate_prompt(target_len)

        print(f"\n[프롬프트 ~{target_len} 토큰] 테스트 중...")

        # 먼저 실제 토큰 수 확인
        usage_result = measure_with_usage_api(prompt, max_tokens)
        actual_prompt_tokens = usage_result.get("prompt_tokens", target_len)

        # 여러 번 측정하여 평균
        measurements = []
        for i in range(num_iterations):
            result = measure_prefill_decode(prompt, max_tokens)
            if result["success"]:
                measurements.append(result)
        
        if measurements:
            avg_ttft = sum(m["ttft"] for m in measurements) / len(measurements)
            avg_decode_time = sum(m["decode_time"] for m in measurements) / len(measurements)
            avg_total = sum(m["total_time"] for m in measurements) / len(measurements)
            avg_tokens = sum(m["completion_tokens"] for m in measurements) / len(measurements)

            # Prefill 처리량: prompt_tokens / ttft
            prefill_throughput = actual_prompt_tokens / avg_ttft if avg_ttft > 0 else 0
            # Decode 처리량: completion_tokens / decode_time
            decode_throughput = avg_tokens / avg_decode_time if avg_decode_time > 0 else 0

            result_data = {
                "target_prompt_tokens": target_len,
                "actual_prompt_tokens": actual_prompt_tokens,
                "completion_tokens": avg_tokens,
                "ttft": avg_ttft,
                "decode_time": avg_decode_time,
                "total_time": avg_total,
                "prefill ratio": (avg_ttft / avg_total) * 100,
                "prefill_throughput": prefill_throughput,
                "decode_throughput": decode_throughput,
            }
            results.append(result_data)

            print(f"  실제 프롬프트: {actual_prompt_tokens} 토큰")
            print(f"  TTFT (Prefill): {avg_ttft:.3f}s")
            print(f"  Decode 시간: {avg_decode_time:.3f}s")
            print(f"  전체 시간: {avg_total:.3f}s")
            print(f"  Prefill 비율: {result_data['prefill ratio']:.1f}%")
            print(f"  Prefill 처리량: {prefill_throughput:.1f} tokens/s")
            print(f"  Decode 처리량: {decode_throughput:.1f} tokens/s")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"{'Prompt':>8} | {'TTFT':>8} | {'Decode':>8} | {'Total':>8} | {'Prefill%':>9} | {'Prefill TPS':>12} | {'Decode TPS':>11}")
    print("-" *  85)

    for r in results:
        print(f"{r['actual_prompt_tokens']:>8} | {r['ttft']:>7.3f} | {r['decode_time']:>7.3f} | {r['total_time']:>7.3f} | {r['prefill ratio']:>8.1f}% | {r['prefill_throughput']:>11.1f} | {r['decode_throughput']:>10.1f}")

    print("=" * 70)

    # 분석
    print("\n[분석]")
    if len(results) >= 2:
        first = results[0]
        last = results[-1]

        print(f"• 프롬프트 {first['actual_prompt_tokens']} → {last['actual_prompt_tokens']} 토큰:")
        print(f"  - TTFT (Prefill): {first['ttft']:.3f}s → {last['ttft']:.3f}s ({last['ttft']/first['ttft']:.1f}x)")
        print(f"  - Prefill 비율: {first['prefill ratio']:.1f}% → {last['prefill ratio']:.1f}%")
        print(f"  - Prefill 처리량: {first['prefill_throughput']:.0f} → {last['prefill_throughput']:.0f} t/s")
        print(f"  - Decode 처리량: {first['decode_throughput']:.0f} → {last['decode_throughput']:.0f} t/s (거의 일정)")

    # JSON 저장
    with open("/tmp/prefill_decode_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: /tmp/prefill_decode_benchmark.json")


if __name__ == "__main__":
    run_benchmark()