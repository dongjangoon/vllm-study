#!/usr/bin/env python3
"""vLLM 부하 테스트 스크립트"""

import requests
import time
import concurrent.futures
from datetime import datetime

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

PROMPTS = [
    "Python에서 리스트와 튜플의 차이점은?",
    "Docker와 Kubernetes의 관계를 설명해줘",
    "REST API란 무엇인가요?",
    "Git rebase와 merge의 차이는?",
    "TCP와 UDP의 차이점을 알려줘",
    "캐시 메모리가 필요한 이유는?",
    "데이터베이스 인덱스란?",
    "HTTP와 HTTPS의 차이는?",
    "프로세스와 스레드의 차이점은?",
    "가비지 컬렉션이란 무엇인가요?",
]

def send_request(prompt: str, request_id: int) -> dict:
    """단일 요청 전송"""
    start = time.time()
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
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

def run_load_test(num_requests: int = 10, concurrency: int = 3):
    """부하 테스트 실행"""
    print(f"\n{'='*60}")
    print(f"vLLM 부하 테스트 시작")
    print(f"총 요청: {num_requests}, 동시성: {concurrency}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            futures.append(executor.submit(send_request, prompt, i + 1))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["success"] else "FAIL"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request #{result['id']:02d}: {status} ({result['time']:.2f}s)")

    total_time = time.time() - start_time

    # 결과 분석
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    print(f"성공: {len(successful)}/{num_requests}")
    print(f"실패: {len(failed)}/{num_requests}")
    print(f"총 소요 시간: {total_time:.2f}s")

    if successful:
        avg_latency = sum(r["time"] for r in successful) / len(successful)
        total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
        total_completion_tokens = sum(r["completion_tokens"] for r in successful)
        throughput = (total_prompt_tokens + total_completion_tokens) / total_time

        print(f"평균 응답 시간: {avg_latency:.2f}s")
        print(f"총 토큰: {total_prompt_tokens + total_completion_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens})")
        print(f"처리량: {throughput:.1f} tokens/s")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run_load_test(num_requests, concurrency)
