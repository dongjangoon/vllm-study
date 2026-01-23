#!/usr/bin/env python3
"""긴 프롬프트로 KV Cache 테스트"""

import requests
import time
import concurrent.futures
from datetime import datetime

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 긴 프롬프트 생성 (약 2000 토큰)
LONG_CONTEXT = """
다음은 소프트웨어 아키텍처에 대한 상세한 설명입니다.

마이크로서비스 아키텍처는 애플리케이션을 작고 독립적인 서비스들의 집합으로 구성하는 소프트웨어 개발 방식입니다. 각 서비스는 특정 비즈니스 기능을 담당하며, 독립적으로 배포, 확장, 유지보수가 가능합니다.

주요 특징:
1. 서비스 독립성: 각 마이크로서비스는 자체 데이터베이스를 가지며, 다른 서비스와 느슨하게 결합됩니다.
2. 기술 다양성: 각 서비스는 해당 기능에 가장 적합한 기술 스택을 선택할 수 있습니다.
3. 확장성: 트래픽이 많은 서비스만 선택적으로 확장할 수 있습니다.
4. 장애 격리: 한 서비스의 장애가 전체 시스템에 영향을 미치지 않습니다.
5. 팀 자율성: 각 팀이 독립적으로 서비스를 개발하고 배포할 수 있습니다.

컨테이너화와 오케스트레이션:
Docker는 애플리케이션과 그 의존성을 컨테이너로 패키징하여 일관된 실행 환경을 제공합니다. Kubernetes는 이러한 컨테이너들을 대규모로 관리하고 오케스트레이션하는 플랫폼입니다.

Kubernetes의 핵심 개념:
- Pod: 하나 이상의 컨테이너를 포함하는 가장 작은 배포 단위
- Service: Pod들에 대한 네트워크 접근을 제공하는 추상화 계층
- Deployment: Pod의 선언적 업데이트를 관리
- ConfigMap/Secret: 설정 정보와 민감한 데이터를 관리
- Ingress: 외부 트래픽을 클러스터 내부 서비스로 라우팅

서비스 메시와 관찰성:
Istio, Linkerd 같은 서비스 메시는 마이크로서비스 간 통신을 관리합니다. 트래픽 관리, 보안, 관찰성 기능을 애플리케이션 코드 수정 없이 제공합니다.

관찰성의 세 가지 핵심 요소:
1. 메트릭: Prometheus로 수집하고 Grafana로 시각화
2. 로그: ELK 스택(Elasticsearch, Logstash, Kibana) 또는 Loki로 중앙 집중화
3. 트레이싱: Jaeger, Zipkin으로 분산 트레이싱 구현

CI/CD 파이프라인:
지속적 통합(CI)은 코드 변경사항을 자주 메인 브랜치에 병합하고 자동으로 빌드/테스트합니다.
지속적 배포(CD)는 프로덕션 환경까지 자동으로 배포하는 것을 의미합니다.

GitOps는 Git을 단일 진실 공급원(Single Source of Truth)으로 사용하여 인프라와 애플리케이션을 선언적으로 관리하는 방식입니다. ArgoCD, Flux 같은 도구가 대표적입니다.

데이터베이스 전략:
- 폴리글랏 퍼시스턴스: 각 서비스가 요구사항에 맞는 데이터베이스 선택
- CQRS: 명령과 조회를 분리하여 성능 최적화
- 이벤트 소싱: 상태 변경을 이벤트로 저장하여 감사 추적 가능
- Saga 패턴: 분산 트랜잭션을 일련의 로컬 트랜잭션으로 관리

보안 고려사항:
- 제로 트러스트 네트워크: 모든 요청을 검증
- mTLS: 서비스 간 통신 암호화
- RBAC: 역할 기반 접근 제어
- 시크릿 관리: Vault, AWS Secrets Manager 활용

성능 최적화:
- 캐싱: Redis, Memcached로 자주 접근하는 데이터 캐싱
- CDN: 정적 콘텐츠 분산 제공
- 비동기 처리: 메시지 큐(Kafka, RabbitMQ)를 활용한 비동기 통신
- 서킷 브레이커: 장애 전파 방지

모니터링과 알림:
효과적인 모니터링 전략은 SLI(Service Level Indicator), SLO(Service Level Objective), SLA(Service Level Agreement)를 정의하는 것에서 시작합니다.

골든 시그널:
1. 지연시간(Latency): 요청 처리 시간
2. 트래픽(Traffic): 시스템에 대한 요청량
3. 오류(Errors): 실패한 요청의 비율
4. 포화도(Saturation): 시스템 리소스 사용률

클라우드 네이티브 애플리케이션 개발에서 12-Factor App 방법론은 중요한 지침을 제공합니다:
1. 코드베이스: 버전 관리되는 하나의 코드베이스
2. 의존성: 명시적으로 선언하고 격리
3. 설정: 환경 변수에 저장
4. 백엔드 서비스: 연결된 리소스로 취급
5. 빌드/릴리스/실행: 엄격하게 분리
6. 프로세스: 무상태 프로세스로 실행
7. 포트 바인딩: 포트 바인딩을 통해 서비스 노출
8. 동시성: 프로세스 모델을 통한 확장
9. 폐기 가능성: 빠른 시작과 그레이스풀 셧다운
10. 개발/프로덕션 일치: 환경 간 차이 최소화
11. 로그: 이벤트 스트림으로 취급
12. 관리 프로세스: 일회성 프로세스로 실행

테스트 전략:
- 단위 테스트: 개별 컴포넌트 테스트
- 통합 테스트: 컴포넌트 간 상호작용 테스트
- 계약 테스트: 서비스 간 API 계약 검증
- E2E 테스트: 전체 시스템 흐름 테스트
- 카오스 엔지니어링: 장애 주입을 통한 복원력 테스트
"""

QUESTIONS = [
    "위 내용을 바탕으로 마이크로서비스의 장점 3가지를 설명해주세요.",
    "Kubernetes의 핵심 개념 중 Pod와 Service의 차이점을 설명해주세요.",
    "관찰성의 세 가지 핵심 요소와 각각에 사용되는 도구를 설명해주세요.",
    "12-Factor App 방법론에서 가장 중요하다고 생각하는 3가지를 선택하고 이유를 설명해주세요.",
    "서비스 메시의 역할과 장점을 위 내용을 바탕으로 설명해주세요.",
]

def send_request(question: str, request_id: int) -> dict:
    """긴 컨텍스트와 함께 요청 전송"""
    start = time.time()
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "당신은 소프트웨어 아키텍처 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 답변해주세요."},
                    {"role": "user", "content": LONG_CONTEXT + "\n\n질문: " + question}
                ],
                "max_tokens": 500,
            },
            timeout=120,
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

def run_load_test(num_requests: int = 10, concurrency: int = 5):
    """긴 프롬프트 부하 테스트"""
    print(f"\n{'='*60}")
    print(f"긴 프롬프트 KV Cache 테스트")
    print(f"총 요청: {num_requests}, 동시성: {concurrency}")
    print(f"예상 프롬프트 길이: ~2000 토큰")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            question = QUESTIONS[i % len(QUESTIONS)]
            futures.append(executor.submit(send_request, question, i + 1))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request #{result['id']:02d}: OK ({result['time']:.2f}s) - {result['prompt_tokens']} prompt tokens")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request #{result['id']:02d}: FAIL - {result.get('error', 'unknown')}")

    total_time = time.time() - start_time
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
        print(f"평균 프롬프트 토큰: {total_prompt_tokens // len(successful)}")
        print(f"총 토큰: {total_prompt_tokens + total_completion_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens})")
        print(f"처리량: {throughput:.1f} tokens/s")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_load_test(num_requests, concurrency)
