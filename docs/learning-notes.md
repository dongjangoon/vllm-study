# 학습 노트

프로젝트 진행 중 발생한 질문과 답변을 정리합니다.

---

## Kubernetes / k3s

### Q: Traefik을 비활성화해도 외부 접속이 가능한가?

**A:** 가능합니다. Traefik은 k3s 기본 Ingress Controller일 뿐이며, 다른 방법으로 접속할 수 있습니다.

| 방법 | 설명 | 사용 예 |
|------|------|---------|
| **NodePort** | 기본 제공, 포트 직접 노출 | `kubectl expose --type=NodePort` |
| **Port-forward** | 개발용, 항상 가능 | `kubectl port-forward svc/vllm 8000:8000` |
| **다른 Ingress** | Nginx, Contour 등 설치 가능 | 원하는 것 선택 |
| **LoadBalancer** | MetalLB 설치 시 사용 가능 | 프로덕션 환경 |

Traefik 비활성화 이유:
- 리소스 절약 (GPU 작업에 집중)
- 필요 시 원하는 Ingress Controller 선택 가능

---

## vLLM / 컨테이너

### Q: Deployment에서 /dev/shm 볼륨은 무엇인가?

**A:** 프로세스 간 공유 메모리 (Shared Memory)입니다.

| 항목 | 설명 |
|------|------|
| **역할** | 프로세스 간 공유 메모리 (RAM 기반 tmpfs) |
| **vLLM 사용 목적** | PyTorch 멀티프로세싱, 내부 데이터 공유 |
| **기본값 문제** | 컨테이너 기본 /dev/shm = 64MB (너무 작음) |
| **설정값** | 4Gi로 확장 |

**왜 필요한가?**
```
vLLM 프로세스 ─┬─ Worker 1 ─┐
               ├─ Worker 2 ─┼─► /dev/shm (공유 메모리)
               └─ Worker 3 ─┘
```

- **Continuous Batching**: 여러 요청을 동시에 처리할 때 데이터 공유
- **Paged Attention**: KV-cache 블록을 효율적으로 관리
- **모델 로딩**: 가중치 데이터 공유

**/dev/shm 부족 시 에러:**
```
RuntimeError: unable to mmap ... No space left on device
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-10 | 초기 작성 (Traefik, /dev/shm) |
