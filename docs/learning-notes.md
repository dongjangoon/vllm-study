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

---

## vLLM 배포

### Q: torch.compile에서 `-lcuda` 에러가 발생하면?

**A:** `--enforce-eager` 플래그를 추가하여 torch.compile 최적화를 비활성화합니다.

**에러 메시지:**
```
subprocess.CalledProcessError: Command '['which', 'c++']' returned non-zero exit status 1
CUDA runtime error: -lcuda not found
```

**원인:**
- vLLM이 기본적으로 torch.compile()을 사용해 CUDA 커널 최적화 시도
- WSL 환경이나 컨테이너 내부에 C++ 컴파일러가 없으면 실패

**해결:**
```yaml
args:
- "--enforce-eager"  # torch.compile 최적화 skip
```

**트레이드오프:**
- 첫 추론 시 JIT 컴파일 시간 절약
- 하지만 최적화된 커널을 사용하지 않아 약간의 성능 저하 가능

---

### Q: GPU 메모리 부족 (OOM) 발생 시 대처법?

**A:** 모델 크기 축소 또는 설정 조정이 필요합니다.

**에러 메시지:**
```
ValueError: The model's max seq len (32768) is larger than the maximum number of tokens
that can be stored in KV cache. Try increasing gpu_memory_utilization or decreasing max_model_len
```

**RTX 5070 Ti (16GB VRAM) 기준:**

| 모델 | 파라미터 | 모델 크기 | KV Cache 여유 | 결과 |
|------|----------|-----------|---------------|------|
| Qwen2.5-7B | 7B | ~14.25GB | -1.61GB | OOM |
| Qwen2.5-3B | 3B | ~6.5GB | ~7GB+ | 성공 |

**해결 방법 (우선순위순):**
1. 더 작은 모델 선택 (7B → 3B)
2. `--max-model-len` 감소 (8192 → 4096 → 2048)
3. `--gpu-memory-utilization` 조정 (0.90 → 0.95, 주의 필요)
4. 양자화 모델 사용 (AWQ, GPTQ)

---

### Q: 단일 GPU 환경에서 Deployment 재배포 에러?

**A:** `strategy: Recreate`를 사용합니다.

**문제:**
- 기본 RollingUpdate 전략은 새 Pod를 먼저 생성 후 기존 Pod 삭제
- 단일 GPU 환경에서는 새 Pod가 GPU를 요청해도 사용 불가
- 새 Pod가 Pending 상태로 무한 대기

**해결:**
```yaml
spec:
  strategy:
    type: Recreate  # 기존 Pod 먼저 삭제 후 새 Pod 생성
```

**동작 차이:**

| 전략 | 동작 순서 | 단일 GPU 적합성 |
|------|----------|-----------------|
| RollingUpdate | 새 Pod 생성 → 기존 삭제 | 불가능 |
| Recreate | 기존 삭제 → 새 Pod 생성 | 적합 |

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-10 | 초기 작성 (Traefik, /dev/shm) |
| 2025-01-10 | vLLM 트러블슈팅 추가 (enforce-eager, OOM, Recreate) |
