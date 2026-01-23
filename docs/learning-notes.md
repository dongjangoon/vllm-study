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

---

## KV Cache / PagedAttention

### Q: KV Cache란 무엇인가?

**A:** Transformer 모델의 Attention 연산에서 재사용되는 Key-Value 텐서를 저장하는 캐시입니다.

**왜 필요한가?**
```
토큰 생성 과정:
"Hello" → "Hello world" → "Hello world !" → ...

각 단계마다 이전 토큰들의 Key/Value를 다시 계산하면 비효율적
→ KV Cache에 저장해두고 재사용
```

**구조:**
```
┌─────────────────────────────────────────┐
│              KV Cache                    │
├─────────────────────────────────────────┤
│ Layer 0: [K₀, V₀] for tokens 0..n       │
│ Layer 1: [K₁, V₁] for tokens 0..n       │
│ ...                                      │
│ Layer L: [Kₗ, Vₗ] for tokens 0..n       │
└─────────────────────────────────────────┘
```

**메모리 사용량 계산:**
```
KV Cache 크기 = 2 × num_layers × hidden_size × num_tokens × dtype_size

예시 (Qwen2.5-3B):
- num_layers: 36
- hidden_size: 2048
- 1000 토큰, FP16
= 2 × 36 × 2048 × 1000 × 2 bytes = ~295MB
```

---

### Q: vLLM의 PagedAttention이란?

**A:** KV Cache를 고정 크기 블록으로 나눠 관리하는 메모리 관리 기법입니다. OS의 가상 메모리 페이징에서 영감을 받았습니다.

**기존 방식의 문제:**
```
┌──────────────────────────────────┐
│ Request 1: ████████░░░░░░░░░░░░ │  ← 예약은 했지만 사용 안 함
│ Request 2: ██████░░░░░░░░░░░░░░ │  ← 메모리 낭비 (내부 단편화)
│ Request 3: (대기)                │  ← 공간 있는데 할당 불가
└──────────────────────────────────┘
```

**PagedAttention:**
```
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │  ← 고정 크기 블록 (16 토큰)
├─────┴─────┼─────┴─────┼─────┴─────┤
│  Req 1    │  Req 2    │  Req 3    │  ← 필요한 만큼만 블록 할당
└───────────┴───────────┴───────────┘
```

**장점:**
| 항목 | 기존 방식 | PagedAttention |
|------|----------|----------------|
| 메모리 효율 | 50-60% | **~95%** |
| 동시 요청 수 | 제한적 | 2-4배 증가 |
| 단편화 | 심함 | 최소화 |

---

### Q: KV Cache 사용률이 낮은 이유는?

**A:** 여러 요인이 있습니다.

**우리 환경에서의 계산:**
```
설정:
- num_gpu_blocks: 12,956 블록
- block_size: 16 토큰/블록
- 총 용량: ~207,000 토큰

테스트 (동시 20개, 2000 토큰/요청):
- 필요: 20 × 2000 / 16 = 2,500 블록
- 사용률: 2,500 / 12,956 = ~19%

실제 측정: ~2%
```

**차이 발생 이유:**

| 요인 | 설명 |
|------|------|
| **Prefix Caching** | 동일 프롬프트의 KV를 공유 → 블록 재사용 |
| **동적 해제** | 요청 완료 시 즉시 블록 반환 |
| **작은 모델** | 3B 모델 = 레이어/hidden 크기 작음 |
| **짧은 출력** | 150-500 토큰 생성 후 종료 |

**KV Cache를 더 많이 사용하려면:**
1. 동시 요청 수 증가 (20 → 50+)
2. 긴 컨텍스트 (8K+ 토큰)
3. 큰 모델 (7B+)
4. Prefix Caching 비활성화

---

### Q: Continuous Batching이란?

**A:** 요청을 동적으로 배치에 추가/제거하는 방식입니다.

**기존 Static Batching:**
```
시간 →
Batch 1: [Req1, Req2, Req3] ───완료───┐
                                      ├─ 전부 끝날 때까지 대기
Batch 2: [Req4, Req5]      대기──────┘
```

**Continuous Batching:**
```
시간 →
Req1: ████████ (완료)
Req2: ████████████████ (완료)
Req3: ████████████ (완료)
Req4:     ████████████████ (Req1 완료 후 즉시 추가)
Req5:         ████████████ (Req3 완료 후 즉시 추가)
```

**성능 영향:**
```
부하 테스트 결과:
동시성 3  → 80.5 tok/s
동시성 5  → 154.4 tok/s (+92%)
동시성 10 → 285.1 tok/s (+85%)
동시성 20 → 607.4 tok/s (+113%)

응답 시간은 거의 동일 (6초) → Continuous Batching 효과!
```

---

## vLLM 모니터링

### Q: vLLM 메트릭을 Prometheus로 수집하는 방법?

**A:** vLLM은 기본으로 `/metrics` 엔드포인트를 제공합니다.

**Prometheus 설정:**
```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-qwen.vllm.svc:8000']
```

**주요 vLLM 메트릭:**

| 메트릭 | 설명 |
|--------|------|
| `vllm:num_requests_running` | 현재 처리 중인 요청 수 |
| `vllm:num_requests_waiting` | 대기 중인 요청 수 |
| `vllm:kv_cache_usage_perc` | KV Cache 사용률 (0-1) |
| `vllm:prefix_cache_hits_total` | Prefix Cache 히트 수 |
| `vllm:prompt_tokens_total` | 처리된 프롬프트 토큰 수 |
| `vllm:generation_tokens_total` | 생성된 토큰 수 |
| `vllm:e2e_request_latency_seconds` | E2E 요청 레이턴시 |
| `vllm:time_to_first_token_seconds` | 첫 토큰까지 시간 (TTFT) |

---

### Q: GPU 메트릭을 수집하는 방법?

**A:** DCGM Exporter를 사용합니다.

**배포:**
```yaml
# dcgm-deployment.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  template:
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
        securityContext:
          privileged: true
```

**주요 DCGM 메트릭:**

| 메트릭 | 설명 |
|--------|------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU 사용률 (%) |
| `DCGM_FI_DEV_FB_USED` | GPU 메모리 사용량 (MB) |
| `DCGM_FI_DEV_FB_FREE` | GPU 메모리 여유량 (MB) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU 온도 (°C) |
| `DCGM_FI_DEV_POWER_USAGE` | GPU 전력 사용량 (W) |
| `DCGM_FI_DEV_SM_CLOCK` | SM 클럭 주파수 (MHz) |

**참고:** Tensor Core 사용률 (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`)은 WSL2에서 지원되지 않음

---

### Q: vLLM 부하 테스트는 어떻게 하나?

**A:** Python 스크립트로 동시 요청을 보냅니다.

**기본 부하 테스트:**
```python
# scripts/load_test.py
import requests
import concurrent.futures

def send_request(prompt, request_id):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
        },
    )
    return response.json()

# 동시성 20으로 40개 요청
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(send_request, f"질문 {i}", i) for i in range(40)]
```

**실행:**
```bash
python3 scripts/load_test.py 40 20  # 40개 요청, 동시성 20
```

**결과 예시:**
```
============================================================
성공: 40/40
총 소요 시간: 11.73s
평균 응답 시간: 5.85s
처리량: 646.4 tokens/s
============================================================
```

---

### Q: Grafana 대시보드 프로비저닝 방법?

**A:** ConfigMap으로 대시보드 JSON과 설정을 마운트합니다.

**구조:**
```
manifests/monitoring/
├── grafana-provisioning.yaml  # Datasource + Dashboard ConfigMaps
├── grafana-deployment.yaml    # 볼륨 마운트 설정
└── ...

grafana/
└── vllm-oneview-dashboard.json  # 대시보드 JSON
```

**Grafana Deployment 볼륨 설정:**
```yaml
volumeMounts:
- name: datasources
  mountPath: /etc/grafana/provisioning/datasources
- name: dashboard-provider
  mountPath: /etc/grafana/provisioning/dashboards
- name: dashboards
  mountPath: /var/lib/grafana/dashboards
volumes:
- name: datasources
  configMap:
    name: grafana-datasources
- name: dashboard-provider
  configMap:
    name: grafana-dashboard-provider
- name: dashboards
  configMap:
    name: grafana-dashboard-vllm
```

**배포:**
```bash
kubectl apply -f manifests/monitoring/grafana-provisioning.yaml
kubectl apply -f manifests/monitoring/grafana-deployment.yaml
kubectl rollout restart deployment/grafana -n monitoring
```

---

## GPU 내부 동작 연구 (Phase 3)

### Q: LLM 추론이 Memory-Bound인 이유?

**A:** Decode 단계에서 토큰 하나를 생성할 때마다 모델 전체 가중치를 메모리에서 읽어야 하기 때문입니다.

**LLM 추론의 두 단계:**

| 단계 | 특성 | 이유 |
|------|------|------|
| **Prefill** | Compute-bound | 프롬프트 전체를 한 번에 처리, 행렬 연산 많음 |
| **Decode** | Memory-bound | 토큰 1개당 전체 가중치 로딩 필요 |

**Arithmetic Intensity (연산 강도):**
```
AI = FLOPs / Bytes Transferred

Decode 단계: AI ≈ 1-2 (매우 낮음 → memory-bound)
Prefill 단계: AI ≈ 수십~수백 (높음 → compute-bound)
```

**왜 Decode가 memory-bound인가?**
```
토큰 1개 생성 과정:
1. 모델 가중치 전체를 GPU 메모리에서 읽음 (Qwen 3B = 6GB)
2. 행렬 연산 수행 (매우 빠름)
3. 다음 토큰 생성

→ 연산보다 메모리 읽기가 병목
```

---

### Q: Batch Size가 Memory Bandwidth 효율에 미치는 영향?

**A:** 배치 크기가 커지면 가중치 로딩을 여러 요청이 공유하여 효율이 급격히 상승합니다.

**RTX 5070 Ti 벤치마크 결과 (Qwen2.5-3B):**

| Batch Size | 처리량 (t/s) | 추정 BW | 활용률 |
|------------|--------------|---------|--------|
| 1 | 21.0 | 126 GB/s | 25.0% |
| 2 | 43.2 | 259 GB/s | 51.4% |
| 4 | 90.0 | 540 GB/s | 107.2% |
| 8 | 182.8 | 1097 GB/s | 217.6% |
| 16 | 383.6 | 2301 GB/s | 456.6% |
| 32 | 701.6 | 4209 GB/s | 835.2% |

**100%를 초과하는 이유:**

단순 추정 모델 `throughput × model_size`는 "토큰당 전체 가중치를 읽는다"고 가정하지만, 실제로는:

1. **GPU L2 캐시 효과**: 48MB L2 캐시에 자주 접근하는 가중치 유지
2. **배치 연산 효율**: 큰 배치 → 행렬 연산으로 묶임 → Tensor Core 활용 증가
3. **가중치 재사용**: 배치 내 여러 요청이 동일 가중치 공유

```
Batch=1:  가중치 1번 로딩 → 1 토큰 생성  (비효율)
Batch=32: 가중치 1번 로딩 → 32 토큰 동시 생성 (32배 효율!)
```

**핵심 인사이트:**
- Batch=1: Memory-bound (대역폭 25% 활용)
- Batch=4+: Compute-bound로 전환 (GPU 연산 효율 급상승)
- Batch=32: ~700 t/s로 단일 요청 대비 **33배 처리량 향상**

---

### Q: 토큰 길이가 처리량에 미치는 영향?

**A:** 단일 요청의 Decode 처리량은 토큰 길이와 거의 무관하게 일정합니다 (memory-bound 특성).

**벤치마크 결과:**

| Max Tokens | 실제 생성 | 처리량 (t/s) | 추정 BW |
|------------|----------|--------------|---------|
| 50 | 50 | 31.1 | 186.5 GB/s |
| 100 | 100 | 28.9 | 173.3 GB/s |
| 200 | 200 | 28.8 | 172.9 GB/s |
| 300 | 300 | 28.8 | 172.5 GB/s |
| 500 | 500 | 28.6 | 171.7 GB/s |

**분석:**
- 처리량이 ~29 t/s로 일정 → 전형적인 **memory-bound 패턴**
- 약간의 감소 (31→28, ~8%)는 **KV Cache 크기 증가**로 인한 메모리 접근 오버헤드
- 긴 시퀀스에서 KV Cache 읽기 비용이 추가됨

---

### Q: Memory Bandwidth 최적화의 실무적 의미?

**A:** LLM 서빙에서 배칭은 선택이 아닌 필수입니다.

**이론적 최대 처리량 계산:**
```
RTX 5070 Ti 메모리 대역폭: ~504 GB/s
Qwen2.5-3B 모델 크기: ~6GB (FP16)

단일 요청 이론 최대: 504 / 6 = 84 tokens/s
실제 측정 (Batch=1): 21 tokens/s (25% 활용)
→ 캐시 미스, 오버헤드 등으로 실제 효율 저하
```

**vLLM의 Continuous Batching이 중요한 이유:**
```
요청이 불규칙하게 들어와도:
- 자동으로 진행 중인 배치에 새 요청 추가
- 완료된 요청은 즉시 배치에서 제거
- 항상 최대한의 배치 크기 유지
→ Memory bandwidth 효율 극대화
```

**실무 권장사항:**

| 상황 | 권장 설정 |
|------|----------|
| 개발/테스트 | Batch 1-4 허용, 응답 속도 우선 |
| 프로덕션 (처리량 중시) | 최소 Batch 8+, max_num_seqs 높게 |
| 레이턴시 민감 | Batch 4-8, 적절한 균형점 |

**벤치마크 스크립트:**
```bash
# Batch Size별 처리량 측정
python3 scripts/benchmark_batch.py

# 토큰 길이별 처리량 측정
python3 scripts/benchmark_token_length.py
```

---

### Q: Batch Size 증가 시 가중치 로딩 공유는 어느 단계에서 발생하나?

**A:** Prefill과 Decode 모두에서 발생하지만, **Decode 단계에서 효과가 훨씬 큽니다.**

**Prefill 단계:**
```
Batch=1: [토큰 100개] × 가중치 → 행렬 연산 1번
Batch=4: [토큰 400개] × 가중치 → 행렬 연산 1번 (더 큰 행렬)

→ 이미 토큰이 많아서 행렬이 크고, 원래 compute-bound
→ 배칭 효과 상대적으로 작음
```

**Decode 단계:**
```
Batch=1:  [토큰 1개] × 가중치 → 가중치 6GB 읽고 연산 거의 없음
Batch=32: [토큰 32개] × 가중치 → 가중치 6GB 1번 읽고 32개 동시 연산

→ 메모리 전송 1번으로 32개 토큰 생성
→ "GPU 왔다갔다 횟수가 적다"가 정확한 이해
```

**핵심 원리:**
- 배치 내 모든 요청이 **동일한 가중치**를 사용
- 가중치를 메모리에서 1번 읽으면 여러 요청에 재사용
- Decode는 토큰 1개당 연산이 작아서 메모리 읽기가 병목 → 배칭 효과 극대화

---

### Q: Decode 단계에서 정말 모델 전체 가중치를 읽나?

**A:** 네, 실제로 토큰 1개 생성 시 모든 레이어의 가중치를 순차적으로 읽습니다.

**Transformer Decoder의 토큰 1개 생성 과정:**
```
입력: 이전 토큰의 hidden state [1 × hidden_size]

Layer 1:
  - Q, K, V projection: [hidden × hidden] × 3 → 가중치 읽기
  - Attention 연산
  - FFN: [hidden × 4×hidden] + [4×hidden × hidden] → 가중치 읽기

Layer 2 ~ Layer 36: (동일 과정 반복)

Output projection: [hidden × vocab_size] → 가중치 읽기

→ 모든 레이어의 모든 가중치를 순차적으로 읽어야 함
```

**Qwen2.5-3B 예시:**
```
파라미터 수: 3B = 3,000,000,000개
FP16 크기: 3B × 2 bytes = 6GB

토큰 1개 생성 = 6GB 메모리 읽기 + 작은 연산
```

**Prefill vs Decode 연산량 차이:**
```
Prefill: [100 × hidden] × [hidden × hidden] = 행렬 × 행렬 (연산량 많음)
Decode:  [1 × hidden] × [hidden × hidden]   = 벡터 × 행렬 (연산량 적음)

→ Decode는 같은 가중치를 읽지만 연산이 100배 작음
→ 메모리 읽기가 대부분의 시간을 차지 (memory-bound)
```

---

### Q: Prefill vs Decode 단계의 성능 특성 차이?

**A:** Prefill은 Compute-bound, Decode는 Memory-bound입니다.

**벤치마크 결과 (RTX 5070 Ti + Qwen 3B):**

| Prompt 토큰 | TTFT (Prefill) | Decode 시간 | Prefill 처리량 | Decode 처리량 |
|------------|----------------|-------------|---------------|--------------|
| 168 | 0.068s | 3.53s | 2,461 t/s | 24.9 t/s |
| 823 | 0.058s | 3.36s | 14,071 t/s | 26.6 t/s |
| 3312 | 0.067s | 3.36s | 49,109 t/s | 27.1 t/s |

**핵심 발견:**
- TTFT(Prefill 시간)가 프롬프트 길이와 거의 무관하게 ~0.06초로 일정
- Prefill 처리량: 2,461 → 49,109 t/s (20배 증가)
- Decode 처리량: ~27 t/s로 일정

**원인 분석:**
```
Prefill: 가중치 1번 읽기 → 많은 토큰 병렬 처리
  [1000 × hidden] × [hidden × hidden] = 대규모 행렬 연산
  → GPU 연산 유닛이 바쁘게 일함 (Compute-bound)
  → RTX 5070 Ti가 3B 모델에 충분히 여유 있음

Decode: 가중치 1번 읽기 → 1개 토큰만 처리
  [1 × hidden] × [hidden × hidden] = 벡터-행렬 연산
  → GPU 연산 유닛이 대부분 놀고 있음 (Memory-bound)
  → 메모리 읽기 시간이 지배적
```

---

### Q: Decode 처리량이 프롬프트 길이와 무관하게 일정한 이유?

**A:** 프롬프트가 길어져도 가중치 크기(6GB)는 동일하고, 메모리 대역폭이 병목이기 때문입니다.

**데이터 흐름:**
```
GPU HBM (메모리)           GPU Tensor Core (연산)
┌──────────────┐           ┌──────────────┐
│ 모델 가중치  │ ────────► │   행렬 연산   │
│ (6GB)        │  504GB/s  │   (매우 빠름) │
└──────────────┘   (병목)  └──────────────┘
```

- 토큰 1개 생성 시 가중치 6GB 읽기 필요
- 메모리 대역폭 504 GB/s → 이론 최대 84 t/s
- 실제 ~27 t/s (캐시 미스, 오버헤드 등)

**메모리 대역폭 증가 시 성능:**
```
RTX 5070 Ti (504 GB/s)  → ~27 t/s
A100 (2,039 GB/s)       → ~100+ t/s
H100 (3,350 GB/s)       → ~150+ t/s

결론: 메모리 대역폭 ↑ = Decode 성능 ↑ (정비례)
```

---

### Q: Prefill에서 KV Cache에 무엇을 저장하는가?

**A:** 각 토큰의 **Key(K)와 Value(V) 벡터**를 레이어별로 저장합니다.

**Attention 연산의 Q, K, V:**
```
Q (Query): "다음 토큰이 어떤 정보를 찾는지" → 매번 새로 계산
K (Key):   "이전 토큰들이 어떤 정보인지" → 캐시에 저장
V (Value): "이전 토큰들의 실제 내용"     → 캐시에 저장
```

**Prefill 과정:**
```
입력: "Hello world"

Layer 1:
  Q = input × W_q  (저장 안 함)
  K = input × W_k  → KV Cache에 저장
  V = input × W_v  → KV Cache에 저장

Layer 2 ~ 36: 동일 과정

KV Cache 상태 (Prefill 후):
  Layer 1: K=[K_hello, K_world], V=[V_hello, V_world]
  Layer 2: K=[K_hello, K_world], V=[V_hello, V_world]
  ...
  Layer 36: K=[K_hello, K_world], V=[V_hello, V_world]
```

**왜 K, V만 저장?**
```
Attention(Q, K, V) = softmax(Q × K^T) × V

새 토큰 생성 시:
  Q_new × [K_cache]^T → 이전 토큰들과의 관계 계산
  → K_cache, V_cache는 재사용 (재계산 불필요!)
  → Q는 현재 토큰에만 필요하므로 저장 불필요
```

---

### Q: Decode 출력값이 어디로 전달되는가?

**A:** GPU 내부에서 모든 계산이 완료되고, 최종 **token_id(정수 1개)** 만 CPU로 전달됩니다.

**Decode 단계 데이터 흐름:**
```
GPU 내부:
┌─────────────────────────────────────────────────────────────┐
│ 1. 이전 토큰 임베딩 [1 × hidden]                             │
│         │                                                    │
│         ▼                                                    │
│ 2. Layer 1~36 순차 처리                                      │
│    - Q, K, V 계산 (K, V는 캐시에 추가)                       │
│    - Attention: Q × K_cache^T → softmax → × V_cache         │
│    - FFN: hidden → 4×hidden → hidden                        │
│         │                                                    │
│         ▼                                                    │
│ 3. Output Projection (LM Head)                               │
│    [1 × hidden] × [hidden × vocab] = [1 × vocab]            │
│    → 각 토큰의 확률 분포                                     │
│         │                                                    │
│         ▼                                                    │
│ 4. Sampling (argmax 또는 temperature)                        │
│    → token_id = 12345                                        │
└─────────────────────────────────────────────────────────────┘
         │
         │ (정수 1개만 전송)
         ▼
CPU: token_id → tokenizer.decode() → "안녕" → HTTP 응답
```

**메모리 대역폭이 필요한 곳:**
- 가중치 읽기: HBM → SM (6GB, 504 GB/s) ← 병목!
- KV Cache 읽기/쓰기: HBM ↔ SM
- 최종 출력: token_id 정수 1개만 CPU로 전송 (무시할 수준)

---

### Q: Compute-bound vs Memory-bound 구분 기준은?

**A:** Roofline Model을 기반으로, 배치 크기별 처리량 패턴으로 구분합니다.

**이론적 배경:**
```
처리량 한계 = min(Compute 한계, Memory 한계)

Memory-bound: 메모리 대역폭이 병목
Compute-bound: 연산 능력(FLOPS)이 병목
```

**우리 벤치마크에서 구분법:**

**방법 1: 활용률 패턴**
```
Batch │ 활용률
    1 │  25%   ← 이론 대역폭의 25%만 사용 → memory-bound
    2 │  51%   ← 선형 증가 → memory-bound
    4 │ 107%   ← 100% 초과 → compute 효율 증가 시작
   32 │ 835%   ← 단순 모델로 설명 불가 → compute-bound
```

**방법 2: 배치 증가 대비 처리량 증가율**
```
Batch 1→2:   21 → 43 t/s  (+105%)  ← 거의 2배 = memory-bound
Batch 2→4:   43 → 90 t/s  (+109%)  ← 거의 2배 = memory-bound
Batch 4→8:   90 → 183 t/s (+103%)  ← 전환점
Batch 16→32: 384 → 702 t/s (+83%)  ← 2배 미만 = 포화 (compute-bound)
```

**구분 기준 정리:**

| 특성 | Memory-bound | Compute-bound |
|------|--------------|---------------|
| 활용률 | 100% 미만 | 100% 초과 |
| 배치 2배 시 | 처리량 ~2배 | 처리량 < 2배 |

---

### Q: PagedAttention의 효과가 언제 드러나는가?

**A:** VRAM이 빠듯한 환경(큰 모델, 긴 시퀀스, 높은 동시성)에서 효과가 드러납니다.

**현재 환경 (RTX 5070 Ti + Qwen 3B):**
```
모델 가중치: ~6GB
남은 VRAM: ~10GB
KV Cache 용량: ~200,000 토큰

동시 30개 × 300 토큰 = 9,000 토큰 사용
→ 사용률: ~4.5% (VRAM 여유 많음)
→ PagedAttention 효과 체감 어려움
```

**PagedAttention이 중요해지는 상황:**

| 상황 | 기존 방식 | PagedAttention |
|------|----------|----------------|
| VRAM 여유 있음 (현재) | 문제없음 | 문제없음 |
| **VRAM 빠듯함** | 단편화로 OOM | 효율적 사용 |
| **긴 시퀀스 (8K+)** | 예약 낭비 심함 | 필요한 만큼만 |
| **높은 동시성** | 메모리 부족 | 더 많이 처리 |

**이론적 효율성 비교:**
```
기존 방식 (연속 할당) - max_seq_len=4096 예약:
  Req 1 (실제 500 토큰): 4096 예약
  Req 2 (실제 200 토큰): 4096 예약
  Req 3 (실제 1000 토큰): 4096 예약
  → 총 예약: 12,288 토큰, 실제 사용: 1,700 토큰
  → 효율: 13.8%

PagedAttention (블록 할당) - 16 토큰/블록:
  Req 1: 32 블록 = 512 토큰
  Req 2: 13 블록 = 208 토큰
  Req 3: 63 블록 = 1,008 토큰
  → 총 할당: 1,728 토큰, 실제 사용: 1,700 토큰
  → 효율: 98.4%

→ 7배 차이! VRAM 부족 시 동시 요청 수 7배 차이
```

**결론:**
- 작은 모델 + 여유 VRAM = PagedAttention 효과 미미
- 큰 모델 + 빠듯한 VRAM = PagedAttention 필수 (2-4배 동시 요청 증가)

---

### Q: PagedAttention 스트레스 테스트 결과 (Mistral 7B AWQ)

**A:** VRAM 97% 사용 환경에서 PagedAttention의 동적 블록 할당 효과를 확인했습니다.

**테스트 환경:**
```
모델: Mistral-7B-AWQ (INT4)
GPU: RTX 5070 Ti (16GB, 97% 사용)
max_model_len: 8192
```

**테스트 1: 동시 요청 수 증가**

| 동시 요청 | 토큰/요청 | 총 토큰 | 결과 | 대기 |
|----------|----------|---------|------|------|
| 5 | ~2,600 | 13,000 | ✅ 성공 | 1 |
| 10 | ~2,600 | 26,000 | ✅ 성공 | 0 |
| 15 | ~2,600 | 39,000 | ✅ 성공 | 0 |
| 20 | ~2,600 | 52,000 | ✅ 성공 | 0 |
| 25 | ~2,600 | 65,000 | ✅ 성공 | 0 |
| 30 | ~2,600 | 78,000 | ✅ 성공 | 0 |

**테스트 2: 시퀀스 길이 증가**

| 동시 요청 | 토큰/요청 | 총 토큰 | 결과 |
|----------|----------|---------|------|
| 10 | 1,300 | 13,000 | ✅ 성공 |
| 10 | 2,591 | 25,910 | ✅ 성공 |
| 10 | 5,176 | 51,760 | ✅ 성공 |
| 10 | 7,710 | 77,100 | ✅ 성공 |
| 10 | 9,842 | - | ❌ max_model_len 초과 |

**핵심 발견:**
```
┌────────────────────────────────────────────────────────────┐
│  PagedAttention 효과                                        │
├────────────────────────────────────────────────────────────┤
│  • 동시 30개 요청 → 대기 없이 모두 병렬 처리                 │
│  • 77,000+ 토큰 동시 처리 성공                              │
│  • 실제 사용량 기준 동적 할당으로 동시성 극대화              │
└────────────────────────────────────────────────────────────┘
```

**기존 방식 vs PagedAttention:**
```
기존 방식 (연속 할당):
  - max_seq_len=8192 전체 예약
  - 동시 요청 = 총 KV Cache / 8192
  - 낭비 심함 (실제 사용 50% 미만)

PagedAttention (블록 할당):
  - 실제 사용 토큰만큼만 블록 할당
  - 동시 요청 = 총 KV Cache / 실제 사용량
  - 효율 ~95%, 동시 요청 2-4배 증가
```

**실무적 의미:**
- VRAM이 빠듯한 환경에서 PagedAttention 필수
- 동시 처리량이 서비스 품질(QoS)에 직결
- 가변 길이 요청 환경에서 효과 극대화

---

### Q: AWQ 양자화가 추론 성능에 미치는 영향?

**A:** AWQ(INT4) 양자화는 모델 크기를 줄여 메모리 대역폭 효율을 높이고, 결과적으로 Decode 처리량을 향상시킵니다.

**벤치마크 비교 (RTX 5070 Ti):**

| Batch | Qwen 3B (FP16) | Mistral 7B (AWQ) | 차이 |
|-------|---------------|------------------|------|
| 1 | 21.0 t/s | 31.1 t/s | +48% |
| 2 | 43.2 t/s | 59.1 t/s | +37% |
| 4 | 90.0 t/s | 115.3 t/s | +28% |
| 8 | 182.8 t/s | 231.1 t/s | +26% |
| 16 | 383.6 t/s | 447.8 t/s | +17% |
| 32 | 701.6 t/s | 777.5 t/s | +11% |

**7B AWQ가 3B FP16보다 빠른 이유:**
```
모델 가중치 크기:
  Qwen 3B FP16:   3B × 2 bytes = 6GB
  Mistral 7B AWQ: 7B × 0.5 bytes = 3.5GB  ← 절반!

Decode 처리량 ∝ Memory Bandwidth / 모델 크기
→ 모델 크기가 작으면 메모리 읽기 시간 감소
→ AWQ (INT4)가 FP16 대비 메모리 효율 2배 향상
```

**양자화 방식별 비교:**

| 방식 | 비트 | 모델 크기 (7B 기준) | 성능 손실 |
|------|-----|-------------------|----------|
| FP16 | 16bit | 14GB | 없음 |
| INT8 | 8bit | 7GB | ~1% |
| **AWQ** | 4bit | 3.5GB | ~2-3% |
| GPTQ | 4bit | 3.5GB | ~3-5% |

**핵심 인사이트:**
```
파라미터 수 ≠ 추론 속도

더 큰 모델(7B)이 더 작은 모델(3B)보다 빠를 수 있음
→ 양자화로 가중치 크기를 줄이면
→ 메모리 대역폭 효율 증가
→ Decode 처리량 향상
```

**실무 권장:**
- VRAM 여유 있음: FP16 (최고 품질)
- VRAM 빠듯함: AWQ/GPTQ (품질 유지하며 속도 향상)
- 배포 환경: AWQ 권장 (품질/속도 균형 최적)

---

## Python / 프로그래밍

### Q: concurrent.futures 사용법?

**A:** Python 표준 라이브러리의 동시성 도구로, 스레드/프로세스 풀을 쉽게 사용할 수 있습니다.

**기본 사용법:**
```python
import concurrent.futures

# ThreadPoolExecutor: I/O 바운드 작업 (네트워크, 파일)
# ProcessPoolExecutor: CPU 바운드 작업 (연산)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 방법 1: submit() - 개별 작업 제출
    future = executor.submit(함수, 인자1, 인자2)
    result = future.result()  # 결과 대기

    # 방법 2: map() - 여러 작업 일괄 제출
    results = executor.map(함수, [인자1, 인자2, 인자3])
```

**우리 벤치마크 코드:**
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    # batch_size개의 HTTP 요청을 동시에 제출
    futures = [executor.submit(send_request, i + 1) for i in range(batch_size)]

    # 완료되는 순서대로 결과 수집 (먼저 끝난 것부터)
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)
```

**동작 흐름:**
```
시간 →
Thread 1: ──[send_request]────────► 완료
Thread 2: ──[send_request]──────────────► 완료
Thread 3: ──[send_request]────► 완료 (먼저 끝남)
Thread 4: ──[send_request]──────► 완료

as_completed()는 완료 순서대로 반환: Thread 3 → 1 → 4 → 2
```

---

### Q: Python f-string 포맷팅 문법?

**A:** `f"{값:포맷}"` 형식으로 정렬, 너비, 소수점 등을 지정합니다.

**기본 문법:**
```python
f"{값:정렬너비.소수점f}"

# 정렬: > (오른쪽), < (왼쪽), ^ (가운데)
# 너비: 총 칸 수
# .소수점f: float 소수점 자릿수
```

**예시:**
```python
value = 42
f"{value:>8}"      # "      42" (8칸, 오른쪽 정렬)
f"{value:<8}"      # "42      " (8칸, 왼쪽 정렬)
f"{value:^8}"      # "   42   " (8칸, 가운데 정렬)

value = 3.14159
f"{value:.2f}"     # "3.14" (소수점 2자리)
f"{value:>10.2f}"  # "      3.14" (10칸, 소수점 2자리)
f"{value:>10.0f}"  # "         3" (10칸, 소수점 0자리 = 정수)
```

**우리 코드:**
```python
f"{r['max_tokens']:>12}"              # 12칸, 오른쪽 정렬
f"{r['avg_completion_tokens']:>8.0f}" # 8칸, 소수점 없이
f"{r['avg_decode_throughput']:>10.1f}" # 10칸, 소수점 1자리

# 출력 결과:
#   Max Tokens |   Actual |   Throughput |    Est. BW
#           50 |       50 |       31.1 t/s |    186.5 GB/s
```

---

### Q: 벤치마크에서 completion_tokens가 Decode 처리량인 이유?

**A:** Prefill은 토큰을 "생성"하지 않고, completion_tokens는 Decode에서만 생성되기 때문입니다.

**vLLM 처리 과정:**
```
요청: "Python으로 피보나치 설명해줘"

1. Prefill 단계:
   - 입력 프롬프트 처리 (100 토큰)
   - 출력: 없음 (KV Cache만 생성)

2. Decode 단계:
   - 토큰 1개씩 생성: "def" → " fib" → "(" → ...
   - 출력: completion_tokens (100개 생성)

API 응답:
{
  "usage": {
    "prompt_tokens": 100,      ← Prefill에서 "처리"
    "completion_tokens": 100   ← Decode에서 "생성"
  }
}
```

**코드에서:**
```python
generation_throughput = total_completion_tokens / total_time
```

- `completion_tokens` = Decode에서 생성된 토큰 수
- `total_time`에는 Prefill 시간도 포함되어 있어 정확한 Decode-only 처리량은 아님
- 하지만 대부분의 시간이 Decode이므로 근사치로 유효함

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-10 | 초기 작성 (Traefik, /dev/shm) |
| 2025-01-10 | vLLM 트러블슈팅 추가 (enforce-eager, OOM, Recreate) |
| 2025-01-11 | KV Cache, PagedAttention, Continuous Batching 원리 추가 |
| 2025-01-11 | vLLM/GPU 모니터링 설정 및 부하 테스트 방법 추가 |
| 2025-01-11 | Phase 3: Memory Bandwidth 분석 결과 추가 |
| 2025-01-11 | 심화 Q&A 추가 (가중치 공유, Compute/Memory-bound, Python 문법) |
| 2025-01-11 | Prefill vs Decode 분석 결과 및 KV Cache 동작 원리 추가 |
| 2025-01-11 | PagedAttention 효과 분석 및 적용 조건 추가 |
| 2025-01-11 | AWQ 양자화 성능 비교 (Qwen 3B FP16 vs Mistral 7B AWQ) 추가 |
| 2025-01-11 | PagedAttention 스트레스 테스트 결과 (Mistral 7B AWQ) 추가 |
