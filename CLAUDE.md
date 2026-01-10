# vLLM GPU 인프라 연구 프로젝트

## 프로젝트 개요

WSL AlmaLinux 9 환경에서 k3s 기반 Kubernetes 클러스터를 구성하고, RTX 5070 Ti GPU를 활용한 vLLM 모델 서빙 및 GPU 내부 동작 연구를 수행한다.

## 환경 정보

- **OS**: WSL2 + AlmaLinux 9
- **GPU**: NVIDIA RTX 5070 Ti (Blackwell, 16GB VRAM, SM 120)
- **Kubernetes**: k3s (직접 설치, v1.34.3+k3s1)
- **Container Runtime**: containerd + nvidia-container-runtime
- **NVIDIA Container Toolkit**: 1.18.1+
- **LLM Serving**: vLLM
- **CUDA**: 13.1 (Driver 591.59)
- **GitHub**: https://github.com/donghyun/vllm-study (TODO: 실제 URL로 변경)

## Git 브랜치 전략

- **main**: 안정 버전, 검증된 코드만 병합
- **develop**: 개발 브랜치, 기능 개발 후 main으로 PR
- **feature/***: 기능별 브랜치 (예: feature/vllm-deploy)

## 커밋 메시지 규칙

```
<type>: <subject>

<body>
```

**Type:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가
- `chore`: 기타 (설정, 빌드 등)

**예시:**
```
feat: vLLM Kubernetes 배포 매니페스트 추가

- PVC, Deployment, Service 매니페스트 작성
- Qwen2.5-7B-Instruct 모델 설정
```

---

## Phase 1: 환경 확인 및 준비

### 1.1 GPU 환경 확인

```
작업 유형: 환경 확인
실행 방식: Claude Skills 활용
```

> **Note**: WSL AlmaLinux 9 및 k3d 클러스터는 이미 구성되어 있음.

**체크리스트:**
- [ ] nvidia-smi로 GPU 인식 확인 (RTX 5070 Ti)
- [ ] CUDA 버전 확인 (12.8+ 필요 for Blackwell)
- [ ] NVIDIA Container Toolkit 동작 확인
- [ ] k3d 클러스터 GPU 연동 상태 확인

**검증:**
```bash
nvidia-smi  # GPU 정보 출력 확인
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubi9 nvidia-smi  # 컨테이너 GPU 접근 확인
```

### 1.2 k3s 클러스터 구성 (완료)

```
작업 유형: 인프라 구축
실행 방식: Claude Skills
상태: ✅ 완료
```

#### k3s 설치
```bash
# k3s 설치 (SELinux 경고 무시, traefik 비활성화)
curl -sfL https://get.k3s.io | INSTALL_K3S_SKIP_SELINUX_RPM=true INSTALL_K3S_SELINUX_WARN=true INSTALL_K3S_EXEC="--disable=traefik" sh -

# kubeconfig 설정
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
```

#### NVIDIA Container Toolkit 설정
```bash
# nvidia-container-toolkit 1.18+ 설치 (repo 추가 필요 시)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit

# containerd에 nvidia 런타임 추가
sudo nvidia-ctk runtime configure --runtime=containerd
```

#### k3s containerd 템플릿 설정 (필수)
```bash
# /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl 생성
sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl << 'EOF'
{{ template "base" . }}

[plugins.'io.containerd.cri.v1.runtime'.containerd]
  default_runtime_name = "nvidia"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = true
EOF

# k3s 재시작
sudo systemctl restart k3s
```

#### NVIDIA Device Plugin 설치
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

#### 검증
```bash
# 클러스터 상태 확인
kubectl get nodes -o wide

# GPU 리소스 확인 (nvidia.com/gpu: 1 표시되어야 함)
kubectl describe node | grep -A 10 "Capacity:"

# GPU 테스트 Pod 실행
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:12.8.0-base-ubi9
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# 결과 확인
kubectl logs gpu-test
```

---

## Phase 2: 모델 선정 및 vLLM 배포

### 2.1 RTX 5070 Ti 적합 모델 선정

```
작업 유형: 분석 및 의사결정
실행 방식: Plan Mode
```

**하드웨어 제약:**
- VRAM: 16GB
- Compute Capability: sm_120 (Blackwell)
- Memory Bandwidth: ~504 GB/s

**권장 모델 (우선순위순):**

| 모델 | 파라미터 | VRAM 요구량 | 특징 |
|------|----------|-------------|------|
| **Qwen2.5-7B-Instruct** | 7B | ~14GB (FP16) | 다국어 지원, 성능 우수 |
| **Mistral-7B-Instruct-v0.3** | 7B | ~14GB (FP16) | 추론 성능 우수 |
| **Llama-3.1-8B-Instruct** | 8B | ~16GB (FP16) | Meta 공식, 범용성 |
| **Phi-3-medium-4k-instruct** | 14B | ~14GB (INT4) | 양자화 시 적합 |
| **gemma-2-9b-it** | 9B | ~12GB (BF16) | Google, 효율적 |

**1차 선정: Qwen2.5-7B-Instruct**
- 이유: 16GB VRAM에서 안정적 동작, 한국어 성능 우수, vLLM 호환성 검증됨

### 2.2 vLLM Kubernetes 배포

```
작업 유형: 복합 배포 작업
실행 방식: 서브에이전트 활용 (Helm 차트 생성, 매니페스트 작성 분담)
```

**배포 구성요소:**
1. PersistentVolumeClaim (모델 캐시)
2. vLLM Deployment
3. Service (ClusterIP/NodePort)
4. HPA (선택적)

**vLLM 배포 매니페스트:**
```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
    namespace: vllm
    spec:
      replicas: 1
        selector:
            matchLabels:
                  app: vllm-qwen
                    template:
                        metadata:
                              labels:
                                      app: vllm-qwen
                                          spec:
                                                containers:
                                                      - name: vllm
                                                              image: vllm/vllm-openai:latest
                                                                      args:
                                                                                - "--model"
                                                                                          - "Qwen/Qwen2.5-7B-Instruct"
                                                                                                    - "--tensor-parallel-size"
                                                                                                              - "1"
                                                                                                                        - "--gpu-memory-utilization"
                                                                                                                                  - "0.90"
                                                                                                                                            - "--max-model-len"
                                                                                                                                                      - "8192"
                                                                                                                                                              resources:
                                                                                                                                                                        limits:
                                                                                                                                                                                    nvidia.com/gpu: 1
                                                                                                                                                                                            ports:
                                                                                                                                                                                                    - containerPort: 8000
                                                                                                                                                                                                            env:
                                                                                                                                                                                                                    - name: HF_TOKEN
                                                                                                                                                                                                                              valueFrom:
                                                                                                                                                                                                                                          secretKeyRef:
                                                                                                                                                                                                                                                        name: hf-secret
                                                                                                                                                                                                                                                                      key: token
                                                                                                                                                                                                                                                                              volumeMounts:
                                                                                                                                                                                                                                                                                      - name: model-cache
                                                                                                                                                                                                                                                                                                mountPath: /root/.cache/huggingface
                                                                                                                                                                                                                                                                                                      volumes:
                                                                                                                                                                                                                                                                                                            - name: model-cache
                                                                                                                                                                                                                                                                                                                    persistentVolumeClaim:
                                                                                                                                                                                                                                                                                                                              claimName: vllm-model-cache
                                                                                                                                                                                                                                                                                                                              ```

                                                                                                                                                                                                                                                                                                                              **검증 피드백 루프:**
                                                                                                                                                                                                                                                                                                                              ```bash
# Step 1: Pod 상태 확인
                                                                                                                                                                                                                                                                                                                              kubectl get pods -n vllm -w

# Step 2: 로그 모니터링
kubectl logs -f -n vllm deployment/vllm-qwen

# Step 3: API 테스트
kubectl port-forward -n vllm svc/vllm-qwen 8000:8000 &
curl http://localhost:8000/v1/models

# Step 4: 추론 테스트
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Hello, ", "max_tokens": 50}'

# 문제 발생 시: OOM → gpu-memory-utilization 감소 또는 max-model-len 축소
```

---

## Phase 3: GPU 내부 동작 연구

### 3.1 연구 주제 선정

```
작업 유형: 연구 계획 수립
실행 방식: Plan Mode → 문헌 조사 서브에이전트 활용
```

**연구 가능 주제:**

#### 주제 1: CUDA Kernel 최적화 분석
- **목표**: vLLM의 핵심 CUDA 커널 동작 방식 이해
- **범위**: Attention kernel, Paged KV-Cache 구현
- **도구**: Nsight Compute, CUDA Profiler

#### 주제 2: NCCL 통신 패턴 분석 (단일 GPU 한계, 추후 확장)
- **목표**: 멀티 GPU 환경에서의 통신 오버헤드 측정
- **범위**: All-Reduce, Ring-AllReduce 패턴
- **현실**: 단일 GPU 환경이므로 시뮬레이션 또는 문헌 연구 중심

#### 주제 3: Memory Bandwidth 최적화 연구 ⭐ (권장)
- **목표**: LLM 추론에서 메모리 병목 현상 분석 및 최적화
- **배경**: LLM 추론은 compute-bound가 아닌 memory-bound
- **실험**: 다양한 batch size에서 메모리 대역폭 활용률 측정

#### 주제 4: Blackwell 아키텍처 특성 연구 ⭐ (권장)
- **목표**: RTX 5070 Ti (sm_120) 고유 특성 파악
- **범위**: FP8 연산, 새로운 Tensor Core 기능
- **실험**: FP16 vs FP8 추론 성능 비교

#### 주제 5: PagedAttention 메모리 효율성 분석
- **목표**: vLLM의 PagedAttention이 메모리 단편화를 줄이는 방식 이해
- **실험**: 다양한 시퀀스 길이에서 메모리 사용 패턴 분석

### 3.2 연구 환경 구축

```
작업 유형: 도구 설치 및 설정
실행 방식: Claude Skills
```

**필수 도구:**
```bash
# NVIDIA Nsight Systems (시스템 레벨 프로파일링)
# NVIDIA Nsight Compute (커널 레벨 프로파일링)
# 설치는 NVIDIA 개발자 사이트에서 다운로드

# Python 프로파일링 도구
pip install py-spy nvitop gpustat

# CUDA 샘플 및 도구
git clone https://github.com/NVIDIA/cuda-samples.git
```

**프로파일링 환경 설정:**
```bash
# nsys 프로파일링 실행 예시
nsys profile -o vllm_profile python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct

# ncu 커널 분석 예시 (특정 커널)
ncu --set full -o kernel_analysis python inference_script.py
```

### 3.3 연구 실험 계획

```
작업 유형: 실험 설계
실행 방식: Plan Mode
```

#### 실험 1: Batch Size별 성능 분석

**가설**: Batch size 증가 시 memory bandwidth 활용률이 향상되어 처리량 증가

**실험 설계:**
```python
# benchmark_batch.py
batch_sizes = [1, 2, 4, 8, 16, 32]
metrics = ['throughput_tokens_per_sec', 'latency_p50', 'latency_p99', 'gpu_memory_used', 'memory_bandwidth_utilization']

for batch_size in batch_sizes:
        # vLLM 벤치마크 실행
            # Nsight로 프로파일링
                # 메트릭 수집 및 저장
                ```

                **검증 피드백 루프:**
                1. 실험 실행
                2. 결과 수집
                3. 이상치 확인
                4. 필요 시 재실험
                5. 결과 분석 및 시각화

#### 실험 2: KV-Cache 메모리 패턴 분석

**가설**: PagedAttention은 연속적 메모리 할당 대비 단편화를 50% 이상 감소

**실험 설계:**
```python
# analyze_kv_cache.py
sequence_lengths = [512, 1024, 2048, 4096, 8192]
concurrent_requests = [1, 2, 4, 8]

for seq_len in sequence_lengths:
        for num_requests in concurrent_requests:
                    # 메모리 할당 패턴 추적
                            # 단편화 정도 측정
                                    # Peak memory vs Actual usage 비교
                                    ```

#### 실험 3: Attention Kernel 성능 분석

**목표**: Flash Attention vs Standard Attention 성능 차이 정량화

**실험 설계:**
```bash
# Nsight Compute로 attention 커널 분석
ncu --kernel-name ".*attention.*" --set full -o attention_analysis \
    python run_inference.py
    ```

    **분석 메트릭:**
    - Achieved Occupancy
    - Memory Throughput
    - Compute Throughput
    - SM Efficiency

    ---

## Phase 4: 모니터링 및 관측성

### 4.1 GPU 메트릭 수집

```
작업 유형: 모니터링 구축
실행 방식: 서브에이전트 활용 (Prometheus/Grafana 배포)
```

**구성요소:**
1. DCGM Exporter (GPU 메트릭)
2. Prometheus (메트릭 수집)
3. Grafana (시각화)

**DCGM Exporter 배포:**
```yaml
# dcgm-exporter.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
    namespace: monitoring
    spec:
      selector:
          matchLabels:
                app: dcgm-exporter
                  template:
                      metadata:
                            labels:
                                    app: dcgm-exporter
                                        spec:
                                              containers:
                                                    - name: dcgm-exporter
                                                            image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
                                                                    ports:
                                                                            - containerPort: 9400
                                                                                    securityContext:
                                                                                              privileged: true
                                                                                                      volumeMounts:
                                                                                                              - name: device-plugin
                                                                                                                        mountPath: /var/lib/kubelet/device-plugins
                                                                                                                              volumes:
                                                                                                                                    - name: device-plugin
                                                                                                                                            hostPath:
                                                                                                                                                      path: /var/lib/kubelet/device-plugins
                                                                                                                                                      ```

                                                                                                                                                      **수집 메트릭:**
                                                                                                                                                      - GPU Utilization (%)
                                                                                                                                                      - GPU Memory Used/Free
                                                                                                                                                      - GPU Temperature
                                                                                                                                                      - Power Usage
                                                                                                                                                      - SM Clock/Memory Clock
                                                                                                                                                      - Tensor Core Utilization

### 4.2 vLLM 메트릭 수집

**vLLM Prometheus 메트릭:**
```
vllm:num_requests_running
vllm:num_requests_waiting
vllm:gpu_cache_usage_perc
vllm:cpu_cache_usage_perc
vllm:avg_prompt_throughput_toks_per_s
vllm:avg_generation_throughput_toks_per_s
```

---

## Phase 5: 연구 결과 정리

### 5.1 문서화

```
작업 유형: 문서 작성
실행 방식: Claude Skills
```

**산출물:**
1. 환경 구축 가이드
2. 실험 결과 리포트
3. 성능 벤치마크 데이터
4. 최적화 권장사항

### 5.2 향후 연구 방향

- 멀티 GPU 환경 확장 (NCCL 통신 연구)
- 양자화 (AWQ, GPTQ) 성능 비교
- Speculative Decoding 구현 및 분석
- Continuous Batching 최적화

---

## 작업 지침

### Claude Code 활용 가이드

#### 서브에이전트 활용 시나리오

1. **인프라 구축 작업**
   - Main: 전체 아키텍처 설계
      - Sub-agent 1: Kubernetes 매니페스트 작성
         - Sub-agent 2: 모니터링 스택 배포
            - Sub-agent 3: 네트워크 정책 설정

            2. **실험 실행 작업**
               - Main: 실험 계획 및 결과 분석
                  - Sub-agent 1: 벤치마크 스크립트 작성
                     - Sub-agent 2: 프로파일링 데이터 수집
                        - Sub-agent 3: 결과 시각화

                        3. **문서화 작업**
                           - Main: 전체 구조 설계
                              - Sub-agent 1: 기술 문서 작성
                                 - Sub-agent 2: 다이어그램 생성
                                    - Sub-agent 3: README 작성

#### Claude Skills 활용 대상

- 패키지 설치 명령어 실행
- 설정 파일 생성 및 수정
- Git 작업 (commit, push)
- 로그 확인 및 기본 트러블슈팅
- 반복적인 kubectl 명령어 실행

#### Plan Mode 활용 시점

1. **새로운 Phase 시작 시**
   - 목표 명확화
      - 필요 리소스 파악
         - 예상 문제점 식별
            - 롤백 계획 수립

            2. **복잡한 작업 시작 전**
               - 의존성 분석
                  - 작업 순서 결정
                     - 병렬화 가능 작업 식별

                     3. **문제 해결 시**
                        - 증상 정리
                           - 가능한 원인 나열
                              - 진단 순서 결정

### 검증 피드백 루프

```
┌─────────────────────────────────────────────────────────┐
│                    작업 실행                              │
└─────────────────────────────────────────────────────────┘
                          │
                                                    ▼
                                                    ┌─────────────────────────────────────────────────────────┐
                                                    │                    검증 단계                              │
                                                    │  - 예상 결과와 실제 결과 비교                              │
                                                    │  - 에러 로그 확인                                         │
                                                    │  - 메트릭 정상 범위 확인                                   │
                                                    └─────────────────────────────────────────────────────────┘
                                                                              │
                                                                                            ┌───────────┴───────────┐
                                                                                                          │                       │
                                                                                                                        ▼                       ▼
                                                                                                                               ┌──────────┐           ┌──────────┐
                                                                                                                                      │  성공    │           │  실패    │
                                                                                                                                             └──────────┘           └──────────┘
                                                                                                                                                           │                       │
                                                                                                                                                                         ▼                       ▼
                                                                                                                                                                                ┌──────────┐           ┌──────────────────┐
                                                                                                                                                                                       │ 다음 단계 │           │ 원인 분석        │
                                                                                                                                                                                              └──────────┘           │ - 로그 상세 확인 │
                                                                                                                                                                                                                            │ - 리소스 상태    │
                                                                                                                                                                                                                                                          │ - 네트워크 확인  │
                                                                                                                                                                                                                                                                                        └──────────────────┘
                                                                                                                                                                                                                                                                                                                              │
                                                                                                                                                                                                                                                                                                                                                                    ▼
                                                                                                                                                                                                                                                                                                                                                                                                  ┌──────────────────┐
                                                                                                                                                                                                                                                                                                                                                                                                                                │ 수정 후 재실행   │
                                                                                                                                                                                                                                                                                                                                                                                                                                                              └──────────────────┘
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    │
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          └──────► (검증 단계로)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ```

### 각 Phase별 검증 기준

| Phase | 검증 항목 | 성공 기준 | 상태 |
|-------|----------|----------|------|
| 1.1 | GPU 인식 | nvidia-smi 출력 정상, CUDA 13.1 | ✅ |
| 1.2 | k3s GPU 연동 | kubectl describe node에 nvidia.com/gpu: 1 표시 | ✅ |
| 2.1 | 모델 로드 | vLLM 로그에 "Model loaded" | ⏳ |
| 2.2 | API 응답 | /v1/models 200 OK | ⏳ |
| 3.x | 프로파일링 | nsys/ncu 리포트 생성 | ⏳ |
| 4.x | 모니터링 | Grafana 대시보드 메트릭 표시 | ⏳ |

---

## 참고 자료

### 공식 문서
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA Nsight](https://developer.nvidia.com/nsight-systems)
- [k3s Documentation](https://docs.k3s.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

### 연구 논문
- PagedAttention: Efficient Memory Management for Large Language Model Serving
- Flash Attention: Fast and Memory-Efficient Exact Attention
- NCCL: Optimized Primitives for Collective Multi-GPU Communication

### 관련 GitHub
- https://github.com/vllm-project/vllm
- https://github.com/NVIDIA/cuda-samples
- https://github.com/NVIDIA/nccl

---

## 학습 노트 작성 규칙

프로젝트 진행 중 사용자가 질문하는 내용은 `docs/learning-notes.md`에 Q&A 형식으로 정리한다.

**정리 대상:**
- 개념 질문 (예: "/dev/shm이 뭐야?")
- 설정 이유 질문 (예: "왜 traefik을 비활성화해?")
- 트러블슈팅 과정에서 배운 내용
- 아키텍처/설계 관련 질문

**형식:**
```markdown
### Q: 질문 내용

**A:** 답변 요약

- 상세 설명
- 코드/명령어 예시
- 관련 참고 자료
```

**파일 위치:** `docs/learning-notes.md`

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-01-09 | 1.0 | 초기 계획 수립 |
| 2025-01-10 | 1.1 | k3d → k3s 직접 설치로 변경, GPU 연동 완료 |
| 2025-01-10 | 1.2 | 학습 노트 작성 규칙 추가 |
