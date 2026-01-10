# vLLM / PyTorch / CUDA 내부 동작 원리

이 문서는 vLLM의 내부 구조와 GPU 연산의 흐름을 정리합니다.

---

## 목차

1. [vLLM 아키텍처 개요](#1-vllm-아키텍처-개요)
2. [Custom CUDA Kernels](#2-custom-cuda-kernels)
3. [PyTorch의 역할](#3-pytorch의-역할)
4. [CUDA / cuDNN / cuBLAS](#4-cuda--cudnn--cublas)
5. [torch.compile과 --enforce-eager](#5-torchcompile과---enforce-eager)
6. [FX Graph vs CUDA Graph](#6-fx-graph-vs-cuda-graph)
7. [부동소수점 포맷 (FP8, FP16, BF16, TF32)](#7-부동소수점-포맷)
8. [GPU 하드웨어 아키텍처](#8-gpu-하드웨어-아키텍처)
9. [참고 문서](#9-참고-문서)

---

## 1. vLLM 아키텍처 개요

### 계층 구조

```
┌─────────────────────────────────────────────────────────┐
│                        vLLM                             │
├─────────────────────────────────────────────────────────┤
│  Custom CUDA Kernels (PagedAttention, FlashAttention)   │
├─────────────────────────────────────────────────────────┤
│  PyTorch (텐서 연산, 모델 로딩, 메모리 관리)               │
├─────────────────────────────────────────────────────────┤
│  CUDA / cuDNN / cuBLAS                                  │
├─────────────────────────────────────────────────────────┤
│  NVIDIA Driver                                          │
└─────────────────────────────────────────────────────────┘
```

### 호출 흐름

```
1. API 요청 (curl → vLLM 서버)
       ↓
2. Python 코드 (vllm/attention/backends/...)
       ↓
3. PyTorch 확장 호출 (torch.ops.vllm.paged_attention)
       ↓
4. C++ 바인딩 (csrc/*.cu)
       ↓
5. CUDA 커널 실행 (kernel<<<grid, block>>>(...))
       ↓
6. GPU 하드웨어 (SM에서 병렬 실행)
```

---

## 2. Custom CUDA Kernels

### 정의

vLLM의 `csrc/` 폴더에 있는 `.cu` 파일들이 Custom CUDA Kernel입니다.

```
vllm/csrc/
├── attention/
│   ├── paged_attention_v1.cu    ← PagedAttention 구현
│   ├── paged_attention_v2.cu
│   └── attention_kernels.cuh    ← 헤더 파일
├── quantization/
│   ├── awq/gemm_kernels.cu      ← 양자화 GEMM
│   └── gptq/
└── ops.cu                        ← 기타 연산
```

### 왜 필요한가?

**PyTorch 기본 연산의 한계:**

```python
# PyTorch 방식 - 4번의 GPU 커널 호출
def attention_pytorch(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))  # 호출 1
    scores = scores / math.sqrt(d_k)               # 호출 2
    attn = torch.softmax(scores, dim=-1)           # 호출 3
    output = torch.matmul(attn, V)                 # 호출 4
    return output
```

```cpp
// Custom Kernel - 1번의 GPU 커널 호출
__global__ void fused_attention_kernel(...) {
    // Q*K^T, scale, softmax, *V를 한 번에 처리
    // 중간 결과를 레지스터/shared memory에만 저장
}
```

### Custom Kernel의 장점

| 항목 | PyTorch 기본 | Custom Kernel |
|------|-------------|---------------|
| GPU 호출 횟수 | 다수 | 최소화 |
| 메모리 사용 | 중간 결과 저장 | 레지스터 활용 |
| 최적화 수준 | 범용 | LLM 특화 |

### PyTorch가 중간에 있는 이유

Custom Kernel도 **PyTorch의 인프라를 활용**합니다:

- **Tensor 추상화**: shape, dtype, device 관리
- **메모리 관리**: GPU 메모리 할당/해제 (Caching Allocator)
- **데이터 교환**: Python ↔ C++ ↔ CUDA 간 데이터 전달

```cpp
// Custom kernel은 PyTorch 텐서를 입력받음
void paged_attention(
    torch::Tensor& out,      // PyTorch 텐서
    torch::Tensor& query,    // PyTorch 텐서
    ...
) {
    // raw 포인터 추출 후 CUDA 커널 실행
    float* out_ptr = out.data_ptr<float>();
    paged_attention_kernel<<<grid, block>>>(out_ptr, ...);
}
```

---

## 3. PyTorch의 역할

### Tensor 클래스

다차원 배열 + 메타데이터를 제공합니다.

```python
x = torch.tensor([[1, 2], [3, 4]], device='cuda')

# 내부 구조
x.data_ptr()  # GPU 메모리 주소
x.shape       # (2, 2)
x.dtype       # torch.int64
x.device      # cuda:0
x.stride()    # 메모리 레이아웃
```

### GPU 메모리 관리 (Caching Allocator)

```
일반 방식:
  할당 → 사용 → cudaFree → cudaMalloc → 사용 (느림)

PyTorch 방식:
  할당 → 사용 → 캐시 반환 → 캐시에서 재사용 (빠름)
```

- `cudaMalloc`/`cudaFree` 호출 최소화
- 해제된 메모리를 캐시에 보관
- 같은 크기 요청 시 재사용
- 메모리 단편화 방지

---

## 4. CUDA / cuDNN / cuBLAS

### CUDA (Compute Unified Device Architecture)

GPU 프로그래밍의 기본 플랫폼입니다.

| 구성 요소 | 설명 |
|----------|------|
| CUDA Driver API | 저수준 GPU 제어 (libcuda.so) |
| CUDA Runtime API | 고수준 API (libcudart.so) |
| NVCC | CUDA C++ 컴파일러 |

**NVCC 컴파일 과정:**

```
.cu 파일 (CUDA C++ 소스)
       ↓ nvcc
PTX (중간 표현, Parallel Thread Execution)
       ↓ ptxas
SASS (GPU 기계어)
       ↓
GPU 실행
```

**일반 컴파일러 vs NVCC:**

| 항목 | g++ | nvcc |
|------|-----|------|
| 입력 | .cpp | .cu |
| 타겟 | CPU (x86, ARM) | GPU (SM) |
| 출력 | 실행파일 | PTX + 호스트 코드 |
| 병렬화 | 수동 | 자동 (수천 스레드) |

### cuBLAS (Basic Linear Algebra Subprograms)

행렬 연산 최적화 라이브러리입니다.

```
주요 연산:
├── GEMM: General Matrix Multiply (C = αAB + βC)
├── GEMV: Matrix-Vector Multiply
└── AXPY: Vector Addition (y = αx + y)
```

**LLM에서의 역할:**
- Linear Layer (y = Wx + b) → cuBLAS GEMM 호출
- Tensor Core 자동 활용

### cuDNN (Deep Neural Network library)

딥러닝 연산 최적화 라이브러리입니다.

```
주요 연산:
├── Convolution (CNN)
├── RNN/LSTM/GRU
├── Normalization (BatchNorm, LayerNorm)
├── Activation (ReLU, GELU, Softmax)
└── Pooling
```

### 실제 동작 흐름

```python
output = model(input_ids)
```

```
Python (model.forward)
       ↓
PyTorch (torch.nn.Linear, F.softmax 등)
       ↓
ATen (C++ 텐서 라이브러리)
       ↓
┌─────────────────────────────────┐
│ 연산 종류에 따라 분기             │
├─────────────────────────────────┤
│ 행렬 곱셈 → cuBLAS              │
│ Softmax → cuDNN 또는 custom     │
│ Attention → vLLM custom kernel  │
└─────────────────────────────────┘
       ↓
CUDA Driver → GPU Hardware
```

---

## 5. torch.compile과 --enforce-eager

### torch.compile이란?

PyTorch 2.0에서 도입된 JIT(Just-In-Time) 컴파일러입니다.

**Eager Mode (기존):**
```python
# 한 줄씩 즉시 실행
x = torch.add(a, b)      # GPU 호출
y = torch.mul(x, c)      # GPU 호출
z = torch.relu(y)        # GPU 호출
# → 3번의 커널 호출, 매번 Python ↔ GPU 왕복
```

**Compiled Mode:**
```python
@torch.compile
def forward(a, b, c):
    x = torch.add(a, b)
    y = torch.mul(x, c)
    z = torch.relu(y)
    return z
# → 전체를 분석해서 하나의 최적화된 커널로 융합
```

### torch.compile 내부 동작

```
Python 코드
    ↓ TorchDynamo
FX Graph (중간 표현)
    ↓ 최적화 pass
최적화된 FX Graph
    ↓ Triton / CUDA 코드 생성
    ↓ C++ 컴파일러 (g++, nvcc)  ← 여기서 -lcuda 필요
실행 가능한 바이너리
```

### -lcuda 에러

```
subprocess.CalledProcessError: Command '['which', 'c++']' returned non-zero
CUDA runtime error: -lcuda not found
```

**원인:** vLLM 컨테이너에 C++ 컴파일러(g++)가 없음

**해결:** `--enforce-eager` 플래그로 torch.compile 비활성화

### --enforce-eager

torch.compile을 비활성화하고 Eager Mode로 강제 실행합니다.

| 항목 | Eager Mode | Compiled Mode |
|------|------------|---------------|
| 시작 시간 | 빠름 | 느림 (컴파일) |
| 추론 속도 | 보통 | 10-30% 빠름 |
| 환경 의존성 | 낮음 | 높음 (컴파일러 필요) |

**vLLM에서는 --enforce-eager로 충분한 이유:**
- vLLM은 이미 PagedAttention 등 핵심 부분을 Custom Kernel로 최적화
- torch.compile의 추가 이점이 제한적

---

## 6. FX Graph vs CUDA Graph

### FX Graph

Python 레벨의 **연산 그래프 표현**입니다. torch.compile의 중간 표현(IR)으로 사용됩니다.

```python
# 원본 코드
def forward(x, y):
    a = torch.add(x, y)
    b = torch.relu(a)
    return b

# FX Graph로 변환
"""
graph():
    %x : Tensor = placeholder[target=x]
    %y : Tensor = placeholder[target=y]
    %add : Tensor = call_function[target=torch.add](args=(%x, %y))
    %relu : Tensor = call_function[target=torch.relu](args=(%add,))
    return relu
"""
```

**특징:**
- 그래프 변환/최적화 가능
- 동적 shape 지원
- 디버깅 용이

### CUDA Graph

GPU 커널 호출 시퀀스를 **캡처하여 재사용**하는 NVIDIA 기술입니다.

```
일반 실행 (매번 CPU→GPU 명령 전송):
  CPU: launch kernel A → GPU 실행
  CPU: launch kernel B → GPU 실행
  CPU: launch kernel C → GPU 실행
  → 매번 CPU 오버헤드 발생

CUDA Graph (캡처 후 재사용):
  1. 캡처: kernel A → B → C 시퀀스 기록
  2. 재생: 전체 시퀀스를 한 번의 CPU 호출로 실행
  → CPU 오버헤드 대폭 감소
```

```python
# CUDA Graph 사용 예시
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)  # 캡처

for _ in range(1000):
    g.replay()  # 빠른 재생
```

### 비교

| 항목 | FX Graph | CUDA Graph |
|------|----------|------------|
| 레벨 | Python/PyTorch | CUDA/GPU |
| 목적 | 연산 그래프 최적화 | 커널 호출 오버헤드 감소 |
| 동작 시점 | 컴파일 타임 | 런타임 |
| 유연성 | 동적 shape 지원 | 고정 shape 필요 |

**LLM에서의 활용:**
- Prefill (프롬프트 처리): 가변 길이 → CUDA Graph 어려움
- Decode (토큰 생성): 동일 연산 반복 → CUDA Graph 효과적

---

## 7. 부동소수점 포맷

### 부동소수점이 표현하는 것

부동소수점은 "더 많은 데이터 로드"가 아니라 **모델의 가중치(Weight)와 연산 중간값**을 표현합니다.

```
┌─────────────────────────────────────────────────────────┐
│ 신경망 = 거대한 숫자 덩어리                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  입력 [0.5, 0.3, 0.8]                                   │
│         ↓                                               │
│  가중치 행렬 W (수십억 개의 숫자)                         │
│  ┌─────────────────────────────┐                        │
│  │ 0.0234  -0.1567   0.0891   │  ← 이 숫자들이          │
│  │ 0.2341   0.0023  -0.3421   │    부동소수점으로        │
│  │-0.0012   0.1234   0.0567   │    저장됨               │
│  └─────────────────────────────┘                        │
│         ↓                                               │
│  출력 [0.7, 0.2, 0.1]                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### AI에서의 구체적인 동작

#### 1. 모델 가중치 저장

Qwen2.5-3B 모델 예시:

```
모델 파라미터: 3,000,000,000개 (30억 개의 숫자)

FP32로 저장 시: 3B × 4 bytes = 12GB
FP16으로 저장 시: 3B × 2 bytes = 6GB   ← 절반!
FP8로 저장 시:  3B × 1 byte  = 3GB    ← 1/4!
```

메모리가 중요한 이유:
- GPU VRAM은 제한적 (RTX 5070 Ti = 16GB)
- 모델이 VRAM에 들어가야 실행 가능
- 작은 포맷 = 더 큰 모델 로드 가능

#### 2. 행렬 곱셈 (Forward Pass)

```python
# Linear Layer: y = Wx + b
output = torch.matmul(weight, input) + bias
```

실제 계산:
```
입력 x = [0.5, 0.3, 0.8, ...]  (FP16: 각 2바이트)

가중치 W:
┌──────────────────────────────┐
│ 0.0234   -0.1567    0.0891   │  (FP16: 각 2바이트)
│ 0.2341    0.0023   -0.3421   │
└──────────────────────────────┘

y[0] = 0.5 × 0.0234 + 0.3 × (-0.1567) + 0.8 × 0.0891 + ...
     = 0.0117 + (-0.0470) + 0.0713 + ...
     = 0.0360  ← 이 결과도 부동소수점
```

#### 3. 학습 시 Gradient

```
Forward:  입력 → 가중치 → 출력 → 손실(Loss)
Backward: 손실 → gradient 계산 → 가중치 업데이트

gradient 예시:
  ∂Loss/∂W = 0.00001234  ← 매우 작은 숫자!

가중치 업데이트:
  W_new = W_old - learning_rate × gradient
  W_new = 0.0234 - 0.001 × 0.00001234
  W_new = 0.02339998766
```

### 왜 포맷이 중요한가?

#### 문제 1: 범위 (Exponent) - Underflow

```
Gradient 값 예시:
  정상:   0.00001 (1e-5)
  작은 값: 0.0000000001 (1e-10)

FP16의 범위: ±65504 ~ ±0.00006 (대략)
  → 0.0000000001은 표현 불가 → 0이 됨 (Underflow)
  → Gradient가 사라짐 → 학습 안 됨!

BF16의 범위: FP32와 동일 (±3.4×10³⁸ ~ ±1.2×10⁻³⁸)
  → 0.0000000001도 표현 가능
  → 학습 안정적
```

이것이 BF16이 학습에 선호되는 이유:
```
FP16:  [S][EEEEE][MMMMMMMMMM]   지수 5비트 → 범위 좁음
BF16:  [S][EEEEEEEE][MMMMMMM]   지수 8비트 → 범위 넓음
```

#### 문제 2: 정밀도 (Mantissa) - 업데이트 손실

```
W_old = 1.0000000
업데이트 값 = 0.0000000001

FP32 (가수 23비트): 1.0 - 0.0000000001 = 0.9999999999 ✓
FP16 (가수 10비트): 1.0 - 0.0000000001 = 1.0          ✗ (변화 없음!)
```

정밀도가 낮으면 작은 업데이트가 무시되어 학습이 정체됩니다.

### 추론 vs 학습

| 단계 | 특징 | 권장 포맷 |
|------|------|----------|
| **추론** | 가중치 고정, 출력만 계산 | FP16/FP8 OK (약간의 오차 허용) |
| **학습** | gradient 계산 + 가중치 업데이트 | BF16/FP32 (정밀도 필요) |

### 실제 메모리/속도 차이

```
Qwen2.5-3B 추론 시:

FP32: 12GB (모델) + 4GB (KV Cache) = 16GB, Tensor Core 미사용
FP16: 6GB (모델) + 2GB (KV Cache) = 8GB, Tensor Core 사용 (2배 빠름)
```

### 표현 방식

숫자를 **부호(Sign) + 지수(Exponent) + 가수(Mantissa)**로 표현합니다.

```
FP32 (32비트):
┌─────┬──────────┬───────────────────────────┐
│Sign │ Exponent │        Mantissa           │
│ 1b  │   8bit   │         23bit             │
└─────┴──────────┴───────────────────────────┘
  ↓        ↓              ↓
부호    범위 결정      정밀도 결정
```

### 포맷 비교

| 포맷 | 비트 | 지수 | 가수 | 범위 | 용도 |
|------|------|------|------|------|------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | 학습 기본 |
| TF32 | 19* | 8 | 10 | FP32 동일 | Tensor Core 학습 |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | 학습/추론 |
| FP16 | 16 | 5 | 10 | ±65504 | 추론 |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | Forward pass |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | Backward (gradient) |

*TF32는 내부적으로 19비트, 입출력은 FP32

### 시각적 비교

```
FP32:  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]  32bit
        1    8               23

TF32:  [S][EEEEEEEE][MMMMMMMMMM]               19bit
        1    8          10

BF16:  [S][EEEEEEEE][MMMMMMM]                  16bit
        1    8         7

FP16:  [S][EEEEE][MMMMMMMMMM]                  16bit
        1   5        10

FP8:   [S][EEEE][MMM]  (E4M3)                  8bit
        1   4    3
```

### BF16 vs FP16

```
동일한 작은 숫자 표현 시:

FP16: 지수 5비트 → 범위 제한 → underflow 가능 (0이 됨)
BF16: 지수 8비트 → FP32와 같은 범위 → 표현 가능

→ BF16이 학습에 더 안정적 (gradient vanishing 방지)
```

### FP8의 두 가지 타입 (E4M3, E5M2)

| 타입 | 용도 | 이유 |
|------|------|------|
| E4M3 | Forward pass | 높은 정밀도 필요 (가수 3비트) |
| E5M2 | Backward pass | 넓은 범위 필요 (지수 5비트) |

### LLM에서의 사용

| 단계 | 권장 포맷 | 이유 |
|------|----------|------|
| 학습 (Forward) | BF16/TF32 | 넓은 범위, 충분한 정밀도 |
| 학습 (Gradient) | FP32/BF16 | Gradient 안정성 |
| 추론 | FP16/BF16 | 속도, 메모리 절약 |
| 양자화 추론 | FP8/INT8 | 최대 속도 |

---

## 8. GPU 하드웨어 아키텍처

### 전체 구조 (RTX 5070 Ti / Blackwell 기준)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA RTX 5070 Ti (Blackwell)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    GPC (Graphics Processing Cluster)             │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                    TPC (Texture Processing Cluster)       │   │   │
│  │  │  ┌────────────────────────────────────────────────────┐  │   │   │
│  │  │  │              SM (Streaming Multiprocessor)          │  │   │   │
│  │  │  │  ┌──────────────────────────────────────────────┐  │  │   │   │
│  │  │  │  │  CUDA Cores (FP32/INT32)        128개        │  │  │   │   │
│  │  │  │  ├──────────────────────────────────────────────┤  │  │   │   │
│  │  │  │  │  Tensor Cores (5th Gen)         4개          │  │  │   │   │
│  │  │  │  ├──────────────────────────────────────────────┤  │  │   │   │
│  │  │  │  │  RT Cores (Ray Tracing)         1개          │  │  │   │   │
│  │  │  │  ├──────────────────────────────────────────────┤  │  │   │   │
│  │  │  │  │  Register File                  256KB        │  │  │   │   │
│  │  │  │  ├──────────────────────────────────────────────┤  │  │   │   │
│  │  │  │  │  Shared Memory / L1 Cache       128KB        │  │  │   │   │
│  │  │  │  └──────────────────────────────────────────────┘  │  │   │   │
│  │  │  └────────────────────────────────────────────────────┘  │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         L2 Cache (48MB)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   GDDR7 Memory (16GB, 256-bit)                   │   │
│  │                   Bandwidth: ~504 GB/s                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 계층 구조

```
GPU
├── GPC (Graphics Processing Cluster) × N개
│   ├── TPC (Texture Processing Cluster) × M개
│   │   └── SM (Streaming Multiprocessor) × 2개
│   └── Raster Engine
├── L2 Cache (전체 공유)
├── Memory Controller
└── GDDR Memory
```

### SM (Streaming Multiprocessor) 상세

SM은 **GPU 연산의 핵심 단위**입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    SM (Streaming Multiprocessor)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Sub-core 0     │  │  Sub-core 1     │                  │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                  │
│  │  │CUDA Cores │  │  │  │CUDA Cores │  │                  │
│  │  │  FP32×32  │  │  │  │  FP32×32  │  │                  │
│  │  │  INT32×32 │  │  │  │  INT32×32 │  │                  │
│  │  └───────────┘  │  │  └───────────┘  │                  │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                  │
│  │  │Tensor Core│  │  │  │Tensor Core│  │                  │
│  │  │    ×2     │  │  │  │    ×2     │  │                  │
│  │  └───────────┘  │  │  └───────────┘  │                  │
│  │  Warp Scheduler │  │  Warp Scheduler │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Shared Memory / L1 Cache (128KB)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 연산 유닛 비교

| 유닛 | 용도 | 연산 방식 | LLM 관련성 |
|------|------|----------|-----------|
| **CUDA Core** | 범용 연산 | 스칼라 (1개씩) | 기본 연산 |
| **Tensor Core** | 행렬 연산 | 4×4 행렬 단위 | 핵심 (GEMM) |
| **RT Core** | 레이 트레이싱 | BVH 탐색 | 없음 |

### Tensor Core 동작

한 사이클에 4×4 행렬 곱셈-덧셈:

```
    A (4×4)      B (4×4)      C (4×4)      D (4×4)
  [a a a a]   [b b b b]   [c c c c]   [d d d d]
  [a a a a] × [b b b b] + [c c c c] = [d d d d]
  [a a a a]   [b b b b]   [c c c c]   [d d d d]
  [a a a a]   [b b b b]   [c c c c]   [d d d d]

→ 64번의 곱셈 + 64번의 덧셈 = 한 사이클
```

### 메모리 계층

```
속도 (빠름 → 느림)
──────────────────────────────────────────────────────────→

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Register │ → │ Shared   │ → │ L2 Cache │ → │  GDDR    │
│          │   │ Memory   │   │          │   │ Memory   │
├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤
│ ~8 TB/s  │   │ ~19 TB/s │   │ ~3 TB/s  │   │ 504 GB/s │
│ 스레드별  │   │ Block별  │   │ 전체 공유 │   │ 전체 공유 │
│ 64KB/SM  │   │ 128KB/SM │   │ 48MB     │   │ 16GB     │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### Warp 실행 모델

```
Thread Block (최대 1024 스레드)
├── Warp 0 (32 스레드) → 동시 실행 (SIMT)
├── Warp 1 (32 스레드) → 동시 실행
├── Warp 2 (32 스레드) → 동시 실행
└── ...
```

**SIMT (Single Instruction, Multiple Threads):**
- 32개 스레드가 같은 명령어를 동시 실행
- 분기(if문)가 있으면 성능 저하 (Warp Divergence)

---

## 9. 참고 문서

### GPU 아키텍처

- [NVIDIA RTX Blackwell GPU Architecture Whitepaper (v1.1)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [NVIDIA Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture)

### CUDA 프로그래밍

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Programming Guide (최신)](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9)

### 부동소수점 포맷

- [NVIDIA Transformer Engine - FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [TensorRT Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [Floating-Point 8: Introduction to Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-11 | 초기 작성 |
| 2025-01-11 | 부동소수점이 AI에서 중요한 이유 추가 |
