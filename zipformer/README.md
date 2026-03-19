# 한국어 Streaming Zipformer Transducer — RK3588 NPU

실시간 스트리밍 한국어 음성인식. 말하는 도중 320ms마다 텍스트 출력.

## 핵심 성능

| 구성 | Encoder/chunk | RTF | CER | 용도 |
|------|:---:|:---:|:---:|------|
| **RKNN nocache-static** | **27.5ms** | **0.10** | **22.97%** | 속도 최적 (권장) |
| **RKNN KL divergence** | 53ms | 0.20 | **21.51%** | 정확도 최적 |
| ONNX INT8 CPU 4-thread | 35ms | 0.13 | 19.95% | CPU 기준 |

> RTF 0.10 = 320ms 음성을 27.5ms에 처리 = **실시간의 10배 속도**
> RKNN이 ONNX CPU 대비 **22% 빠름** (27.5ms vs 35ms)

---

## 모델 아키텍처

### Streaming Transducer (Encoder + Decoder + Joiner)

```
마이크 입력 (16kHz)
  │
  ▼
┌─────────────────────────────────────────┐
│  Feature Extraction (KaldiFbank)        │
│  16kHz raw → 80-dim log-mel filterbank  │
│  25ms window, 10ms shift                │
└────────────┬────────────────────────────┘
             │ [1, 39, 80] (39프레임 = 320ms 오디오)
             ▼
┌─────────────────────────────────────────┐
│  Encoder (Zipformer v1, 15 layers)      │  ← 27.5ms (RKNN INT8)
│  5-Stage Multi-Scale Conformer          │
│  입력: audio [1,39,80] + 35 캐시 텐서   │
│  출력: encoder_out [1,12,512]           │
│        + 35 갱신된 캐시 텐서            │
└────────────┬────────────────────────────┘
             │ 12개 프레임 × [1, 512]
             ▼
┌─────────────────────────────────────────┐
│  Decoder                                │  ← < 1ms
│  입력: 마지막 2개 BPE 토큰 ID [1, 2]    │
│  출력: context embedding [1, 512]       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Joiner                                 │  ← 0.7ms/frame
│  입력: encoder_out + decoder_out        │
│  출력: logits [1, 5000] → argmax        │
│  BLANK(0) = 무출력, 그 외 = 토큰 출력   │
└────────────┬────────────────────────────┘
             │
             ▼
         한국어 텍스트 (5000 BPE 토큰)
```

### 스트리밍 동작

일반 STT(wav2vec2 등)는 전체 녹음이 끝나야 처리. Zipformer는 **320ms마다 결과 출력:**

```
일반 모델:    [────── 전체 녹음 ──────] → 처리 → "거실 불 꺼줘"

Zipformer:   [0.32초] → ""
             [0.32초] → ""
             [0.32초] → "거실"
             [0.32초] → "불"
             [0.32초] → "꺼줘"
```

이전 청크의 문맥을 **35개 캐시 텐서** (총 2.0MB)에 저장하고, 새 청크마다 갱신.

### Encoder 상세: 5-Stage Multi-Scale

```
Stack 0: 2 layers, 64 heads, 384-dim   (넓은 패턴 감지)
Stack 1: 4 layers, 32 heads, 384-dim   (세밀한 분석)
Stack 2: 3 layers, 16 heads, 384-dim   (압축/병목)
Stack 3: 2 layers,  8 heads, 384-dim   (압축 특징 판단)
Stack 4: 4 layers, 32 heads, 384-dim   (최종 정제)

총 15 encoder layers, ~69M parameters
```

각 레이어: Self-Attention + Depthwise Separable Conv + FFN + Residual

### 캐시 구조 (스택당 7개 × 5스택 = 35개)

| 캐시 | 용도 | Shape |
|------|------|-------|
| `cached_key` | Attention key 히스토리 | [N, H, 1, K] |
| `cached_val` | Attention value 1 | [N, H, 1, V] |
| `cached_val2` | Attention value 2 | [N, H, 1, V] |
| `cached_avg` | 특징 러닝 평균 | [N, 1, 384] |
| `cached_len` | 프레임 카운터 (int64) | [N, 1] |
| `cached_conv1` | Conv 히스토리 1 | [N, 1, 384, 30] |
| `cached_conv2` | Conv 히스토리 2 | [N, 1, 384, 30] |

---

## 모델 스펙

| 항목 | 값 |
|------|-----|
| 원본 모델 | [k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16](https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16) |
| 학습 코드 | [icefall PR #1651](https://github.com/k2-fsa/icefall/pull/1651) |
| Zipformer 버전 | v1 |
| 학습 데이터 | KsponSpeech (한국어 자연발화) |
| 출력 어휘 | 5000 BPE 토큰 |
| 파라미터 | ~69M (Encoder 대부분) |
| 청크 크기 | CHUNK=39프레임, OFFSET=32 (320ms 음성) |
| 입력 | `x [1, 39, 80]` + 35 캐시 = 총 36개 텐서 (2.0MB/chunk) |
| Feature | 80-dim log-mel filterbank (Kaldi 호환) |

---

## RKNN 모델 파일

### Encoder (핵심 — 여러 변종)

| 파일 | 크기 | 양자화 | 속도 | CER | 용도 |
|------|------|--------|------|-----|------|
| `encoder-int8-cumfix-nocache-static.rknn` | 83MB | INT8 | **27.5ms** | 22.97% | **속도 최적 (권장)** |
| `encoder-epoch-99-avg-1-int8-cumfix-kl-100s.rknn` | 80MB | INT8-KL | 53ms | **21.51%** | **정확도 최적** |
| `encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn` | 80MB | INT8+C API | 33ms | 26.25% | C API 사용 |
| `encoder-epoch-99-avg-1-int8-cumfix.rknn` | 80MB | INT8 | 53ms | 22.97% | 표준 |

### Decoder, Joiner

| 파일 | 크기 | 역할 |
|------|------|------|
| `decoder-epoch-99-avg-1.rknn` | 6.7MB | 컨텍스트 임베딩 (마지막 2토큰 → [1,512]) |
| `joiner-epoch-99-avg-1.rknn` | - | 토큰 예측 (encoder_out + decoder_out → [1,5000]) |

### ONNX 원본 (GitHub 100MB 초과, 별도 다운로드)

```bash
# HuggingFace에서 다운로드
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.onnx
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.int8.onnx
```

FP32: 279MB, INT8: 122MB

---

## 양자화 파이프라인

```
encoder-epoch-99-avg-1.onnx (원본 FP32, 279MB)
  │
  ├─ fix_cumsum.py (CumSum 15개 → MatMul 교체, RKNN 버그 우회)
  │   └─ encoder-epoch-99-avg-1-cumfix.onnx
  │
  ├─ 방법 A: nocache-static (권장, 속도 최적)
  │   └─ build/build_nocache_static.py
  │       → encoder-int8-cumfix-nocache-static.rknn (83MB, 27.5ms)
  │
  ├─ 방법 B: KL divergence (정확도 최적)
  │   └─ build/convert_encoder_int8_cumfix.py --quantized_algorithm kl_divergence --calib_samples 100
  │       → encoder-epoch-99-avg-1-int8-cumfix-kl-100s.rknn (80MB, CER 21.51%)
  │
  └─ 방법 C: 표준 INT8
      └─ build/convert_encoder_int8_cumfix.py
          → encoder-epoch-99-avg-1-int8-cumfix.rknn (80MB, 53ms)
```

### 양자화 알고리즘 비교

| 양자화 | 캘리브레이션 | CER | 비고 |
|--------|-----------|-----|------|
| INT8 normal | 30 samples | 22.97% | baseline |
| **INT8 KL divergence** | **100 samples** | **21.51%** | **-1.46pp, 캘리브 수 중요** |
| INT8 KL divergence | 30 samples | 24.12% | 캘리브 부족 |
| INT8 KL (다른 도메인) | testset | 37.63% | 도메인 불일치 → 악화 |
| INT8 MMSE | - | - | 메모리 부족 크래시 |

> 캘리브레이션 샘플 30→100개로 늘리면 CER **-2.27pp** 개선. 도메인 일치 필수.

---

## RKNN 포팅 과정에서 해결한 문제

### 1. CumSum 버그 (Critical — CER 91.89% → 22.97%)

**증상:** 첫 번째 청크는 정상, 두 번째 청크부터 캐시 발산 → 출력 붕괴

```
Chunk 0: encoder_out max_diff=0.0088 ✓ (초기 캐시=0이라 CumSum 정상)
Chunk 1: cached_avg_4 diff=44.0 (정상값 0.22의 200배) ✗
Chunk 2: encoder_out diff=3.52 → 완전 발산
```

**원인:** RKNN SDK의 CumSum 연산이 **초기값 ≠ 0**일 때 잘못된 결과 반환.
Zipformer는 이전 청크 캐시(cached_avg)를 CumSum 초기값으로 사용 → 치명적.

**해결:** 15개 CumSum 노드를 하삼각 행렬 MatMul로 교체 (`fix_cumsum.py`):
```python
# 원본: y = CumSum(x, axis=0)         [T, 1, 384]
# 교체: L = tril(ones(T,T))
#       y = MatMul(L, reshape(x))       수학적 동일, RKNN MatMul은 정상
```

### 2. RKNN이 ONNX보다 느린 문제 (52.7ms → 27.5ms)

**원인:** 36개 입력 텐서 전송 + NPU dispatch overhead. NPU 순수 연산은 2.6ms뿐, 나머지 96%가 오버헤드.

**최적화 과정:**
```
52.7ms  rknnlite inputs_set (Python API)
 ↓ -13.5ms  C API set_io_mem (zero-copy DMA)
39.2ms
 ↓  -8.5ms  remove_reshape=True (경계 Reshape 제거)
30.7ms
 ↓  -3.2ms  nocache-static (캐시 분리 + 정적 그래프)
27.5ms  ← ONNX 35ms보다 22% 빠름!
```

### 병목 상세

```
RKNN inference() 63ms 분해:
├─ NPU 순수 연산: 2.6ms (4%)
└─ 오버헤드: ~60ms (96%)
   ├─ 36 입력 텐서 DMA 전송: 2.0MB @ 33MB/s ≈ 60ms
   └─ ~4832 내부 레이어 × 5.9μs dispatch = 28.5ms
```

### SDK 최적화 시도 (15가지 — 모두 속도 무영향)

| 시도 | 결과 |
|------|------|
| onnxsim (5293→2100 노드) | 변화 없음 |
| flash_attention / model_pruning / compress_weight | 변화 없음 |
| quantized_method (channel/layer) | 변화 없음 |
| optimization_level 1/2/3 | 모두 동일 |
| w4a16 / w8a16 / sparse_infer | RK3588 미지원 |
| 멀티코어 (2코어/3코어) | 오히려 느림 (+2~4ms) |

**결론:** RKNN 컴파일러는 ONNX 구조/양자화/SDK 옵션과 무관하게 동일한 내부 그래프(~4832 layers) 생성. 속도 = 레이어 수 × 5.9μs. **근본적 개선은 모델 레이어 수 감소**로만 가능.

---

## 레이어 프루닝 실험

Encoder 내부 레이어 수를 줄여 속도 측정:

| 구성 | 레이어 | 속도 | 크기 | CER | 비고 |
|------|:---:|:---:|:---:|:---:|------|
| 2,4,3,2,4 (원본) | 15 | 27.5ms | 79MB | 22.97% | production |
| 2,3,2,2,3 | 12 | 26.7ms | 64MB | 93.75%* | fine-tune 필요 |
| 1,3,2,1,3 | 10 | 23.3ms | 53MB | 90.00%* | fine-tune 필요 |
| **1,2,2,1,2** | **8** | **19.9ms** | **44MB** | 143.75%* | **목표: 20ms** |

\* Fine-tuning 없이 weight pruning만 적용. Fine-tuning 시 CER 회복 예상.

### NPU 멀티코어 테스트

| 코어 설정 | 속도 |
|:---:|:---:|
| **core0 only (최적)** | **27.57ms** |
| core1 only | 28.59ms |
| core2 only | 39.98ms |
| core0+1 | 29.47ms |
| core0+1+2 | 31.41ms |

> 순차적 레이어 구조 → 코어 간 동기화 오버헤드가 병렬화 이득 초과. **single core가 최적.**

---

## CER 상세 (test_wavs 4개)

| 파일 | 길이 | 정답 (REF) | RKNN INT8-KL (HYP) | CER |
|------|:---:|------|------|:---:|
| 3.wav | 2.7s | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? | **0.0%** |
| 2.wav | 6.6s | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. | 부모가 저질에는 큰 실수증 하나는 자기 아이를 다른 집안과 비교하는 것이다 | 21.2% |
| 1.wav | 3.4s | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 있지 | 25.0% |
| 0.wav | 3.5s | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척할고 에스린 것 같았다 | 41.2% |
| **평균** | | | | **21.85%** |

---

## Feature Extraction: KaldiFbank

Kaldi 호환 80-dim log-mel filterbank. 순수 NumPy/SciPy 구현 (외부 의존성 없음).

| 파라미터 | 값 |
|----------|-----|
| Sample Rate | 16000 Hz |
| Frame Length | 25ms (400 samples) |
| Frame Shift | 10ms (160 samples) |
| FFT Size | 512 |
| Mel Bins | 80 |
| Frequency Range | 20 ~ 7600 Hz |
| Pre-emphasis | 0.97 |
| Window | Povey (raised cosine^0.85) |
| Snip Edges | False (center-pad) |

---

## 사용법

```bash
# 1. CumSum 패치 (최초 1회)
python fix_cumsum.py

# 2. RKNN 변환 (속도 최적)
conda run -n RKNN-Toolkit2 python rk3588/build/build_nocache_static.py

# 3. RKNN 추론 (NPU)
conda run -n RKNN-Toolkit2 python rk3588/inference_rknn.py

# 4. ONNX 추론 (CPU, 비교용)
python rk3588/inference_onnx.py

# 5. 벤치마크
conda run -n RKNN-Toolkit2 python rk3588/bench/bench_nocache.py
```

---

## 폴더 구조

```
zipformer/
├── README.md                   ← 이 파일
├── fix_cumsum.py               CumSum → MatMul 패치
├── zipformer_onnx_test.py      Sherpa-ONNX 래핑 추론
├── tokens.txt                  5000 BPE 어휘
├── bpe.model                   SentencePiece BPE 모델
├── test_wavs/                  테스트 음성 4개 + trans.txt
│
├── *.onnx                      ONNX 모델들 (원본/패치/변종)
│   ├── encoder-epoch-99-avg-1.onnx         (FP32, 279MB, 다운로드 필요)
│   ├── encoder-epoch-99-avg-1.int8.onnx    (INT8, 122MB, 다운로드 필요)
│   ├── decoder-epoch-99-avg-1.onnx
│   └── joiner-epoch-99-avg-1.onnx
│
└── rk3588/                     RK3588 NPU 포팅 코드
    ├── README.md               성능 요약
    ├── MODEL.md                아키텍처 상세 설명
    ├── RESULTS.md              전체 벤치마크 결과
    ├── WHY_ONNX_BEATS_RKNN.md  병목 원인 분석
    │
    ├── inference_rknn.py       RKNN NPU 추론 (메인)
    ├── inference_onnx.py       ONNX CPU 추론 (비교용)
    ├── encoder_capi.py         RKNN C API wrapper
    ├── fbank.py                80-dim log-mel feature extraction
    │
    ├── *.rknn                  RKNN 모델들
    ├── build/                  ONNX→RKNN 변환 스크립트 (22개)
    ├── bench/                  벤치마크 스크립트 (16개)
    ├── onnx_surgery/           ONNX 그래프 수정 실험 (15개)
    └── experiments/            디버그/테스트 (29개)
```

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| [rk3588/MODEL.md](rk3588/MODEL.md) | 아키텍처 원리 상세 (스트리밍, 캐시, 디코딩) |
| [rk3588/RESULTS.md](rk3588/RESULTS.md) | 전체 벤치마크 (모든 변종, 멀티코어, SDK 옵션) |
| [rk3588/WHY_ONNX_BEATS_RKNN.md](rk3588/WHY_ONNX_BEATS_RKNN.md) | 병목 원인 분석 (dispatch overhead, DMA 대역폭) |

---

## 향후 과제

1. **8-layer 모델 fine-tuning** → 목표 20ms/chunk + CER < 25%
2. **대규모 평가** — KsponSpeech 전체 데이터셋으로 CER 검증
3. **로컬 테스트셋 평가** — 7F_KSK/HJY, modelhouse로 실환경 CER 측정
4. **kaldifeat 통합** → 더 정확한 filterbank → CER 추가 개선
