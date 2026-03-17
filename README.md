# rknn-stt — RK3588 NPU 한국어 음성인식

RK3588 보드의 NPU(신경망 처리 장치)에서 한국어 음성인식(STT) 모델을 실시간으로 구동하는 프로젝트.

> **왜 NPU인가?**
> 음성인식 모델은 수억 개의 연산을 수행한다. CPU로 처리하면 수 초가 걸리지만,
> NPU는 행렬 곱셈에 특화된 하드웨어로 같은 작업을 수백 밀리초 안에 처리한다.
> RK3588의 NPU는 6 TOPS(초당 6조 회 연산)로, 저전력 임베디드 환경에서
> 클라우드 없이 음성인식을 가능하게 한다.

---

## 핵심 결과

| | Zipformer | wav2vec2 |
|------|-----------|----------|
| **아키텍처** | Streaming Transducer | CTC (non-streaming) |
| **동작 방식** | 320ms씩 잘라서 실시간 처리 | 5초 음성을 한 번에 처리 |
| **모델 크기** | 80MB (INT8) | 480MB (INT8-KL + FP16 split) |
| **추론 속도** | **27.5ms / 320ms 청크** | **427ms / 5초 음성** |
| **실시간 배율** | 10배 (RTF 0.10) | 12배 (RTF 0.085) |
| **글자 오류율 (CER)** | 21.85% | **11.78%** |
| **CPU 대비 속도** | 1.3배 | **7.7배** |

> **RTF(Real-Time Factor)란?**
> "처리 시간 ÷ 음성 길이"로 계산한다. RTF 0.085는 5초 음성을 0.425초에 처리한다는 뜻이다.
> 1.0 미만이면 실시간보다 빠르고, 숫자가 작을수록 좋다.

> **CER(Character Error Rate)이란?**
> "인식 결과에서 잘못된 글자 수 ÷ 정답 글자 수"로 계산한다.
> CER 11.78%는 100글자 중 약 12글자가 틀린다는 뜻이다. 숫자가 작을수록 좋다.

---

## 최적화 여정 요약

### wav2vec2: CER 35.96% → 11.78% (24.18pp 개선)

```
 원본 FP16 ─────────────── CER 35.96%
   │
   ├─ KL divergence 양자화 ─ CER 35.25%  (-0.71pp)  속도 10% 향상
   │   └─ 전체 INT8은 출력 붕괴 → encoder 절반만 INT8로 분할 해결
   │
   ├─ amplitude norm 0.95 ── CER 18.25%  (-16.99pp)  🔑 핵심 발견 #1
   │   └─ 입력 음량을 일정하게 맞추면 INT8 정확도 급상승
   │
   └─ amplitude norm 5.0 ── CER 11.78%  (-6.47pp)   🔑 핵심 발견 #2
       └─ "크게" 정규화할수록 INT8 dynamic range 활용도 증가
```

### Zipformer: CER 91.89% → 21.85% (70.04pp 개선)

```
 원본 INT8 ─────────────── CER 91.89%  ← 거의 쓸 수 없는 수준
   │
   ├─ CumSum 버그 패치 ──── CER 22.97%  (-68.92pp)  🔑 핵심 발견
   │   └─ RKNN SDK의 CumSum 연산 버그 → 하삼각 행렬 곱으로 교체
   │
   └─ KL divergence 양자화 ─ CER 21.85%  (-1.12pp)
       └─ 캘리브레이션 100샘플로 양자화 범위 최적화
```

---

## 핵심 기법 상세

### 1. Amplitude Normalization (wav2vec2)

**문제:** INT8 양자화는 가중치를 8비트 정수(-128~127)로 표현한다.
입력 음성의 볼륨이 작으면 이 범위의 일부만 사용하게 되어 정밀도가 낭비된다.

**해결:** 입력 음성의 최대 진폭(peak)을 기준값으로 정규화한다.

```python
peak = np.max(np.abs(audio))
audio = audio / peak * 5.0  # peak를 5.0으로 맞춤
```

**왜 5.0인가?** 0.5부터 10.0까지 14가지 값을 실험한 결과:

| 정규화 목표값 | CER (702개 테스트) | 빈 출력 수 | 완벽 인식 비율 |
|:---:|:---:|:---:|:---:|
| 정규화 없음 | 35.25% | 23건 | 29% |
| 0.95 | 18.25% | 1건 | 44% |
| **5.0** | **11.78%** | **0건** | **54%** |
| 10.0 | 12.19% | 0건 | 53% |

> 목표값이 클수록 파형이 INT8 표현 범위를 넓게 활용하여 정밀도가 올라간다.
> 단, 너무 크면(>10) 클리핑이 발생하여 오히려 악화된다.

### 2. Split INT8+FP16 아키텍처 (wav2vec2)

**문제:** wav2vec2의 24-layer encoder를 전부 INT8로 양자화하면
LayerNorm·Softmax·GELU 연산의 정밀도 손실로 출력이 완전히 붕괴한다.

**해결:** 모델을 4개 파트로 분할하여, 민감하지 않은 전반부만 INT8로 양자화한다.

```
오디오 입력
  │
  ▼
┌─────────────────┐
│  Part1 (FP16)   │  Feature Extractor — CNN으로 음성 특징 추출
│  12.7MB         │
└────────┬────────┘
         │
  ▼
┌─────────────────┐
│  Part2A (INT8)  │  Encoder 전반부 (layer 0~11) — 양자화에 강함
│  167MB          │  ← KL divergence 알고리즘으로 양자화 범위 최적화
└────────┬────────┘
         │
  ▼
┌─────────────────┐
│  Part2B (FP16)  │  Encoder 후반부 (layer 12~23) — 양자화에 민감
│  295MB          │  ← LayerNorm+Softmax+GELU → FP16 필수
└────────┬────────┘
         │
  ▼
┌─────────────────┐
│  Part3 (FP16)   │  LM Head — 토큰 확률 출력
│  5.2MB          │
└────────┬────────┘
         │
  ▼
텍스트 출력
```

### 3. CumSum 버그 패치 (Zipformer)

**문제:** RKNN SDK의 CumSum(누적합) 연산이 초기값이 0이 아닐 때 잘못된 결과를 반환한다.
Zipformer는 스트리밍 처리를 위해 이전 청크의 상태(캐시)를 CumSum의 초기값으로 사용하므로,
두 번째 청크부터 계산이 발산하여 CER 91.89%라는 사실상 무의미한 결과가 나온다.

**해결:** 15개 CumSum 노드를 하삼각 행렬(lower triangular matrix) 곱으로 교체한다.
수학적으로 동일한 연산이지만 RKNN의 MatMul은 정상 동작한다.

```python
# CumSum([a, b, c]) = [a, a+b, a+b+c]
# 이것은 아래 행렬 곱과 같다:
# [[1,0,0],    [a]     [a    ]
#  [1,1,0],  × [b]  =  [a+b  ]
#  [1,1,1]]    [c]     [a+b+c]
```

### 4. KL Divergence 양자화

**일반 INT8 양자화:** 텐서의 최솟값~최댓값을 -128~127로 균등 매핑한다.
이상치(outlier)가 있으면 대부분의 값이 좁은 범위에 몰려 정밀도가 낭비된다.

**KL divergence 양자화:** 원본 분포와 양자화 분포 사이의 정보 손실(KL divergence)을
최소화하는 clipping 범위를 탐색한다. 이상치를 잘라내더라도 전체적인 정보 보존이 더 좋다.

| 모델 | 일반 INT8 CER | KL INT8 CER | 개선 |
|------|:---:|:---:|:---:|
| Zipformer | 22.97% | 21.85% | -1.12pp |
| wav2vec2 (split11) | 43.90% | 35.25% | -8.65pp |

---

## wav2vec2 성능 상세

702개 스마트홈 명령어(5초 고정)로 측정. RK3588 NPU 3코어 사용.

| 구성 | 속도 | RTF | CER | CPU 대비 |
|------|:---:|:---:|:---:|:---:|
| **Split11 INT8-KL + norm 5.0** | **427ms** | **0.085** | **11.78%** | **7.7x** |
| Split15 INT8-KL + norm 4.0 | 404ms | 0.081 | 11.74% | 8.2x |
| Split17 INT8-KL + norm 5.0 | 391ms | 0.078 | 11.96% | 8.4x |
| Split11 INT8-KL + norm 0.95 | 427ms | 0.085 | 18.25% | 7.7x |
| Split11 INT8-KL (정규화 없음) | 427ms | 0.085 | 35.25% | 7.7x |
| RKNN FP16 단일 모델 | 477ms | 0.095 | 35.96% | 6.9x |
| ONNX FP32 CPU 4스레드 | 3,291ms | 0.658 | — | 1.0x |

> **Split15가 최적 균형:** Split11과 거의 같은 정확도(11.74%)에 5% 더 빠름(404ms).

### 인식 결과 예시

```
정답: "안방 온도 좀 올려줘"     → 인식: "안방 온도 좀 올려줘"     ✓ (CER 0%)
정답: "거실 난방 꺼줘"         → 인식: "거실 난방 꺼줘"         ✓ (CER 0%)
정답: "매일 아침 여섯 시에 깨워줘" → 인식: "매일 아침 여섯시에 깨워줘" ✓ (CER 0%)
정답: "난방 온는거 맞춰줘"      → 인식: "너무"                ✗ (CER 100%)
```

> 702개 중 378개(54%)를 완벽하게 인식. 실패하는 경우는 주로 비표준 발화나 극히 짧은 음성.

---

## Zipformer 성능 상세

4개 테스트 음성(2.7~6.6초)으로 측정. NPU 단일 코어 사용.

| 구성 | Encoder 속도 | RTF | CER |
|------|:---:|:---:|:---:|
| **RKNN INT8 KL (100-sample)** | **27.5ms** | **0.10** | **21.85%** |
| RKNN INT8 일반 | 27.5ms | 0.10 | 22.97% |
| ONNX INT8 CPU 4스레드 | 35ms | 0.13 | 19.95% |

### 속도 최적화 히스토리

```
52.7ms  rknnlite inputs_set (baseline)
  │
  ├─ C API set_io_mem ──────── 39.2ms  (-13.5ms)  DMA 전송 제거
  │
  ├─ remove_reshape=True ───── 30.7ms  (-8.5ms)   내부 Reshape 제거
  │
  └─ nocache-static 변환 ──── 27.5ms  (-3.2ms)   정적 그래프 최적화
                                                   ONNX CPU보다 22% 빠름 ✓
```

### 속도 병목 분석

RKNN 컴파일러는 Zipformer를 ~4,832개 내부 레이어로 변환한다.
각 레이어마다 ~5.9μs의 dispatch overhead가 발생하여,
실제 NPU 연산(2.6ms, 9%)보다 dispatch overhead(25ms, 91%)가 훨씬 크다.

이 병목은 SDK 옵션으로 해결할 수 없으며(15가지 시도, 모두 무영향),
모델의 레이어 수를 줄여야만 개선 가능하다.

| 레이어 구성 | RKNN 속도 | 비고 |
|:---:|:---:|------|
| 15개 (원본) | 27.5ms | 현재 사용 중 |
| 12개 | 26.7ms | Fine-tuning 필요 |
| 10개 | 23.3ms | Fine-tuning 필요 |
| 8개 | **19.9ms** | Fine-tuning 필요 |

---

## 빠른 시작

### 환경

- **보드:** RK3588 (NPU 6 TOPS, 3코어)
- **SDK:** RKNN-Toolkit2 2.3.2
- **Python:** 3.8 (conda env: `RKNN-Toolkit2`)

### wav2vec2 추론

```bash
cd wav2vec2/python

# 단일 파일 추론
conda run -n RKNN-Toolkit2 python inference_split_rknn.py ../input/call_elevator.wav

# 702개 테스트셋 벤치마크
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --bench

# 속도 우선 모드 (split15, 5% 빠름)
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --split split15 --bench
```

### Zipformer 추론

```bash
cd zipformer/rk3588

# RKNN NPU 추론
conda run -n RKNN-Toolkit2 python inference_rknn.py

# ONNX CPU 추론 (비교용)
conda run -n RKNN-Toolkit2 python inference_onnx.py
```

### 모델 준비

모델 파일은 GitHub 용량 제한으로 레포에 포함되지 않는다.

```bash
# Zipformer: HuggingFace에서 다운로드 (279MB)
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.onnx -P zipformer/

# CumSum 패치 + RKNN 변환
cd zipformer/rk3588
python fix_cumsum.py
conda run -n RKNN-Toolkit2 python build/build_nocache_static.py
```

---

## 프로젝트 구조

```
rknn-stt/
├── zipformer/                     # Streaming Zipformer Transducer
│   ├── rk3588/                    # RKNN 포팅 코드
│   │   ├── inference_rknn.py      # NPU 추론 (스트리밍)
│   │   ├── inference_onnx.py      # CPU 추론 (비교용)
│   │   ├── fix_cumsum.py          # CumSum → MatMul 패치
│   │   ├── fbank.py               # 음성 특징 추출 (filterbank)
│   │   ├── build/                 # ONNX → RKNN 변환 스크립트
│   │   ├── RESULTS.md             # 전체 벤치마크 결과
│   │   └── WHY_ONNX_BEATS_RKNN.md # 병목 분석 보고서
│   ├── test_wavs/                 # 테스트 음성 4개 (2.7~6.6초)
│   └── *.onnx                     # ONNX 모델 (다운로드 필요)
│
├── wav2vec2/                      # wav2vec2-xls-r-300m (3억 파라미터)
│   ├── python/
│   │   ├── inference_split_rknn.py  # Split INT8+FP16 추론
│   │   ├── convert.py               # ONNX → RKNN 변환
│   │   └── bench_rknn.py            # 속도 벤치마크
│   ├── model/                     # RKNN 모델 파일 (gitignore)
│   ├── input/                     # 테스트 오디오
│   │   └── wav2vec2_stt_testset/  # 702개 스마트홈 명령어
│   └── README.md                  # wav2vec2 상세 문서
│
└── testset/                       # 공용 테스트셋
```

---

## 원본 모델

| 모델 | 출처 | 파라미터 수 |
|------|------|:---:|
| Zipformer | [icefall Korean Zipformer](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12) | ~70M |
| wav2vec2 | [wav2vec2-xls-r-300m-korean](https://huggingface.co/) | 300M |

## 상세 문서

| 문서 | 내용 |
|------|------|
| [wav2vec2/README.md](wav2vec2/README.md) | wav2vec2 상세 (Split 아키텍처, 양자화 실험, 정규화 발견) |
| [zipformer/rk3588/RESULTS.md](zipformer/rk3588/RESULTS.md) | Zipformer 전체 벤치마크 (모든 변환 모델 비교) |
| [zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md](zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md) | NPU 병목 원인 분석 (dispatch overhead) |
