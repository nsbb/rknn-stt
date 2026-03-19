# wav2vec2 RK3588 NPU 포팅

한국어 wav2vec2-xls-r-300m(3억 파라미터) 모델을 RK3588 NPU에서 실시간 실행.

> **요약:** CPU 대비 **7.7배 빠른 속도**, **CER 11.78%** 달성.
> 3가지 핵심 기법의 조합: Split INT8+FP16 아키텍처 + KL divergence 양자화 + amplitude normalization.

---

## 성능

702개 스마트홈 명령어(5초 고정, 16kHz)로 측정. RK3588 NPU 3코어 사용.

| 구성 | 속도 | RTF | CER | CPU 대비 |
|------|:---:|:---:|:---:|:---:|
| **Split11 INT8-KL + norm 5.0** | **427ms** | **0.085** | **11.78%** | **7.7x** |
| **Split15 INT8-KL + norm 4.0** | **404ms** | **0.081** | **11.74%** | **8.2x** |
| Split17 INT8-KL + norm 5.0 | 391ms | 0.078 | 11.96% | 8.4x |
| Split11 INT8-KL + norm 0.95 | 427ms | 0.085 | 18.25% | 7.7x |
| Split11 INT8-KL (정규화 없음) | 427ms | 0.085 | 35.25% | 7.7x |
| RKNN FP16 단일 모델 3코어 | 477ms | 0.095 | 35.96% | 6.9x |
| RKNN FP16 단일 코어 | 722ms | 0.144 | - | 4.6x |
| ONNX FP32 CPU 4스레드 | 3,291ms | 0.658 | - | 1.0x |

> **Split15 추천:** Split11과 거의 같은 정확도(11.74%)에 5% 빠름(404ms).
> **Split17 추천:** 최고 속도(391ms)가 필요할 때. 정확도 차이 미미(+0.18pp).

---

## 핵심 기법 #1: Amplitude Normalization

INT8 양자화 모델의 정확도를 극적으로 개선하는 전처리 기법.

### 원리

INT8은 값을 -128~127 범위의 정수로 표현한다. 입력 음성의 볼륨이 작으면
이 범위의 일부만 사용하게 되어 **유효 비트 수가 감소**한다.

입력 음성의 최대 진폭(peak)을 찾아 목표값으로 스케일링하면,
INT8 표현 범위를 최대한 활용하여 양자화 오차가 줄어든다.

```python
peak = np.max(np.abs(audio))
if peak > 0:
    audio = audio / peak * 5.0  # peak를 5.0으로 정규화
```

### 목표값 탐색 실험

0.5부터 10.0까지 14가지 값을 702개 테스트 파일로 실험한 결과:

| 목표값 | CER | 빈 출력 | 완벽 인식(CER=0) |
|:---:|:---:|:---:|:---:|
| 정규화 없음 | 35.25% | 23건 | 204건 (29%) |
| 0.50 | 29.16% | 4건 | — |
| 0.70 | 21.82% | 0건 | — |
| 0.95 | 18.25% | 1건 | 311건 (44%) |
| 1.50 | 13.12% | 0건 | — |
| 2.50 | 12.03% | 0건 | — |
| **5.00** | **11.78%** | **0건** | **378건 (54%)** |
| 7.00 | 11.80% | 0건 | — |
| 10.00 | 12.19% | 0건 | — |

> **발견:** 목표값 5.0이 최적. 그 이상에서는 파형 클리핑으로 인해 미미하게 악화.
> 빈 출력(모델이 아무것도 인식하지 못하는 경우)이 23건에서 0건으로 완전 해소.

### Split별 최적 목표값

동일한 탐색을 각 Split 구성별로 수행한 결과:

| Split | 최적 목표값 | CER |
|:---:|:---:|:---:|
| Split11 | 5.0 | 11.78% |
| Split15 | 4.0 | 11.74% |
| Split17 | 5.0 | 11.96% |

---

## 핵심 기법 #2: Split INT8+FP16 아키텍처

### 문제: 전체 INT8 양자화 실패

wav2vec2의 24-layer Transformer encoder를 전부 INT8로 양자화하면
출력이 완전히 붕괴한다 (전 프레임 `<pad>` 토큰 출력).

원인: Transformer 후반부의 **LayerNorm + Softmax + GELU** 조합이
INT8 정밀도에 극도로 민감하다.

### 해결: 모델 분할

ONNX 모델을 4개 파트로 분할하여, 양자화에 강한 부분만 INT8로 처리한다.

```
┌────────────────────┐
│   Part1 (FP16)     │  Feature Extractor (CNN)
│   12.7MB           │  음성 → 특징 벡터
└─────────┬──────────┘
          ▼
┌────────────────────┐
│   Part2A (INT8-KL) │  Encoder 전반부 (layer 0~N)
│   167MB (N=11)     │  양자화에 강함 → INT8로 속도 향상
└─────────┬──────────┘
          ▼
┌────────────────────┐
│   Part2B (FP16)    │  Encoder 후반부 (layer N+1~23)
│   295MB (N=11)     │  LayerNorm/Softmax/GELU → FP16 필수
└─────────┬──────────┘
          ▼
┌────────────────────┐
│   Part3 (FP16)     │  LM Head
│   5.2MB            │  특징 벡터 → 토큰 확률
└─────────┬──────────┘
          ▼
       텍스트
```

Split 지점(N)에 따라 속도-정확도 트레이드오프가 달라진다:
- **Split11:** encoder의 절반(0~11)을 INT8 → 정확도 최적
- **Split15:** 더 많은 부분(0~15)을 INT8 → 5% 빠름
- **Split17:** 대부분(0~17)을 INT8 → 8% 빠름

### INT8 양자화 시도 전체 이력

| 시도 | 결과 |
|------|------|
| INT8 normal (전체 모델) | 전 프레임 `<pad>` 출력 — 완전 실패 |
| INT8 KL divergence (전체) | 전 프레임 `<pad>` 출력 |
| Encoder-only INT8 | cos=0.37, 빈 출력 |
| auto_hybrid (cos=0.98) | "네네네네네다" — 부분 실패 |
| Split L0-11 INT8 (normal) | CER 43.90% |
| **Split L0-11 INT8-KL** | **CER 35.25%** — KL이 결정적 |
| **+ amplitude norm 5.0** | **CER 11.78%** — 최종 최적 |

---

## 핵심 기법 #3: KL Divergence 양자화

### 일반(normal) 양자화 vs KL divergence 양자화

**일반 양자화**는 텐서의 [min, max]를 [-128, 127]에 균등 매핑한다.
이상치(outlier)가 하나라도 있으면 대부분의 값이 좁은 범위에 몰려 정밀도가 낭비된다.

**KL divergence 양자화**는 원본 분포와 양자화 분포 사이의 **정보 손실을 최소화**하는
클리핑 범위를 탐색한다. 극단값을 일부 잘라내더라도 전체 분포의 보존이 더 좋다.

| 양자화 방식 | CER |
|:---:|:---:|
| 일반 INT8 | 43.90% |
| **KL divergence INT8** | **35.25%** |
| FP16 (양자화 없음) | 35.96% |

> **KL divergence INT8이 FP16보다 CER 0.7pp 좋다.** 양자화가 일종의 정규화(regularization) 역할을 하는 것으로 추정.

---

## 인식 결과 예시

Split11 INT8-KL + amplitude norm 5.0, 702개 스마트홈 명령어 중 대표.

### 정확하게 인식한 경우 (378/702 = 54%)

| 정답 | 인식 결과 | CER |
|------|------|:---:|
| 안방 온도 좀 올려줘 | 안방 온도 좀 올려줘 | 0% |
| 거실 난방 꺼줘 | 거실 난방 꺼줘 | 0% |
| 지금 거실 온도 몇 도야 | 지금 거실 온도 몇 도야 | 0% |
| 매일 아침 여섯 시에 깨워줘 | 매일 아침 여섯시에 깨워줘 | 0% |
| 전체 난방 꺼 | 전체 난방 꺼 | 0% |

### 틀린 경우

| 정답 | 인식 결과 | CER |
|------|------|:---:|
| 난방 온는거 맞춰줘 | 너무 | 100% |
| 오늘 추워 | 뭘써 | 100% |

> **패턴:** 명확한 월패드 명령은 대부분 정확. 비표준 발화("온는거")나 극히 짧은 음성에서 실패.

---

## 모델 파일

| 파일 | 크기 | 설명 |
|------|:---:|------|
| `wav2vec2_part1_features_fp16.rknn` | 12.7MB | Feature Extractor (FP16) |
| `wav2vec2_part2a_int8_kl.rknn` | 167MB | Encoder L0-11 (INT8-KL) |
| `wav2vec2_part2b_fp16.rknn` | 295MB | Encoder L12-23 (FP16) |
| `wav2vec2_part3_lmhead_fp16.rknn` | 5.2MB | LM Head (FP16) |

> GitHub 100MB 제한으로 레포에 미포함. 별도 빌드 또는 다운로드 필요.

- **원본:** [wav2vec2-xls-r-300m-korean](https://huggingface.co/) (HuggingFace)
- **입력:** `[1, 80000]` (5초 × 16kHz)
- **출력:** `[1, 249, 2617]` (CTC logits, 2617 한국어 토큰)

---

## 사용법

### Split INT8+FP16 추론 (최고 성능)

```bash
cd python

# 단일 파일 추론
conda run -n RKNN-Toolkit2 python inference_split_rknn.py ../input/call_elevator.wav

# 벤치마크 (702개)
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --bench

# 속도 우선 (split15, 5% 빠름)
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --split split15 --bench
```

### RKNN FP16 추론 (단일 모델, 비교용)

```bash
cd python
conda run -n RKNN-Toolkit2 python wav2vec2_kor.py \
    --model_path ../model/wav2vec-xls-r-300m_5s_fp16.rknn \
    --vocab_path ../json/vocab.json \
    --input ../input
```

### ONNX → RKNN 변환

```bash
cd python
conda run -n RKNN-Toolkit2 python convert.py ../model/wav2vec-xls-r-300m_5s.onnx rk3588 fp
```

---

## 멀티코어 벤치마크

| 코어 설정 | 속도 | CPU 대비 |
|-----------|:---:|:---:|
| core0 only | 722ms | 4.6x |
| core0+1 (dual) | 526ms | 6.3x |
| **core0+1+2 (triple)** | **476ms** | **6.9x** |

> 3코어 사용 시 단일 코어 대비 **34% 속도 향상**.
> API: `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2)`

---

## 파일 구조

```
wav2vec2/
├── python/
│   ├── inference_split_rknn.py    # Split INT8+FP16 추론 (정규화 내장)
│   ├── wav2vec2_kor.py            # 한국어 추론 (ONNX/RKNN)
│   ├── convert.py                 # ONNX → RKNN 변환
│   ├── export_onnx.py             # PyTorch → ONNX export
│   ├── bench_rknn.py              # C API 벤치마크
│   └── prepare_calibration_data.py
├── json/
│   ├── vocab.json                 # 한국어 2617 토큰
│   └── tokenizer_config.json
├── model/                         # RKNN 모델 파일 (gitignore)
├── input/                         # 테스트 오디오
│   └── wav2vec2_stt_testset/      # 702개 스마트홈 명령어
└── infer_results_rk3588_rknn_int8kl_split11/
    └── evaluation_results.csv     # 전체 평가 결과 (702건)
```

---

## 로컬 테스트셋 평가 (368 샘플, 실환경)

702개 스마트홈 외에 실제 환경(모델하우스, 다양한 거리/소음)에서 추가 평가.

| 테스트셋 | 샘플수 | CER | Avg ms | 설명 |
|----------|:------:|:---:|:------:|------|
| 7F_KSK | 108 | **4.7%** | 437 | 월패드, 화자 KSK |
| 7F_HJY | 107 | 9.9% | 438 | 월패드, 화자 HJY |
| modelhouse_2m | 51 | 9.6% | 435 | 모델하우스, 2m 거리 |
| modelhouse_2m_noheater | 51 | 6.2% | 436 | 모델하우스, 2m, 난방 소음 없음 |
| modelhouse_3m | 51 | 18.5% | 434 | 모델하우스, 3m 거리 |
| **전체** | **368** | **9.0%** | **437** | RTF 0.087 (11.5배 실시간) |

> citrinet(39.9%)과 비교하면 **4.4배 정확**. 상세: [`../eval_results/wav2vec2/README.md`](../eval_results/wav2vec2/README.md)

---

## 참고

- wav2vec2는 **non-streaming** 모델: 전체 음성을 한 번에 처리 (Zipformer와 다름)
- Zipformer와 달리 CumSum 등 RKNN SDK 버그 없음
- INT8 전체 양자화는 RKNN SDK에서 불가 (LayerNorm + Softmax + GELU 민감도)
- `remove_reshape=True`는 입력 layout 문제(NCHW)로 wav2vec2에 사용 불가
