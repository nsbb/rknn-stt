# 한국어 STT 모델 로컬 테스트셋 평가 결과

평가일: 2026-03-18
플랫폼: RK3588 NPU (3코어)

---

## 평가 대상 모델

### 1. wav2vec2 — Split INT8-KL (권장)

| 항목 | 내용 |
|------|------|
| 원본 모델 | facebook/wav2vec2-xls-r-300m (Large, **300M params**) + 한국어 CTC head |
| 양자화 방법 | **Split INT8** — Encoder 24개 레이어 중 전반부만 INT8, 후반부는 FP16 |
| 최대 입력 | 5초 (80,000 samples @ 16kHz) |
| 정규화 | amplitude peak → 5.0 (INT8 정확도 핵심) |

#### Split INT8이란?

전체 Encoder를 INT8로 양자화하면 LayerNorm/Softmax/GELU가 깨져서 출력이 garbage가 된다.
이를 해결하기 위해 **Encoder를 두 파트로 분할**:

- **Layer 0-11** → INT8-KL 양자화 (KL divergence, 100-sample 캘리브레이션) → **속도 이득**
- **Layer 12-23** → FP16 유지 (상위 레이어는 의미 표현을 다루므로 정밀도 필수) → **정확도 보존**

| 양자화 방식 | INT8 범위 | CER | 비고 |
|------------|----------|-----|------|
| 전체 FP16 | 없음 | 35.96% | baseline |
| 전체 INT8 | Layer 0-23 | garbage | 출력 깨짐 |
| **Split 11 (채택)** | **Layer 0-11** | **35.25%** | **FP16보다 오히려 좋음** |
| Split 15 | Layer 0-15 | 37.06% | 약간 나빠짐, 더 빠름 |
| Split 17 | Layer 0-17 | 37.57% | 더 나빠짐, 가장 빠름 |

**RKNN 모델 파일 (4개, 순서대로 실행):**

| 순서 | 파일명 | 크기 | 데이터타입 | 역할 |
|:----:|--------|------|:----------:|------|
| 1 | `wav2vec2_part1_features_fp16.rknn` | 13MB | FP16 | CNN 특징 추출기 |
| 2 | `wav2vec2_part2a_int8_kl.rknn` | 160MB | **INT8** | Encoder Layer 0-11 (핵심 연산) |
| 3 | `wav2vec2_part2b_fp16.rknn` | 295MB | FP16 | Encoder Layer 12-23 |
| 4 | `wav2vec2_part3_lmhead_fp16.rknn` | 5.2MB | FP16 | CTC Head |
| | **합계** | **462MB** | | |

**양자화 파이프라인:**
```
wav2vec2-xls-r-300m (PyTorch, 한국어 fine-tuned)
  ├─ ONNX export (5초 고정 입력)
  ├─ ONNX Split: Encoder Layer 0-11 / Layer 12-23 분리
  │
  ├─ Part1 (CNN)         → FP16  → wav2vec2_part1_features_fp16.rknn
  ├─ Part2A (Enc L0-11)  → INT8-KL (100-sample cal) → wav2vec2_part2a_int8_kl.rknn
  ├─ Part2B (Enc L12-23) → FP16  → wav2vec2_part2b_fp16.rknn
  └─ Part3 (LM Head)     → FP16  → wav2vec2_part3_lmhead_fp16.rknn
```

### 2. citrinet — FP16

| 항목 | 내용 |
|------|------|
| 원본 모델 | NeMo CitriNet 한국어 CTC (Jasper-based CNN) |
| 토크나이저 | SentencePiece BPE (2048 토큰 + blank) |
| 양자화 방법 | **FP16** (양자화 없음) |
| 최대 입력 | 3초 (300 frames, 80-dim mel spectrogram) |
| ONNX 수정 | LogSoftmax 제거, SE→ReduceMean, ReduceMean→Conv, Squeeze 제거 |

**RKNN 모델 파일 (1개):**

| 파일명 | 크기 | 역할 |
|--------|------|------|
| `citrinet_fp16.rknn` | 281MB | 전체 모델 (FP16) |

**양자화 파이프라인:**
```
NeMo CitriNet (PyTorch, .nemo)
  → ONNX export (3초 고정 입력)
  → fix_citrinet_graph.py (RKNN 버그 4개 우회)
  → FP16 RKNN 변환
  → 1개 RKNN 파일
```

---

## wav2vec2/model/ 파일 설명

30개 RKNN 파일 중 **실제 사용하는 4개**와 나머지 실험 파일:

### 실제 사용 (split11, 권장)
| 파일 | 크기 | 설명 |
|------|------|------|
| `wav2vec2_part1_features_fp16.rknn` | 13MB | CNN feature extractor (공통) |
| `wav2vec2_part2a_int8_kl.rknn` | 160MB | Encoder L0-11, INT8 KL divergence |
| `wav2vec2_part2b_fp16.rknn` | 295MB | Encoder L12-23, FP16 |
| `wav2vec2_part3_lmhead_fp16.rknn` | 5.2MB | CTC head (공통) |

### 실험: 다른 split 지점
| 파일 | 크기 | 설명 |
|------|------|------|
| `wav2vec2_enc6a_int8.rknn` | 85MB | Encoder L0-5, INT8 (normal) |
| `wav2vec2_enc6b_fp16.rknn` | 442MB | Encoder L6-23, FP16 |
| `wav2vec2_enc8a_int8.rknn` | 110MB | Encoder L0-7, INT8 (normal) |
| `wav2vec2_enc8b_fp16.rknn` | 393MB | Encoder L8-23, FP16 |
| `wav2vec2_enc15a_int8_kl.rknn` | 210MB | Encoder L0-14, INT8-KL |
| `wav2vec2_enc15a_int8.rknn` | 210MB | Encoder L0-14, INT8 (normal) |
| `wav2vec2_enc15b_fp16.rknn` | 197MB | Encoder L15-23, FP16 |
| `wav2vec2_enc17a_int8_kl.rknn` | 235MB | Encoder L0-16, INT8-KL |
| `wav2vec2_enc17a_int8.rknn` | 235MB | Encoder L0-16, INT8 (normal) |
| `wav2vec2_enc17b_fp16.rknn` | 148MB | Encoder L17-23, FP16 |
| `wav2vec2_enc19a_int8_kl.rknn` | 260MB | Encoder L0-18, INT8-KL |
| `wav2vec2_enc19a_int8.rknn` | 260MB | Encoder L0-18, INT8 (normal) |
| `wav2vec2_enc19b_fp16.rknn` | 99MB | Encoder L19-23, FP16 |
| `wav2vec2_enc20a_int8.rknn` | 273MB | Encoder L0-19, INT8 (normal) |
| `wav2vec2_enc20b_fp16.rknn` | 74MB | Encoder L20-23, FP16 |

### 실험: 비분할 모델
| 파일 | 크기 | 설명 |
|------|------|------|
| `wav2vec-xls-r-300m_5s_fp16.rknn` | 625MB | 전체 모델 FP16, 5초 입력 |
| `wav2vec-xls-r-300m_5s_int8.rknn` | 320MB | 전체 모델 INT8, 5초 (정확도 나쁨) |
| `wav2vec-xls-r-300m_5s_layer-kl.rknn` | 319MB | 전체 모델 layer-wise KL |
| `wav2vec-xls-r-300m_5s_layer.rknn` | 319MB | 전체 모델 layer-wise normal |
| `wav2vec-xls-r-300m_20s_fp16.rknn` | 653MB | 전체 모델 FP16, 20초 입력 |
| `wav2vec-xls-r-300m_20s.rknn` | 653MB | 전체 모델, 20초 입력 |
| `wav2vec2_part2_encoder_int8.rknn` | 310MB | 전체 encoder INT8 (garbage output) |
| `wav2vec2_part2_encoder_kl.rknn` | 310MB | 전체 encoder KL |
| `wav2vec2_part2a_fp16.rknn` | 313MB | Encoder L0-11, FP16 (baseline) |
| `wav2vec2_part2a_int8.rknn` | 160MB | Encoder L0-11, INT8 (normal) |
| `wav2vec2_part2b_int8.rknn` | 151MB | Encoder L12-23, INT8 (정확도 나쁨) |

---

## 평가 결과

### 전체 요약

| 모델 | 전체 CER | 평균 지연 | RTF | 실시간 배수 | 모델 크기 | 최대 입력 |
|------|---------|----------|-----|-----------|----------|----------|
| **wav2vec2 split INT8-KL** | **9.0%** | 437ms | 0.087 | 11.5배 | 462MB | 5초 |
| citrinet FP16 | 39.9% | **63ms** | **0.021** | **48배** | 281MB | 3초 |

> **RTF** (Real-Time Factor) = 처리시간 / 오디오길이. RTF 0.087 = 5초 오디오를 437ms에 처리 = 실시간의 11.5배 속도.

### 테스트셋별 비교

| 테스트셋 | 샘플수 | wav2vec2 CER | citrinet CER | wav2vec2 시간 (RTF) | citrinet 시간 (RTF) |
|----------|--------|-------------|-------------|-------------------|-------------------|
| 7F_KSK | 108 | **4.7%** | 27.5% | 437ms (0.087) | 64ms (0.021) |
| 7F_HJY | 107 | **9.9%** | 54.9% | 438ms (0.088) | 64ms (0.021) |
| modelhouse_2m | 51 | **9.6%** | 37.5% | 435ms (0.087) | 63ms (0.021) |
| modelhouse_2m_noheater | 51 | **6.2%** | 25.6% | 436ms (0.087) | 60ms (0.020) |
| modelhouse_3m | 51 | **18.5%** | 51.4% | 434ms (0.087) | 61ms (0.020) |
| **전체** | **368** | **9.0%** | **39.9%** | **437ms (0.087)** | **63ms (0.021)** |

### 테스트셋 설명

| 테스트셋 | 설명 |
|----------|------|
| 7F_KSK | 월패드 스마트홈 명령 (화자 KSK, 108개) |
| 7F_HJY | 월패드 스마트홈 명령 (화자 HJY, 107개) |
| modelhouse_2m | 모델하우스 환경, 2m 거리 (51개) |
| modelhouse_2m_noheater | 모델하우스 환경, 2m 거리, 난방 소음 없음 (51개) |
| modelhouse_3m | 모델하우스 환경, 3m 거리 (51개) |

---

## 분석

### wav2vec2 강점
- CER 9.0%로 citrinet (39.9%) 대비 압도적 정확도
- 7F_KSK (가까운 거리, 조용한 환경)에서 CER 4.7%로 실용 수준
- 5초 입력으로 더 긴 발화 처리 가능

### citrinet 강점
- **63ms로 wav2vec2 (437ms) 대비 7배 빠름**
- 모델 크기 281MB로 가벼움
- 3초 입력 제한이 스마트홈 명령에는 충분

### 공통 약점
- 3m 거리 (modelhouse_3m)에서 두 모델 모두 정확도 하락
- 7F_HJY 화자가 7F_KSK보다 인식 어려움 (발음 차이?)

---

## 재현

```bash
# 전체 평가
conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model all

# 개별 모델
conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model wav2vec2
conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model citrinet
```

결과 CSV: `eval_results/<model>/<testset>.csv`
