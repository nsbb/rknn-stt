# wav2vec2 Split INT8-KL 평가 결과

## 모델 정보

| 항목 | 내용 |
|------|------|
| **원본** | facebook/wav2vec2-xls-r-300m (Large, **300M params**) + 한국어 CTC head |
| **양자화** | **Split INT8** — 아래 상세 설명 참조 |
| **최대 입력** | 5초 (80,000 samples @ 16kHz) |
| **합계 크기** | 462MB |

## Split INT8 양자화란?

wav2vec2-xls-r-300m은 Transformer Encoder **24개 레이어**로 구성되어 있다.
전체를 INT8로 양자화하면 LayerNorm / Softmax / GELU 등 정밀도에 민감한 연산이 깨져서 **출력이 garbage**가 된다.

이를 해결하기 위해 **Encoder를 두 파트로 분할**한다:

```
오디오 입력 (5초, 16kHz)
  │
  ▼
┌─────────────────────────────────────┐
│  Part1: CNN Feature Extractor       │  ← FP16 (정밀도 유지)
│  wav2vec2_part1_features_fp16.rknn  │     13MB
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Part2A: Encoder Layer 0-11         │  ← ★ INT8-KL ★ (속도 이득)
│  wav2vec2_part2a_int8_kl.rknn       │     160MB
│                                     │
│  KL divergence 양자화:              │
│  100개 음성 샘플로 캘리브레이션,    │
│  각 레이어의 activation 분포를      │
│  최대한 보존하는 양자화 범위 결정   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Part2B: Encoder Layer 12-23        │  ← FP16 (정밀도 유지)
│  wav2vec2_part2b_fp16.rknn          │     295MB
│                                     │
│  상위 레이어는 의미 표현을 다루므로 │
│  INT8 양자화 시 정확도 손실이 큼    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Part3: CTC Head (Linear)           │  ← FP16
│  wav2vec2_part3_lmhead_fp16.rknn    │     5.2MB
└────────────┬────────────────────────┘
             │
             ▼
         텍스트 출력
```

**왜 Layer 0-11만 INT8인가?**

| 시도 | INT8 범위 | CER | 비고 |
|------|----------|-----|------|
| 전체 FP16 | 없음 | 35.96% | baseline, 느림 |
| 전체 INT8 | Layer 0-23 | garbage | 출력 깨짐 |
| **Split 11** | **Layer 0-11** | **35.25%** | **FP16보다 오히려 좋음** |
| Split 15 | Layer 0-15 | 37.06% | 약간 나빠짐, 더 빠름 |
| Split 17 | Layer 0-17 | 37.57% | 더 나빠짐, 가장 빠름 |

Split 11이 정확도 최적 — FP16 baseline(35.96%)보다 오히려 CER이 낮다.
INT8-KL 양자화가 일종의 regularization 효과를 준 것으로 추정.

**amplitude normalization (target=5.0)** 도 핵심:
- target=0.95 (기본): CER 18.2%
- **target=5.0**: CER 11.5% → INT8 양자화 범위에 최적화된 입력 스케일

## RKNN 모델 파일 (4개)

| 순서 | 파일명 | 크기 | 데이터타입 | 역할 |
|:----:|--------|------|:----------:|------|
| 1 | `wav2vec2_part1_features_fp16.rknn` | 13MB | FP16 | CNN 특징 추출기 |
| 2 | `wav2vec2_part2a_int8_kl.rknn` | 160MB | **INT8** | Encoder Layer 0-11 (핵심 연산) |
| 3 | `wav2vec2_part2b_fp16.rknn` | 295MB | FP16 | Encoder Layer 12-23 |
| 4 | `wav2vec2_part3_lmhead_fp16.rknn` | 5.2MB | FP16 | CTC Head → 문자 출력 |

추론 시 4개 모델을 **순서대로 실행**한다 (Part1 출력 → Part2A 입력 → ...).

## 양자화 파이프라인

```
wav2vec2-xls-r-300m (PyTorch, 한국어 fine-tuned)
  │
  ├─ ONNX export (5초 고정 입력, 80000 samples)
  │
  ├─ ONNX Split: Encoder Layer 0-11 / Layer 12-23 분리
  │
  ├─ Part1 (CNN)         → RKNN FP16 변환 → wav2vec2_part1_features_fp16.rknn
  ├─ Part2A (Enc L0-11)  → RKNN INT8 변환 (KL divergence, 100-sample cal) → wav2vec2_part2a_int8_kl.rknn
  ├─ Part2B (Enc L12-23) → RKNN FP16 변환 → wav2vec2_part2b_fp16.rknn
  └─ Part3 (LM Head)     → RKNN FP16 변환 → wav2vec2_part3_lmhead_fp16.rknn
```

## 테스트셋별 결과

| Testset | Samples | CER | Avg ms |
|---------|--------:|----:|-------:|
| 7F_KSK | 108 | **4.7%** | 437 |
| 7F_HJY | 107 | 9.9% | 438 |
| modelhouse_2m | 51 | 9.6% | 435 |
| modelhouse_2m_noheater | 51 | 6.2% | 436 |
| modelhouse_3m | 51 | 18.5% | 434 |
| **TOTAL** | **368** | **9.0%** | **437** |
