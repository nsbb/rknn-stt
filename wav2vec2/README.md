# wav2vec2 RK3588 NPU 포팅

한국어 wav2vec2-xls-r-300m 모델을 RK3588 NPU에서 실시간 실행.

## 성능

| 방식 | 속도 (5초 오디오) | RTF | CER (702) | CPU 대비 | 비고 |
|------|:---:|:---:|:---:|:---:|------|
| **Split11 INT8-KL 3-core** | **427ms** | **0.085** | **35.25%** | **7.7x** | encoder L0-11 INT8, FP16보다 정확 |
| Split15 INT8-KL 3-core | 404ms | 0.081 | 37.06% | 8.2x | encoder L0-15 INT8, 15% faster |
| Split17 INT8-KL 3-core | 391ms | 0.078 | 37.57% | 8.4x | encoder L0-17 INT8, 18% faster |
| RKNN FP16 3-core | 477ms | 0.095 | 35.96% | 6.9x | 단일 모델 baseline |
| RKNN FP16 core0 | 722ms | 0.144 | - | 4.6x | C API set_io_mem |
| ONNX FP32 CPU 4t | 3291ms | 0.658 | - | 1.0x | onnxruntime |

> **Split11 INT8-KL: FP16보다 CER 0.7pp 좋고 10% 빠름** (427ms vs 477ms)
> **Split17 INT8-KL: 최고 속도, CER +1.6pp** (391ms, RTF 0.078)

### Split INT8+FP16 아키텍처

ONNX 모델을 4개 파트로 분할, encoder 전반부만 INT8(KL divergence) 양자화:

```
Part1(FP16) → Part2A(INT8-KL, layers 0-N) → Part2B(FP16, layers N-23) → Part3(FP16)
features       encoder 전반                  encoder 후반 + layer_norm     lm_head
12.7MB         167MB (N=11)                  295MB (N=11)                  5.2MB
```

- **핵심 발견: KL divergence 양자화가 normal 대비 CER 8pp 개선** (35.25% vs 43.90%)
- Encoder 후반부(layers 12-23)는 LayerNorm+Softmax+GELU의 INT8 민감도로 FP16 필수
- Encoder 전반부(layers 0-11) INT8-KL: CER 35.25% (FP16 35.96%보다 좋음)
- 전체 encoder INT8: cosine 0.37-0.47, 출력 완전 실패

### INT8 양자화 시도 이력

| 방식 | 결과 |
|------|------|
| INT8 normal (전체) | 전 프레임 `<pad>` 출력 |
| INT8 KL divergence (전체) | 전 프레임 `<pad>` 출력 |
| INT8 layer method (전체) | 전 프레임 `<pad>` 출력 |
| Encoder-only INT8 | cos=0.37, 빈 출력 |
| Encoder half INT8 (L12-23) | cos=0.40, 빈 출력 |
| auto_hybrid (cos=0.98) | "네네네네네다" (부분 실패) |
| Split L0-11 INT8 (normal) | CER 43.90% (+8pp) |
| **Split L0-11 INT8-KL** | **CER 35.25% (-0.7pp, 최적)** |

> **핵심: KL divergence 알고리즘이 wav2vec2 encoder INT8에 결정적.**
> normal quantization은 CER 43.90%이지만 KL divergence는 35.25% (FP16보다 좋음).

## 추론 결과 예시 (Split11 INT8-KL, 702개 중 대표)

### 잘 되는 경우 (CER 0%)

| REF (정답) | HYP (추론 결과) | CER |
|------|------|:---:|
| 안방 온도 좀 올려줘 | 안방 온도 좀 올려줘 | 0.0% |
| 거실 난방 꺼줘 | 거실 난방 꺼줘 | 0.0% |
| 거실 온도 좀 높여줘 | 거실 온도 좀 높여줘 | 0.0% |
| 지금 거실 온도 몇 도야 | 지금 거실 온도 몇 도야 | 0.0% |
| 거실 설정 온도 알려줘 | 거실 설정 온도 알려줘 | 0.0% |

### 틀리는 경우

| REF (정답) | HYP (추론 결과) | CER |
|------|------|:---:|
| 난방 온는거 맞춰줘 | 너무  | 100% |
| 전체 난방 꺼 | (빈 출력) | 100% |
| 오늘 추워 | 뭘써 | 100% |
| 난방 이십 칠도 |  필동 | 100% |
| 내일 아침 여섯 시에 깨워줘 | (빈 출력) | 100% |

> 잘 되는 경우: 명확한 발화, 일반적인 월패드 명령
> 안 되는 경우: 짧은 발화, 잡음 심한 음성, 비표준 발화 ("온는거"), 월패드 도메인 외 명령 ("깨워줘")
>
> 테스트셋: 월패드 명령어 702개 (5초 고정). 평균 CER 33.80%.
> 결과 CSV: `infer_results_rk3588_rknn_int8kl_split11/evaluation_results.csv`

## 모델

| 파일 | 크기 | 설명 |
|------|:---:|------|
| `wav2vec-xls-r-300m_5s.onnx` | 1.2GB | ONNX FP32, 5초 입력 |
| `wav2vec-xls-r-300m_5s_fp16.rknn` | 625MB | RKNN FP16, 5초 입력 |
| `wav2vec2_part1_features_fp16.rknn` | 12.7MB | Part1: feature extractor FP16 |
| `wav2vec2_part2a_int8_kl.rknn` | 167MB | Part2A: encoder L0-11 INT8-KL |
| `wav2vec2_part2b_fp16.rknn` | 295MB | Part2B: encoder L12-23 FP16 |
| `wav2vec2_part3_lmhead_fp16.rknn` | 5.2MB | Part3: lm_head FP16 |

모델 파일은 GitHub 100MB 제한으로 레포에 미포함. 별도 다운로드 필요.

- 원본: [wav2vec2-xls-r-300m-korean](https://huggingface.co/) (HuggingFace)
- 입력: `[1, 80000]` (5초 × 16kHz) 또는 `[1, 320000]` (20초 × 16kHz)
- 출력: `[1, 249, 2617]` (CTC logits, 2617 한국어 토큰)

## 사용법

### Split INT8+FP16 추론 (최고 성능)

```bash
cd python
conda run -n RKNN-Toolkit2 python inference_split_rknn.py ../input/call_elevator.wav
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --bench  # 벤치마크
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --split split15 --bench  # 속도 우선
```

### RKNN FP16 추론 (단일 모델)

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

## 파일 구조

```
wav2vec2/
├── python/
│   ├── wav2vec2_kor.py          # 한국어 추론 (ONNX/RKNN)
│   ├── wav2vec2.py              # 영어 추론 (영어 토큰)
│   ├── convert.py               # ONNX → RKNN 변환
│   ├── export_onnx.py           # PyTorch → ONNX export
│   ├── resize_onnx_sim.py       # ONNX 입력 크기 변경
│   ├── prepare_calibration_data.py
│   ├── bench_rknn.py            # C API 벤치마크
│   └── inference_split_rknn.py  # Split INT8+FP16 추론
├── json/
│   ├── vocab.json               # 한국어 2617 토큰
│   └── tokenizer_config.json
├── model/                       # 모델 파일 (gitignore)
└── input/                       # 테스트 오디오
    └── wav2vec2_stt_testset/    # 702개 스마트홈 명령어
```

## 멀티코어 벤치마크

| 코어 설정 | 속도 | 개선 |
|-----------|:---:|:---:|
| core0 only | 722ms | baseline |
| core1 only | 744ms | -3% |
| core2 only | 756ms | -5% |
| core0+1 (dual) | 526ms | **-27%** |
| core0+1+2 (triple) | **476ms** | **-34%** |

> 3코어 사용 시 `rknn_set_core_mask(ctx, 7)` (core0+1+2)

## 참고

- wav2vec2는 non-streaming (전체 오디오를 한번에 처리)
- Zipformer와 달리 CumSum 등 RKNN 버그 없음
- INT8 전체 양자화는 RKNN SDK에서 불가 (LayerNorm + Softmax + GELU 조합이 극도로 민감)
- 해결: encoder를 layer 11/12에서 분할, 전반부만 INT8-KL → FP16과 동등 정확도 + 10% 속도 향상
- KL divergence 양자화가 핵심 — normal quantization 대비 CER 8pp 개선
- `remove_reshape=True`는 입력 layout 문제(NCHW)로 사용 불가
