# Zipformer RK3588 NPU 포팅

한국어 Streaming Zipformer Transducer를 RK3588 NPU(RKNN)에서 실시간 실행.

## 성능

| 모드 | Encoder/chunk | RTF | CER | 비고 |
|------|:---:|:---:|:---:|------|
| **RKNN NPU (속도)** | **27.5ms** | **0.10** | **22.97%** | nocache-static, core0 |
| **RKNN NPU (정확도)** | **53ms** | **0.20** | **21.51%** | KL divergence 양자화 |
| ONNX INT8 CPU | 35ms | 0.13 | 19.95% | 4-thread |
| ONNX FP32 CPU | 46ms | 0.18 | 19.95% | 4-thread |

> RKNN이 ONNX CPU 대비 **22% 빠름** (27.5ms vs 35ms)
> KL divergence 양자화로 CER 22.97% → 21.51% (-1.46pp) 개선

## 속도 최적화 과정

```
rknnlite inputs_set       52.7ms   기본 Python API
→ C API set_io_mem        39.2ms   zero-copy DMA 메모리
→ remove_reshape=True     30.7ms   경계 Reshape 제거
→ nocache-static          27.5ms   캐시 텐서 분리 + static shape
```

## 추론 결과 예시 (ONNX INT8, test_wavs 4개)

| 파일 | 길이 | CER | 정답 (REF) | 추론 결과 (HYP) |
|------|:---:|:---:|------|------|
| 3.wav | 2.7s | **0.0%** | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? |
| 2.wav | 6.6s | 15.6% | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. | 부모가 저질에는 큰 실수 중 하나는 자기 아이를 다른 집안과 비교하는 것이다 |
| 1.wav | 3.4s | 26.7% | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 하진 |
| 0.wav | 3.5s | 37.5% | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척 할려구 에스는 거 같았다 |

> 테스트셋: 일반 한국어 문장 4개 (2.7~6.6초). 평균 CER 19.95%.
> 결과 CSV: `infer_results_rk3588_onnx_int8/evaluation_results.csv`

## 발견한 RKNN 버그

**CumSum 연산 오류** — non-zero 초기값에서 누적합이 틀림. 15개 CumSum을 하삼각 MatMul로 교체하여 해결.

```
패치 전 CER: 91.89% (사용 불가)
패치 후 CER: 22.97%
```

## 병목 분석

RKNN 내부에서 ~4832개 레이어로 분해됨. 레이어당 dispatch overhead ~5.9us.
**실제 NPU 연산 2.6ms (9%), 나머지 91%가 dispatch overhead.**

SDK 옵션 15가지 시도 → 모두 속도 무영향:

| 시도 | 결과 |
|------|------|
| onnxsim (5293→2100 노드) | 변화 없음 |
| flash_attention / model_pruning / compress_weight | 변화 없음 |
| quantized_method (channel/layer) | 변화 없음 |
| quantized_algorithm (normal/kl_divergence) | 변화 없음 |
| w4a16 / w8a16 / sparse_infer | RK3588 미지원 |
| 멀티코어 (2코어/3코어) | 오히려 느림 (+2~4ms) |
| optimization_level 1/2/3 | 동일 |

**결론: 속도 = 레이어 수 x 5.9us. 줄이려면 모델 레이어 수를 줄여야 함.**

## 레이어 프루닝 실험

PyTorch 체크포인트에서 encoder 내부 레이어를 줄여 re-export → RKNN 변환:

| 구성 | 레이어 | 속도 | 파일 | CER |
|------|:---:|:---:|:---:|:---:|
| 2,4,3,2,4 (원본) | 15 | 27.5ms | 79MB | 22.97% |
| 2,3,2,2,3 | 12 | 26.7ms | 64MB | 93.75%* |
| 1,3,2,1,3 | 10 | 23.3ms | 53MB | 90.00%* |
| **1,2,2,1,2** | **8** | **19.9ms** | **44MB** | **143.75%*** |

*Fine-tuning 없이 weight pruning만 적용한 결과. **20ms 달성은 아키텍처적으로 확인됨 — fine-tuning 필요.**

## 파일 안내

### 루트 — 핵심 코드

| 파일 | 설명 |
|------|------|
| `inference_rknn.py` | RKNN NPU 추론 (C API set_io_mem) |
| `inference_onnx.py` | ONNX CPU 추론 (INT8/FP32) |
| `encoder_capi.py` | RKNN C API wrapper |
| `fix_cumsum.py` | CumSum → MatMul 패치 |
| `fbank.py` | 80-dim log-mel feature extraction |

### `build/` — RKNN 변환 스크립트 (22개)

ONNX → RKNN 변환, 양자화, 다양한 옵션 조합 빌드.
주요: `build_nocache_static.py`, `convert_encoder_int8_cumfix.py`

### `bench/` — 벤치마크 (16개)

NPU 속도 측정, 멀티코어 비교, 변환 옵션별 비교.
주요: `bench_nocache.py`, `bench_multicore.py`

### `onnx_surgery/` — ONNX 그래프 수정 실험 (15개)

Reshape/Transpose 제거, 레이어 분리, 캐시 최적화 등 시도.

### `experiments/` — 디버그/테스트/일회성 실험 (29개)

캐시 발산 디버그, C API 테스트, 성능 프로파일링 등.

### 문서

| 파일 | 설명 |
|------|------|
| [RESULTS.md](RESULTS.md) | 전체 벤치마크 결과 상세 |
| [WHY_ONNX_BEATS_RKNN.md](WHY_ONNX_BEATS_RKNN.md) | 병목 원인 분석 |

## 사용법

```bash
# 1. ONNX 추론 (CPU)
python inference_onnx.py

# 2. CumSum 패치
python fix_cumsum.py

# 3. RKNN 변환
conda run -n RKNN-Toolkit2 python build/build_nocache_static.py

# 4. RKNN 추론 (NPU)
python inference_rknn.py
```

## 향후 과제

1. 8-layer(1,2,2,1,2) 모델 fine-tuning → 20ms + 사용 가능 CER 달성
2. RKNN INT8 양자화 정확도 개선 (22.97% → 목표 <20%)
3. KsponSpeech 전체 데이터셋으로 CER 평가
