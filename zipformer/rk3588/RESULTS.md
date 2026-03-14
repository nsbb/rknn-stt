# Zipformer RK3588 NPU 포팅 결과 보고서

## 요약

한국어 Zipformer Streaming Transducer ASR 모델을 RK3588 NPU(RKNN)로 포팅한 결과:
- **RKNN 최적 속도: ~27.5ms/chunk** (nocache-static, single core0)
- **RKNN rmreshape + C API: ~33ms/chunk** (cache 변환 포함)
- RKNN CER 22.97% (ONNX INT8 19.95% 대비 +3pp — INT8 양자화 영향)
- `remove_reshape=True` 옵션으로 rknn_run 39.2ms → 30.7ms 달성 (-22%)
- C API `set_io_mem`으로 rknnlite 대비 16ms 절감
- **layer pruning 실험으로 20ms 달성 확인** (8-layer 모델, fine-tuning 필요)

---

## 레이어 프루닝 실험 (2026-03-14)

PyTorch 체크포인트에서 내부 레이어를 줄여 re-export → RKNN INT8 변환.

| 구성 | 레이어 수 | RKNN 속도(med) | RKNN min | 파일 크기 | Weight | ONNX CER |
|------|----------|---------------|---------|----------|--------|----------|
| 2,4,3,2,4 (원본) | 15 | 29.45ms | 29.13ms | 79MB | 70.0MB | 22.50% |
| **2,3,2,2,3** | **12** | **26.68ms** | **26.32ms** | **64MB** | **56.5MB** | 93.75%* |
| 1,3,2,1,3 | 10 | 23.32ms | 22.99ms | 53MB | 46.0MB | 90.00%* |
| **1,2,2,1,2** | **8** | **19.86ms** | **19.55ms** | **44MB** | **38.5MB** | 143.75%* |

*CER: fine-tuning 없이 weight pruning만 적용한 결과. **Fine-tuning 필수.**

**핵심 발견:**
- RKNN 레이어 수는 내부 encoder 레이어 수에 비례 (~322 RKNN layers/encoder layer)
- RKNN 레이어당 dispatch overhead: ~5.9µs
- **20ms 목표 달성 가능**: 8-layer(1,2,2,1,2) 아키텍처로 19.86ms 확인
- **단, fine-tuning 없이는 CER 사용 불가** — 모든 bypass_scale이 0.27~0.80 범위로 모든 레이어가 유의미하게 기여

**프루닝 파이프라인:**
1. HuggingFace에서 pretrained.pt 다운로드
2. `/tmp/zipformer_export/export_pruned.py`로 reduced-layer ONNX export
3. CumSum fix + static 변환
4. RKNN INT8 빌드

### NPU 멀티코어 테스트

| 코어 설정 | nocache-static 속도 |
|-----------|-------------------|
| **core0 only** | **27.57ms (최적)** |
| core1 only | 28.59ms |
| core2 only | 39.98ms |
| core0+1 | 29.47ms |
| core0+1+2 | 31.41ms |

멀티코어는 이 모델에서 오히려 느림 (순차적 레이어 구조, 동기화 오버헤드).

### ONNX 최적화 시도 결과

| 시도 | 결과 |
|------|------|
| onnxsim (5293→2100 노드) | RKNN 속도 변화 없음 (29.36ms) |
| SVD weight compression (50%) | 정확도 파괴 (cosine 0.83) |
| op_target CPU offload | 크래시 (exMatMul CPU 미지원) |
| pass_through=1 | 불안정/느림 (38-6400ms) |
| optimization_level 1,2,3 | 모두 동일 속도 |
| Encoder layer 제거 (ONNX surgery) | 캐시 구조 복잡성으로 불가 |
| enable_flash_attention=True | 동일 속도 (29.29ms) |
| model_pruning=True | 동일 속도 (29.54ms) |
| compress_weight=True | 동일 속도 (29.51ms) |
| quantized_method='layer' | 동일 속도 (29.23ms) |
| quantized_algorithm='kl_divergence' | 동일 속도 (29.31ms) |
| w4a16 / w8a16 | RK3588 미지원 |
| sparse_infer=True | RK3588 미지원 |
| NPU 주파수 확인 | 이미 최대 1GHz |

**결론: RKNN 컴파일러는 ONNX 노드 수/양자화 방법/SDK 옵션과 무관하게 동일한 내부 그래프(~4832 layers) 생성. 속도 = 레이어 수 × dispatch overhead(~5.9µs). 개선은 모델 아키텍처 변경(레이어 수 감소)으로만 가능.**

---

## 모델 구성

| 구성 요소 | 모델 파일 | 역할 |
|-----------|-----------|------|
| Encoder | encoder-epoch-99-avg-1.onnx (279MB) / .int8.onnx (122MB) | 청크 입력 → encoder_out + 캐시 업데이트 |
| Decoder | decoder-epoch-99-avg-1.onnx | 컨텍스트 임베딩 |
| Joiner  | joiner-epoch-99-avg-1.onnx | encoder_out × decoder_out → logits |
| RKNN Encoder FP16 | encoder-epoch-99-avg-1-fp16.rknn (148MB) | NPU 변환 (FP16, CumSum 버그) |
| RKNN Encoder FP16 CumFix | encoder-epoch-99-avg-1-cumfix.rknn | NPU 변환 (FP16, CumSum 패치, CER 29.32%) |
| RKNN Encoder INT8 | encoder-epoch-99-avg-1-int8.rknn (80MB) | NPU 변환 (INT8, CumSum 버그) |
| **RKNN Encoder INT8 CumFix** | **encoder-epoch-99-avg-1-int8-cumfix.rknn** | **NPU 변환 (INT8, CumSum 패치, CER 22.97%)** |
| RKNN Decoder | decoder-epoch-99-avg-1.rknn | NPU 변환 |
| RKNN Joiner | joiner-epoch-99-avg-1.rknn | NPU 변환 |

**스트리밍 설정:** CHUNK=39, OFFSET=32 (39프레임 입력, 32프레임 이동)
- 입력 x: [1, 39, 80]
- 캐시: 35개 텐서 (cached_len × 5, cached_avg × 5, cached_key × 5, cached_val × 5, cached_val2 × 5, cached_conv1 × 5, cached_conv2 × 5)
- 총 입력: 36개, 총 크기: 2.0 MB/청크

---

## 성능 벤치마크 (RK3588 보드)

### 모든 RKNN 변환 모델 비교 (set_io_mem, single core0)

| 모델 | med(ms) | min(ms) | 파일 | Weight |
|------|---------|---------|------|--------|
| **nocache-static (원본 15L)** | **27.57** | **27.33** | 79MB | 70.0MB |
| nocache | 29.19 | 29.02 | 79MB | 70.0MB |
| nocache-opt (onnxsim) | 29.31 | 29.12 | 79MB | 70.0MB |
| baseline (rmreshape) | 31.80 | 31.54 | 80MB | 70.0MB |
| rmreshape + onnxsim | 30.91 | 30.72 | 80MB | 70.0MB |
| **pruned 12L (2,3,2,2,3)** | **26.68** | **26.32** | **64MB** | **56.5MB** |
| **pruned 10L (1,3,2,1,3)** | **23.32** | **22.99** | **53MB** | **46.0MB** |
| **pruned 8L (1,2,2,1,2)** | **19.86** | **19.55** | **44MB** | **38.5MB** |

### Encoder 단위 테스트 (`bench_hybrid_timing.py`)

| 방식 | Encoder/청크 | CER | 비고 |
|------|-------------|-----|------|
| RKNN FP16 (rknnlite.api) | 63ms | 29.32% | NPU 순수 계산 2.6ms + 전송 60ms |
| RKNN INT8 cumfix (rknnlite.api) | 52.7ms | 25.00% | inputs_set 방식 |
| RKNN INT8 cumfix (set_io_mem) | 39.2ms | 25.00% | C API zero-copy |
| **RKNN INT8 rmreshape (C API)** | **32.7ms** | **26.25%** | **remove_reshape + set_io_mem + batch sync** |
| ONNX FP32 (4-thread) | 40ms | 19.95% | |
| ONNX INT8 (4-thread) | 35ms | 19.95% | |
| RKNN Hybrid (순차) | 103ms | 19.95% | RKNN + ONNX 모두 실행 |

### Joiner 성능 비교

| 방식 | 시간/호출 | 비고 |
|------|-----------|------|
| ONNX FP32 | 0.69ms | |
| ONNX INT8 | 0.07ms | **10배 빠름** |
| RKNN | 0.79ms | FP32보다 약간 느림 |

### End-to-End 성능 (test_wavs 기준)

| 모드 | Enc/청크 | Joi/프레임 | RTF | CER |
|------|---------|------------|-----|-----|
| **RKNN nocache-static** | **27.5ms** | **0.7ms** | **~0.10** | **22.97%** |
| RKNN rmreshape C API | 33ms | 0.7ms | ~0.12 | 26.25% |
| ONNX INT8 | 35ms | 0.13ms | 0.130 | 19.95% |
| ONNX FP32 | 46ms | 0.63ms | 0.182 | 19.95% |

---

## CER 분석 (test_wavs 4개 파일)

| 파일 | 참조 | 가설 (ONNX INT8) | CER |
|------|------|-----------------|-----|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척 할려구 에스는 거 같았다 | 37.5% |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 하진 | 26.7% |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. | 부모가 저질에는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다 | 15.6% |
| 3.wav | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? | 0.0% |
| **평균** | | | **19.95%** |

---

## RKNN 포팅 과정에서 발생한 문제와 해결책

### 문제 1: Encoder 캐시 발산 (RKNN Pure 실패)

**증상:**
```
Chunk 0: encoder_out max_diff=0.0088 ✓
Chunk 1: cached_avg_4 diff=44.0073 (max_val=0.22) → 200배 오차!
Chunk 2: cached_conv2_4 diff=159.85, encoder_out diff=3.52 (파국)
```

**근본 원인:** RKNN의 CumSum 연산이 non-zero 초기 상태에서 잘못 계산됨

**해결책: CumSum → MatMul 패치 (2026-03-13)**
- `fix_cumsum.py`: 15개 CumSum 노드를 하삼각 행렬 MatMul로 교체
- 결과: CER 91.89% → 22.97% (INT8 기준)

### 문제 2: RKNN이 ONNX보다 느림 → 해결됨

**원인:** 36개 입력 텐서 전송 오버헤드 + NPU 내부 Reshape/Transpose dispatch 오버헤드

**해결 과정:**
1. C API `set_io_mem` + CACHEABLE 메모리: 52.7ms → 39.2ms
2. `remove_reshape=True`로 변환: rknn_run 39.2ms → 30.7ms
3. nocache 변환 (cache 제거): 30.7ms → 28.5ms
4. **최종: 27.5ms/chunk — ONNX 35ms보다 22% 빠름**

---

## 향후 과제

1. **Fine-tuning으로 20ms 달성**: 8-layer(1,2,2,1,2) 모델 fine-tune → CER 확인
   - 아키텍처 확인 완료, 학습 파이프라인(icefall) 설정 필요
   - Export 도구: `/tmp/zipformer_export/export_pruned.py`
2. **CER 개선**: RKNN INT8 양자화 정확도 개선 (22.97% → 목표 <20%)
3. **kaldifeat 설치**: 더 정확한 fbank → CER 추가 개선
4. **대용량 평가**: KsponSpeech 전체로 CER 측정
