# Zipformer RK3588 NPU 포팅 — 벤치마크 결과

## 요약

한국어 Streaming Zipformer Transducer를 RK3588 NPU로 포팅한 결과:

- **최고 속도:** 27.5ms/chunk (320ms 음성 → RTF 0.10, 실시간의 10배)
- **최고 정확도:** CER 21.85% (KL divergence INT8, 100-sample 캘리브레이션)
- **CPU 대비:** ONNX INT8 4-thread(35ms)보다 **22% 빠름**
- **핵심 해결:** CumSum 버그 패치 (CER 91.89% → 22.97%), KL 양자화 (→ 21.85%)

---

## End-to-End 성능 (test_wavs 4개, 2.7~6.6초)

| 모드 | Encoder/청크 | Joiner/프레임 | RTF | CER |
|------|:---:|:---:|:---:|:---:|
| **RKNN nocache-static (속도 최적)** | **27.5ms** | **0.7ms** | **~0.10** | **22.97%** |
| **RKNN KL divergence (정확도 최적)** | **53ms** | **0.7ms** | **~0.20** | **21.85%** |
| RKNN rmreshape + C API | 33ms | 0.7ms | ~0.12 | 26.25% |
| ONNX INT8 CPU 4-thread | 35ms | 0.13ms | 0.130 | 19.95% |
| ONNX FP32 CPU 4-thread | 46ms | 0.63ms | 0.182 | 19.95% |

> **nocache-static이 권장 구성:** 속도와 정확도의 최적 균형.
> KL divergence 모델은 rknnlite 기반으로 속도는 느리지만 정확도 최고.

---

## 모든 RKNN Encoder 모델 비교 (set_io_mem, single core0)

| 모델 | 속도(med) | 속도(min) | 파일 크기 | Weight |
|------|:---:|:---:|:---:|:---:|
| **nocache-static (15L, 원본)** | **27.57ms** | **27.33ms** | 79MB | 70.0MB |
| nocache | 29.19ms | 29.02ms | 79MB | 70.0MB |
| nocache-opt (onnxsim) | 29.31ms | 29.12ms | 79MB | 70.0MB |
| baseline (rmreshape) | 31.80ms | 31.54ms | 80MB | 70.0MB |
| rmreshape + onnxsim | 30.91ms | 30.72ms | 80MB | 70.0MB |
| **pruned 12L (2,3,2,2,3)** | **26.68ms** | **26.32ms** | 64MB | 56.5MB |
| **pruned 10L (1,3,2,1,3)** | **23.32ms** | **22.99ms** | 53MB | 46.0MB |
| **pruned 8L (1,2,2,1,2)** | **19.86ms** | **19.55ms** | 44MB | 38.5MB |

---

## 양자화 알고리즘 비교

| 양자화 방식 | 속도(rknnlite) | CER | 비고 |
|------|:---:|:---:|------|
| INT8 일반 (30-sample) | 53ms | 22.97% | baseline |
| **INT8 KL divergence (100-sample)** | **53ms** | **21.85%** | **-1.12pp** |
| INT8 KL divergence (30-sample) | 53ms | 24.12% | 캘리브 부족 |
| INT8 KL diverse (testset) | 53ms | 37.63% | 도메인 불일치 |
| INT8 MMSE | — | — | 빌드 크래시 (메모리 부족) |

> **캘리브레이션 샘플 수가 중요:** 30개→100개로 늘리면 CER -2.27pp 개선.
> **캘리브레이션 도메인이 중요:** 다른 도메인 데이터를 사용하면 오히려 악화.

---

## 레이어 프루닝 실험

Zipformer의 encoder layer 수를 줄여 속도를 높이는 실험.
PyTorch 체크포인트에서 내부 레이어를 제거하고 re-export.

| 구성 | 레이어 수 | RKNN 속도 | 파일 크기 | CER |
|:---:|:---:|:---:|:---:|:---:|
| 2,4,3,2,4 (원본) | 15 | 27.5ms | 79MB | 22.97% |
| 2,3,2,2,3 | 12 | 26.7ms | 64MB | 93.75%* |
| 1,3,2,1,3 | 10 | 23.3ms | 53MB | 90.00%* |
| 1,2,2,1,2 | 8 | **19.9ms** | 44MB | 143.75%* |

\* Fine-tuning 없이 weight pruning만 적용. Fine-tuning 시 CER 회복 예상.

**핵심 발견:**
- RKNN 내부 레이어 수 ≈ encoder layer × 322
- 레이어당 dispatch overhead: ~5.9μs
- 8-layer(1,2,2,1,2) 구성으로 **20ms 미만** 달성 확인
- Fine-tuning 없이는 모든 프루닝 모델의 CER 사용 불가

### NPU 멀티코어 테스트

| 코어 설정 | nocache-static 속도 |
|:---:|:---:|
| **core0 only** | **27.57ms (최적)** |
| core1 only | 28.59ms |
| core2 only | 39.98ms |
| core0+1 | 29.47ms |
| core0+1+2 | 31.41ms |

> 멀티코어는 이 모델에서 오히려 느림. 순차적 레이어 구조로 인해
> 코어 간 동기화 오버헤드가 병렬화 이득보다 크기 때문.

---

## SDK 최적화 시도 결과 (모두 속도 무영향)

| 시도 | 결과 |
|------|------|
| onnxsim (5293→2100 노드) | 속도 변화 없음 (29.36ms) |
| SVD weight compression (50%) | 정확도 파괴 (cosine 0.83) |
| op_target CPU offload | 크래시 (exMatMul CPU 미지원) |
| pass_through=1 | 불안정/느림 (38-6400ms) |
| optimization_level 1,2,3 | 모두 동일 속도 |
| enable_flash_attention | 동일 속도 (29.29ms) |
| model_pruning | 동일 속도 (29.54ms) |
| compress_weight | 동일 속도 (29.51ms) |
| quantized_method='layer' | 동일 속도 (29.23ms) |
| w4a16 / w8a16 | RK3588 미지원 |
| sparse_infer | RK3588 미지원 |
| NPU 주파수 확인 | 이미 최대 1GHz |

> **결론:** RKNN 컴파일러는 ONNX 구조/양자화 방식/SDK 옵션과 무관하게
> 동일한 내부 그래프(~4832 layers)를 생성한다.
> 속도 = 레이어 수 × dispatch overhead(~5.9μs).
> 근본적 개선은 **모델 아키텍처 변경**(레이어 수 감소)으로만 가능하다.

---

## 모델 구성

| 구성요소 | 역할 | 파일 |
|------|------|------|
| Encoder | 음성 특징 → encoder_out + 캐시 갱신 | encoder-epoch-99-avg-1.rknn (80MB) |
| Decoder | 컨텍스트 임베딩 생성 | decoder-epoch-99-avg-1.rknn |
| Joiner | encoder_out × decoder_out → 토큰 확률 | joiner-epoch-99-avg-1.rknn |

**스트리밍 설정:**
- 청크 크기: CHUNK=39 프레임, OFFSET=32 프레임 (320ms 음성)
- 입력: `x [1, 39, 80]` + 35개 캐시 텐서 (총 36개 입력, 2.0MB/청크)
- Encoder 모델별: INT8 일반(cumfix), INT8 KL(kl-100s), FP16(cumfix) 등

---

## CER 분석 (test_wavs 4개)

| 파일 | 정답 | 인식 결과 (RKNN INT8 KL-100s) | CER |
|------|------|------|:---:|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척할고 에스린 것 같았다 | 41.2% |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 있지 | 25.0% |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는... | 부모가 저질에는 큰 실수증 하나는... | 21.2% |
| 3.wav | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? | 0.0% |
| **평균** | | | **21.85%** |

---

## RKNN 포팅 과정에서 해결한 문제

### 문제 1: CumSum 버그 (CER 91.89%)

**증상:** 첫 번째 청크는 정상이지만, 두 번째 청크부터 캐시가 발산하여 출력 붕괴.

```
Chunk 0: encoder_out max_diff=0.0088 ✓
Chunk 1: cached_avg_4 diff=44.0 (정상값 0.22의 200배) ✗
Chunk 2: encoder_out diff=3.52 → 완전 발산
```

**원인:** RKNN SDK의 CumSum 연산이 초기값 ≠ 0일 때 잘못된 결과 반환.
Zipformer는 이전 청크 캐시를 CumSum 초기값으로 사용하므로 치명적.

**해결:** 15개 CumSum 노드를 하삼각 행렬 MatMul로 교체 (`fix_cumsum.py`).
수학적으로 동일한 연산이지만 RKNN의 MatMul은 정상 동작한다.

### 문제 2: RKNN이 ONNX보다 느림

**원인:** 36개 입력 텐서 전송 오버헤드 + NPU 내부 dispatch overhead.
NPU 순수 연산은 2.6ms이지만 전체 호출은 52.7ms (전송이 95%).

**해결 과정:**
1. C API `set_io_mem` (DMA 전송 제거): 52.7ms → 39.2ms
2. `remove_reshape=True` (경계 Reshape 제거): → 30.7ms
3. nocache-static 변환 (정적 그래프): → **27.5ms** (ONNX 35ms 대비 22% 빠름)

---

## 향후 과제

1. **Fine-tuning으로 20ms 달성:** 8-layer(1,2,2,1,2) 모델 학습 → CER 검증
2. **대용량 평가:** KsponSpeech 전체 데이터셋으로 CER 재측정
3. **kaldifeat 설치:** 더 정확한 filterbank 특징 추출 → CER 추가 개선
