# Zipformer RK3588 NPU 포팅 결과 보고서

## 요약

한국어 Zipformer Streaming Transducer ASR 모델을 RK3588 NPU(RKNN)로 포팅한 결과:
- **RKNN rmreshape + C API: ~35ms/chunk** (batch sync + prune 적용)
- RKNN CER 26.25% (ONNX INT8 19.95% 대비 +6.3pp — INT8 양자화 영향)
- `remove_reshape=True` 옵션으로 rknn_run 39.2ms → 30.7ms 달성 (-22%)
- C API `set_io_mem`으로 rknnlite 대비 16ms 절감
- **소프트웨어 최적화 한계 도달** — 12가지 방법 시도, 상세: `OPTIMIZATION_LOG.md`
- 20ms 목표: **NPU 드라이버 업그레이드(0.9.6→최신) 필수** (커널 재빌드)

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

**최적화 이력:**
1. rknnlite inputs_set: 52.7ms
2. C API set_io_mem + CACHEABLE: 39.2ms (-13.5ms)
3. `remove_reshape=True`: 30.7ms (rknn_run) → 33ms (cache update 포함) (-6.2ms)
4. batch sync 최적화: 33ms → **32.7ms** (cache convert 2.2ms → 1.2ms)
5. 최종: **RKNN 32.7ms < ONNX 35ms** (RKNN이 ONNX를 이김)

**`remove_reshape=True` 효과:**
- 경계 Reshape/Transpose 제거 → NPU dispatch 오버헤드 -8.5ms
- perf_detail: Reshape 181ops(8.2ms) + Transpose 202ops(6.7ms) = 14.9ms → 상당수 제거
- 단점: 입출력 shape 변경 → cache 변환에 reshape+transpose 필요 (~3ms CPU)

### Joiner 성능 비교

| 방식 | 시간/호출 | 비고 |
|------|-----------|------|
| ONNX FP32 | 0.69ms | |
| ONNX INT8 | 0.07ms | **10배 빠름** |
| RKNN | 0.79ms | FP32보다 약간 느림 |

### End-to-End 성능 (test_wavs 기준, `bench_final.py`)

| 모드 | Enc/청크 | Joi/프레임 | RTF | CER |
|------|---------|------------|-----|-----|
| **RKNN rmreshape C API** | **33ms** | **0.7ms** | **~0.12** | **26.25%** |
| ONNX INT8 | 35ms | 0.13ms | 0.130 | 19.95% |
| ONNX FP32 | 46ms | 0.63ms | 0.182 | 19.95% |
| RKNN INT8 cumfix (set_io_mem) | 39.2ms | 0.7ms | ~0.15 | 25.00% |
| RKNN INT8 cumfix (rknnlite) | 52.7ms | 0.7ms | 0.175 | 25.00% |
| RKNN Hybrid | 65+40ms | 0.79ms | 0.349 | 19.95% |

**속도: RKNN rmreshape > ONNX INT8 (33ms < 35ms)**
**정확도: ONNX INT8 > RKNN (19.95% < 26.25%)**

---

## CER 분석 (test_wavs 4개 파일)

| 파일 | 참조 | 가설 (ONNX INT8) | CER |
|------|------|-----------------|-----|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척 할려구 에스는 거 같았다 | 37.5% |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 하진 | 26.7% |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. | 부모가 저질에는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다 | 15.6% |
| 3.wav | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? | 0.0% |
| **평균** | | | **19.95%** |

### CER 비교 (모드별)

| 모드 | 0.wav | 1.wav | 2.wav | 3.wav | 평균 |
|------|-------|-------|-------|-------|------|
| ONNX INT8 | 37.5% | 26.7% | 15.6% | 0.0% | **19.95%** |
| RKNN INT8 CumSum 패치 | 56.2% | 20.0% | 15.6% | 0.0% | **22.97%** |
| RKNN FP16 CumSum 패치 | 37.5% | 26.7% | 53.1% | 0.0% | **29.32%** |
| RKNN Pure (CumSum 버그) | - | - | - | - | **91.89%** |

**CumSum 패치 효과:** 91.89% → 22.97% (INT8 기준, ONNX INT8 대비 +3pp)

**CER 개선 이력:**
- 초기 구현 (preemphasis 없음): 26.2%
- preemphasis=0.97 추가, frame mean 제거: **19.95% (-6.3pp)**

**남은 CER 원인:**
1. 0.wav, 1.wav의 오인식은 기본 ONNX 모델 자체의 한계 (sherpa-onnx 기준 모델)
2. kaldifeat 미사용 (numpy 자체 구현) - 추가 격차 가능
3. 4개 파일 샘플로 일반화 주의 필요

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
- `new_cached_avg` 계산 그래프: `CumSum → Add(누적합 + prev_avg*prev_len) → Mul(1/new_len) → Gather(마지막)`
- chunk 0: 모든 캐시 = 0 → 정확 (CumSum(0)=0)
- chunk 1+: prev_avg ≠ 0 → RKNN CumSum 오차 발생

**해결책 1: 하이브리드 추론**
- RKNN: encoder_out 계산 (정확)
- ONNX: 캐시 업데이트 (정확)
- 결과: CER 동일, RTF 0.349 (ONNX만보다 느림)

**해결책 2: CumSum → MatMul 패치 (2026-03-13)**
- `fix_cumsum.py`: 15개 CumSum 노드를 하삼각 행렬 MatMul로 교체
- `encoder-epoch-99-avg-1-cumfix.onnx` → `encoder-epoch-99-avg-1-cumfix.rknn` 변환
- 결과: CER 29.32% (ONNX 19.95% 대비 +9.4pp, FP16 노이즈 영향)
- ONNX 대비 max_diff: 0.0 (수학적으로 동일), NPU 실측 max_diff: 0.013

### 문제 2: RKNN이 ONNX보다 느림 → 해결됨

**원인:** 36개 입력 텐서 (총 2MB) 전송 오버헤드 + NPU 내부 Reshape/Transpose dispatch 오버헤드

**해결 과정:**
1. C API `set_io_mem` + CACHEABLE 메모리: 52.7ms → 39.2ms
2. `remove_reshape=True`로 변환: rknn_run 39.2ms → 30.7ms
3. cache 변환 (reshape+transpose): +~3ms
4. **최종: 33ms/chunk — ONNX 35ms보다 빠름**

**perf_detail 분석으로 발견한 핵심 병목:**
- Reshape 181ops (8.2ms, 17%) + Transpose 202ops (6.7ms, 14%) = 14.9ms (31%)
- `remove_reshape=True`로 경계 Reshape 제거 → 8.5ms 절감

### 문제 3: RKNN 변환 오류들

| 오류 | 해결 |
|------|------|
| `input shape ['N', 2] not support` | `load_onnx(inputs=input_names, input_size_list=...)` 추가 |
| `len of mean_values ([0]) wrong` | config에서 `mean_values`/`std_values` 완전 제거 |
| `core_mask not support in simulator` | rknn.api → rknnlite.api 로 변경 |
| `input shape (2,1,384,30) wrong, expect nhwc` | 4D 텐서 NCHW→NHWC 변환 (`np.transpose(a,(0,2,3,1))`) |
| INT8 캘리브레이션 format 오류 | .npy 파일로 저장 + dataset.txt 방식 사용 |
| INT8 캘리브레이션 NHWC 오류 | 캘리브레이션 데이터는 NCHW 그대로 저장 |

### 문제 4: fbank 구현 오차

**증상:** ONNX CER 26.2% (예상 5-10%)

**원인:** kaldifeat와의 차이
- preemphasis 누락 (kaldifeat 기본값 0.97)
- frame mean 제거 불필요

**해결:** preemphasis=0.97 추가, frame mean 제거 → CER 19.95%

**fbank 성능 최적화:**
- 기존 (프레임별 루프 FFT): ~50ms/파일
- 벡터화 FFT (stride_tricks): ~10ms/파일 (5x 빠름)

---

## 파일 구성

```
rk3588/
├── fbank.py                    # Kaldi-호환 80-bin fbank (numpy 벡터화, preemphasis=0.97)
├── convert_encoder.py          # Encoder ONNX → RKNN FP16 변환
├── convert_encoder_int8.py     # Encoder ONNX → RKNN INT8 변환 (ONNX 캘리브레이션)
├── convert_decoder_joiner.py   # Decoder/Joiner ONNX → RKNN 변환
├── inference_onnx.py           # ONNX 추론 (INT8 기본, 4-thread) ← 권장
├── inference_hybrid.py         # RKNN Hybrid 추론 (RKNN encoder_out + ONNX 캐시)
├── inference_rknn.py           # RKNN Pure 추론 (cumfix.rknn 사용 시 CER 29.32%)
├── fix_cumsum.py               # CumSum → MatMul 패치 스크립트
├── convert_encoder_cumfix.py   # cumfix ONNX → RKNN FP16 변환
├── eval_cer.py                 # CER 종합 평가 (ONNX FP32/INT8 + Hybrid + Pure)
├── bench_final.py              # 최종 성능 벤치마크
├── debug_cache2.py             # 캐시 발산 분석 도구
├── debug_per_chunk.py          # 청크별 ONNX vs RKNN 비교
├── bench_hybrid_timing.py      # Hybrid 레이턴시 분석
├── bench_rknn_reuse.py         # RKNN 입력 재사용 테스트
└── encoder-epoch-99-avg-1.rknn         # RKNN FP16 (148MB)
└── encoder-epoch-99-avg-1-int8.rknn    # RKNN INT8 (80MB)
└── decoder-epoch-99-avg-1.rknn
└── joiner-epoch-99-avg-1.rknn
```

---

## 권장 추론 방법

**속도 우선: RKNN rmreshape + C API (33ms/chunk)**
```python
from encoder_capi import EncoderCAPI
enc = EncoderCAPI('rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn')
cache = enc.init_cache()
enc_out, cache = enc.run(x_nhwc, cache)  # 33ms
```

**정확도 우선: ONNX INT8 (35ms/chunk, CER 19.95%)**
```python
from inference_onnx import ZipformerONNX
model = ZipformerONNX(use_int8=True)
stats = model.transcribe('/path/to/audio.wav')
```

---

## 향후 과제

1. **CER 개선**: RKNN INT8 양자화 정확도 개선 (26.25% → 목표 <22%)
   - `quantized_algorithm='mmse'` (메모리 부족으로 미완료)
   - hybrid quantization (민감 레이어만 FP16)
2. **kaldifeat 설치**: 더 정확한 fbank → CER 추가 개선
3. **대용량 평가**: 4개 test_wavs 외 KsponSpeech 전체로 CER 측정
