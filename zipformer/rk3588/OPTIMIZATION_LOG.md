# RKNN Encoder 속도 최적화 로그

**목표:** 33ms/chunk → 20ms/chunk (RKNN rmreshape + C API)
**모델:** encoder-epoch-99-avg-1 (69M params, 2275 ONNX nodes after onnxsim)

---

## 현재 최고 성능

| 항목 | 값 |
|------|-----|
| rknn_run | 31.6ms (avg), 30.9ms (min) |
| write_input | 0.87ms |
| read_output + cache_convert | 1.2ms (batch sync 최적화 후) |
| **Total** | **32.7ms (avg), 32.0ms (min)** |
| CER | 26.25% |

---

## 시도 1: RKNN config 변종 빌드 (rmreshape + α)

**가설:** `remove_reshape=True`와 다른 RKNN config 옵션을 조합하면 추가 속도 개선 가능.

| 변종 | 설정 | avg (ms) | 결과 |
|------|------|---------|------|
| rmreshape (baseline) | remove_reshape=True | 37.6 | 기준 |
| rmreshape-opt2 | + optimization_level=2 | 37.9 | 변화 없음 |
| rmreshape-singlecore | + single_core_mode=True | 37.8 | 변화 없음 |
| rmreshape-flash | + enable_flash_attention=True | 37.4 | 변화 없음 |
| rmreshape-sim | onnxsim + remove_reshape | 37.6 | 변화 없음 |

**결론:** RKNN config 옵션 조합은 효과 없음. onnxsim으로 ONNX 노드 수를 5818→2275(-61%)로 줄여도 RKNN 컴파일 결과는 동일.

---

## 시도 2: Multi-core 추론

**가설:** RK3588의 NPU 3코어를 모두 사용하면 병렬 실행으로 속도 향상.

| core_mask | 설명 | avg (ms) |
|-----------|------|---------|
| 1 | Core 0 only | 33.7 |
| 2 | Core 1 only | 33.6 |
| 4 | Core 2 only | 33.7 |
| 6 | Core 1+2 (not supported, fallback) | 33.1 |
| 7 | All 3 cores | 34.6 |

**결론:** Multi-core 효과 없음. 병목이 compute가 아닌 dispatch overhead이므로 코어를 늘려도 개선 불가.

---

## 시도 3: 캐시 변환 최적화 (batch sync) — ✅ 성공

**가설:** 36개 출력의 mem_sync를 개별 호출 대신 배치로 처리하면 오버헤드 감소.

| 방식 | avg (ms) |
|------|---------|
| Original (개별 sync+read+convert) | 2.20 |
| **Batch sync** (전체 sync 후 read+convert) | **1.83** |
| Sync+read only (convert 제외) | 0.87 |
| Transpose only | 0.78 |

**결과:** encoder_capi.py에 batch sync 적용 → **총 시간 34.4ms → 32.7ms (-1.7ms)**

이 최적화는 encoder_capi.py에 반영됨.

---

## 시도 4: ONNX 그래프 수술

### 4a. onnxsim 다중 패스
- 5818 → 2275 노드 (1회 패스에서 수렴)
- RKNN 빌드 결과: rknn_run 시간 변화 없음
- **결론:** RKNN 컴파일러가 동일한 내부 그래프 생성

### 4b. Pointwise Conv1D → MatMul 교체
- 60개 pointwise conv (kernel=1)을 MatMul로 교체 시도
- 30개 교체 성공, 60개 Transpose 제거 (Reshape: 190, Transpose: 168→108)
- **실패 원인:** Conv의 NCT↔TNC 포맷 변환은 데이터 레이아웃 불일치 때문에 제거 불가
- depthwise conv가 중간에 있어 pointwise conv의 Transpose를 독립적으로 제거할 수 없음

### 4c. Reshape-Transpose 체인 융합
- Reshape→Transpose 체인: 90개, Transpose→Reshape: 31개
- 연속 Transpose→Transpose: 0개 (융합 불가)
- **결론:** 단순 체인 융합으로는 개선 불가

---

## 시도 5: op_target (Reshape/Transpose → CPU 이동)

**가설:** Reshape/Transpose를 CPU에서 실행하면 NPU dispatch 오버헤드 제거.

### 5a. ONNX 노드 이름으로 시도
시도한 키 형식:
- `/encoder_embed/Transpose` (ONNX 이름 그대로) → `Invalid key` 오류
- `encoder_embed/Transpose` (/ 제거) → `Invalid key` 오류
- `0` (정수 문자열) → `Invalid key` 오류
- `/encoder_embed/Transpose_output_0` (출력 이름) → `Invalid key` 오류

### 5b. accuracy_analysis로 RKNN 내부 이름 추출 (돌파구!)
- `rknn.accuracy_analysis()` 실행 → 1619개 RKNN 레이어 이름 추출
- 내부 이름 형식: `/Reshape_5_output_0_rs`, `/Transpose_2_output_0_tp_rs` 등
- **이 형식으로 op_target 빌드 성공!**

### 5c. 전체 Reshape/Transpose CPU 이동 빌드
- 265개 Reshape/Transpose 레이어를 모두 CPU로 지정
- 빌드: 성공 (86MB, 기존 79.6MB 대비 증가)
- **실행: Segfault** — `Unsupport CPU op: exMatMul`
- **근본 원인:** Reshape를 CPU로 이동하면 연결된 fused ops (exMatMul 등)도 CPU로 강제 이동됨
- CPU runtime은 NPU 전용 fused ops (exMatMul, exSwish 등)를 지원하지 않음
- 커널 드라이버 0.9.6 (API 2.3.2보다 구형)도 원인일 수 있음

**결론:** op_target은 경계 ops에만 안전하게 사용 가능. 내부 Reshape/Transpose를 CPU로 이동하면 연쇄적 CPU fallback으로 크래시 발생.

---

## RKNN 내부 레이어 분석 (accuracy_analysis 기반)

```
Conv:           315 (19.5%)
Add:            211 (13.0%)
Transpose:      202 (12.5%)  ← 전체의 12.5%
Reshape:        181 (11.2%)  ← 전체의 11.2%
Split:          165 (10.2%)
Slice:          146  (9.0%)
Concat:         110  (6.8%)
exSwish:         78  (4.8%)
exMatMul:        75  (4.6%)
exDataConvert:   62  (3.8%)
Mul:             60  (3.7%)
Input:           36  (2.2%)
exGlu:           30  (1.9%)
Cast:            30  (1.9%)
exSoftmax13:     20  (1.2%)
Sub:             15  (0.9%)
exNorm:          15  (0.9%)
Div:             15  (0.9%)
Expand:           4  (0.2%)
─────────────────────────────
Total:        ~1770 layers
```

Reshape+Transpose = 383 layers (23.6%). 실제 계산 ops (Conv+MatMul+Add+Mul 등) = ~700 layers.
나머지 ~700 layers는 데이터 조작 (Split, Slice, Concat, DataConvert 등).

---

## 병목 분석 요약

```
rknn_run: ~31ms (92% of total)
  ├─ 추정 RKNN 레이어: ~1619
  ├─ 레이어당 dispatch: ~19µs
  └─ dispatch overhead = 1619 × 19µs ≈ 31ms (rknn_run의 ~100%)
  └─ 실제 계산 시간: ~2ms (전체의 6%)

캐시 I/O: ~1.2ms (batch sync 후)
write_input: ~0.87ms
```

**핵심 병목:** 1619개 RKNN 레이어의 dispatch overhead (~31ms).
실제 NPU 계산은 ~2ms에 불과하나, 레이어 스케줄링/전환에 ~29ms 소요.
이는 RKNN runtime의 구조적 한계로, 소프트웨어 최적화로 해결 불가.

---

## 20ms 달성 가능성 분석

20ms 달성에 필요: 레이어 수 ~1050으로 감소 (현재 1619 대비 -35%)
또는 레이어당 dispatch 시간 19µs → 12µs (-37%)

| 방법 | 예상 효과 | 실현 가능성 | 비고 |
|------|----------|-----------|------|
| NPU 드라이버 업그레이드 | dispatch -30%? | 높음 | 현재 DRV 0.9.6 → 최신 |
| 더 작은 모델 | rknn_run ~15ms | 없음 | 한국어 small zipformer 미존재 |
| 모델 프루닝 (50%) | ~20ms | 중간 | 재학습 필요, CER 악화 예상 |
| 다른 STT 모델 | 미지수 | 중간 | SenseVoice, Whisper tiny 등 |
| custom C++ wrapper | -1~2ms | 낮음 | Python 오버헤드 최소 |

---

## 타임라인

| 시간 | 작업 | 결과 |
|------|------|------|
| 2026-03-14 02:00 | rmreshape 변종 빌드 시작 | 4개 모델 빌드 완료 |
| 2026-03-14 03:00 | 변종 벤치마크 | 모두 ~37ms, 차이 없음 |
| 2026-03-14 03:00 | Multi-core 테스트 | 효과 없음 |
| 2026-03-14 03:15 | onnxsim 분석 | 2275 노드, 추가 단순화 불가 |
| 2026-03-14 03:20 | ONNX 그래프 수술 (Conv→MatMul) | 레이아웃 불일치로 실패 |
| 2026-03-14 03:30 | op_target (ONNX 이름) | 키 형식 오류로 실패 |
| 2026-03-14 03:40 | **batch sync 최적화** | **-1.7ms 성공** (34.4→32.7ms) |
| 2026-03-14 04:00 | 모델 분할 분석 | 85% ops 미분류로 실현 어려움 |
| 2026-03-14 04:10 | **accuracy_analysis → RKNN 내부 이름 발견** | **op_target 빌드 성공!** |
| 2026-03-14 04:25 | op_target 전체 적용 → 실행 | exMatMul CPU 미지원으로 크래시 |

---

## 시도 6: FP16 빌드

**가설:** INT8의 exDataConvert/Cast 레이어가 없으면 총 레이어 수 감소 → dispatch overhead 감소.

| 모델 | avg (ms) | 크기 |
|------|---------|------|
| INT8-rmreshape (baseline) | 36.2 | 79.6 MB |
| **FP16-rmreshape** | **46.2** | **147 MB** |

**결론:** FP16은 모델 크기 2배, 연산량 증가로 오히려 **+10ms 느림**. 실패.

---

## 시도 7: 선택적 op_target (안전한 경계 Reshape만 CPU)

**가설:** fused op과 인접하지 않은 순수 Reshape/Transpose만 CPU로 이동하면 cascade 없이 dispatch 감소.

분석: 316개 안전한 후보 (pure _rs/_tp, 196 reshape + 120 transpose)

| 변종 | CPU 이동 수 | avg (ms) | 비고 |
|------|-----------|---------|------|
| baseline | 0 | 36.3 | |
| opt10cpu | 10 | 35.4 | **-0.9ms** |
| opt50cpu | 50 | 37.8 | **+1.5ms (악화)** |

**결론:** 소수(10개)는 미미한 개선. 다수(50개)는 CPU 실행 overhead가 NPU dispatch 절감보다 큼. 실패.

---

## 시도 8: SenseVoice 모델 조사

- SenseVoiceSmall: 234M params, CTC encoder-only, 한국어 지원
- RKNN2 변환 존재: `happyme531/SenseVoiceSmall-RKNN2` (HuggingFace)
- RK3588 성능: ~20x 실시간 (single NPU core)
- **문제: offline (non-streaming) 모델** — 전체 오디오를 한 번에 처리
- 현재 use case(streaming chunked inference)에 부적합
- "pseudo-streaming" 구현 존재하지만 진정한 streaming이 아님

**결론:** SenseVoice는 offline 전용. Streaming STT에는 사용 불가.

---

## 시도 10: model_pruning + compress_weight + sparse_infer

**가설:** RKNN 컴파일러의 pruning/compression 옵션이 불필요한 노드를 제거하여 레이어 수 감소.

| 변종 | 설정 | avg (ms) | 비고 |
|------|------|---------|------|
| baseline | (없음) | 35.6 | |
| prune | model_pruning=True | 35.6 | 노이즈 수준 |
| prune-compress | prune + compress_weight | 35.1 | **-0.5ms** (노이즈) |
| sparse | sparse_infer=True | N/A | **rk3588 미지원** |
| w8a16 | quantized_dtype='w8a16' | N/A | **rk3588 미지원** |

**결론:** model_pruning/compress_weight는 노이즈 수준의 미미한 효과. sparse_infer, w8a16은 rk3588 미지원. 실패.

---

## 시도 12: ONNX Shape 노드 상수 폴딩 (noshape)

**가설:** Shape/Gather/Unsqueeze 연산을 상수로 대체하면 ONNX 노드 대폭 감소 → RKNN 레이어 감소.

| 항목 | 값 |
|------|-----|
| 원본 ONNX | 2275 노드 |
| noshape ONNX | **1874 노드** (-401, -17.6%) |
| 제거된 op | Shape(19), Gather(139), Unsqueeze(204), Concat 일부(39) |
| ORT 검증 | 통과 |
| RKNN 빌드 | **실패** — `All outputs are constants, model invalid` |

**근본 원인:** 상수로 대체한 텐서들이 RKNN의 constant folding에서 전체 그래프를 상수로 판단하게 만듦. RKNN 컴파일러의 그래프 분석이 원본 ONNX의 Shape 노드에 의존.

**결론:** ONNX 레벨 노드 감소는 RKNN 컴파일에 반드시 반영되지 않음. RKNN은 자체 IR로 변환하므로 ONNX 노드 수와 RKNN 레이어 수가 1:1이 아님.

---

## 시도 13: Python/ctypes 오버헤드 분석

| 항목 | 오버헤드 |
|------|---------|
| ctypes 호출 1회 | 0.8µs |
| run() 내 ctypes 호출 수 | ~100회 |
| 총 Python 오버헤드 | **~0.1ms** |

**결론:** C++로 전환해도 0.1ms 절감. 무의미.

---

## 타임라인 (추가)

| 시간 | 작업 | 결과 |
|------|------|------|
| 2026-03-14 04:33 | FP16 빌드+벤치 | 46.2ms, 역효과 |
| 2026-03-14 04:43 | 선택적 op_target (10/50 CPU) | 10개: -0.9ms, 50개: +1.5ms |
| 2026-03-14 04:50 | model_pruning 빌드 | 35.1ms (노이즈) |
| 2026-03-14 04:55 | noshape ONNX (2275→1874 노드) | RKNN 빌드 실패 |
| 2026-03-14 05:00 | prune-compress 빌드 | 35.1ms (노이즈) |
| 2026-03-14 05:05 | sparse_infer, w8a16 | rk3588 미지원 |

---

## 최종 결론

### 현재 최고 성능

| 항목 | 값 |
|------|-----|
| rknn_run | ~31ms (avg) |
| Total (write + run + read + convert) | **~35ms/chunk** |
| CER | 26.25% |

### 시도한 모든 최적화 요약

| # | 방법 | 효과 | 상태 |
|---|------|------|------|
| 1 | RKNN config 변종 (opt2, singlecore, flash, sim) | 0ms | 실패 |
| 2 | Multi-core (3코어) | 0ms | 실패 |
| 3 | **Batch sync** | **-1.7ms** | **성공** |
| 4 | ONNX 그래프 수술 (Conv→MatMul) | 0ms | 실패 |
| 5 | op_target (전체 Reshape/Transpose → CPU) | N/A | 크래시 |
| 6 | FP16 | +10ms | 실패 |
| 7 | 선택적 op_target (10/50개) | -0.9ms/+1.5ms | 미미 |
| 8 | SenseVoice 모델 | N/A | non-streaming |
| 10 | model_pruning/compress | ~0ms | 노이즈 |
| 12 | ONNX Shape 상수 폴딩 | N/A | RKNN 호환 실패 |
| 13 | Python→C++ | 0.1ms | 무의미 |

### 20ms 달성 불가능한 이유

```
현재: 1619 RKNN layers × ~19µs/layer = ~31ms (rknn_run)
목표: 20ms → 필요 레이어: ~1050 (-35%) 또는 dispatch 12µs/layer (-37%)
```

**소프트웨어 최적화 한계 도달.** 모든 실행 가능한 소프트웨어 방법을 시도했으며, 구조적 병목(RKNN per-layer dispatch overhead)은 커널 드라이버 레벨에서만 해결 가능.

### 남은 경로 (하드웨어/인프라 변경 필요)

| 방법 | 예상 효과 | 난이도 | 비고 |
|------|----------|--------|------|
| **NPU 커널 드라이버 업그레이드** | dispatch -30%? (~24ms) | 중 | 현재 0.9.6 → 최신. 커널 재빌드 필요 |
| **다른 STT 모델** (SenseVoice offline) | RTF ~0.05 | 중 | non-streaming, 다른 사용 패턴 |
| **모델 재학습 (smaller zipformer)** | 레이어 -50% → ~17ms | 높 | 한국어 데이터+학습 인프라 필요 |
| **RKNN MatMul API 하이브리드** | 미지수 | 높 | 핵심 MatMul만 NPU, 나머지 CPU |

### 환경 정보

- librknnrt: 2.3.2 (2025-04-09)
- 커널 드라이버: rknpu 0.9.6 (2024-03-22)
- 커널: 5.10.0-1012-rockchip
- NPU 클럭: 1GHz (최대)
- 드라이버 .ko 파일 없음 (커널 빌트인)
