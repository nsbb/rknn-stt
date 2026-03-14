# 왜 RKNN이 ONNX보다 느린가 — Zipformer Encoder 분석

## 결론 요약

| 방식 | Encoder/청크 | RTF | CER |
|------|-------------|-----|-----|
| ONNX FP32 (4-thread) | 46ms | 0.182 | 19.95% |
| **ONNX INT8 (4-thread)** | **35ms** | **0.130** | **19.95%** |
| RKNN FP16 | 63ms | — | — |
| RKNN Hybrid (RKNN+ONNX) | 105ms | 0.349 | 19.95% |

**RKNN은 ONNX보다 1.5~3배 느리다.**

---

## 원인: 데이터 전송 오버헤드

### 측정값

```
RKNN inference() 호출 시간: 63ms
  └─ NPU 순수 계산 시간:    2.6ms  ← NPU는 충분히 빠름
  └─ 데이터 전송 오버헤드:  ~60ms  ← 병목
```

벤치마크 방법:
- `make_dummy_inputs()` 로 pre-built 입력을 **재사용**: 2.6ms (전송 없음)
- 매번 새 배열로 `pack_rknn_inputs()` 호출: 63ms (매번 DMA 전송)

### 왜 전송이 오래 걸리나

Zipformer encoder는 입력이 **36개 텐서, 합계 2.0MB**다.

```
cached_conv1_0~4:  각 [2,1,384,30] × 5 = 460KB
cached_conv2_0~4:  각 [2,1,384,30] × 5 = 460KB
cached_key_0~4:    각 [2~4,32~64,1,192] × 5 = 470KB
cached_val_0~4:    각 [2~4,32~64,1,96]  × 5 = 230KB
cached_val2_0~4:   각 [2~4,32~64,1,96]  × 5 = 230KB
cached_avg_0~4:    각 [2~4,1,384]       × 5 = 20KB
cached_len_0~4:    각 [2~4,1] int64     × 5 ≈ 0KB
x:                 [1,39,80]                  = 12KB
───────────────────────────────────────────────────────
합계:                                       ≈ 2,056 KB
```

RK3588 NPU DMA 전송 속도: 2MB / 60ms ≈ **33 MB/s** (PCIe/AHB 버스 기준)

이 속도는 NPU의 연산 처리 속도와 전혀 무관하다. 입력을 NPU 메모리로 복사하는 시간이 실제 연산 시간(2.6ms)의 23배다.

### Wakeword(BCResNet-t2)와 비교

wakeword 모델은 단일 입력 `[1, 1, 128, 157]` = 80KB → 전송 오버헤드 무시 가능.
Zipformer encoder는 36개 입력, 2MB → 전송 오버헤드가 압도적.

**NPU가 STT에서 느린 근본 원인: Transformer 계열 캐시 heavy 모델 구조**

---

## RKNN Hybrid가 더 느린 이유

Hybrid 방식은 encoder_out 정확도를 위해 RKNN과 ONNX를 **둘 다** 실행한다:

```
청크당:
  RKNN inference: 63ms  (encoder_out 취함)
+ ONNX inference: 40ms  (캐시 업데이트 취함)
= 총 103ms/청크

반면 ONNX만: 35ms/청크 (INT8 기준)
```

Python threading으로 병렬화를 시도했으나 **GIL 때문에 오히려 130ms**로 더 느려짐.

---

## RKNN Pure가 틀린 이유: CumSum 버그

### 증상

```
Chunk 0: encoder_out max_diff=0.0088 ✓  (캐시 모두 0일 때는 정확)
Chunk 1: cached_avg_4 diff=44.0073      ← 200배 오차!
Chunk 2: cached_conv2_4 diff=159.85, encoder_out diff=3.52  (파국)
```

### 원인

RKNN의 `CumSum` 연산이 **non-zero 초기값**을 받으면 잘못 계산함.

`new_cached_avg` 계산 그래프:
```
CumSum(현재 청크 데이터)
 → Add(누적합 + prev_avg * prev_len)  ← prev_avg ≠ 0이면 오차 시작
 → Mul(1 / new_len)
 → Gather(마지막 프레임)
 = new_cached_avg
```

chunk 0에서는 `prev_avg = 0` → CumSum 오차가 hidden. chunk 1부터 `prev_avg ≠ 0` → 오차 폭발.

### 같은 문제가 생기는 다른 모델

streaming 방식으로 누적 통계(평균, 분산)를 캐시로 유지하는 모델 전반.
CumSum, ReduceMean, ReduceSum 조합이 non-zero 입력에서 문제를 일으킴.

---

## RKNN으로 빠르게 만들려면

현재 구조에서 RKNN을 ONNX보다 빠르게 만들기 위한 방향:

### 방향 1: 입력 텐서 수/크기 축소

캐시 텐서를 합쳐서 전송량 줄이기:
- 예: `cached_conv1_0~4` (5개) → 하나의 큰 텐서로 concat → 1번 전송
- ONNX 그래프 수정 필요 (입력 이름/형태 변경)
- 이론상 36번 전송 → 몇 번으로 줄일 수 있음

### 방향 2: 상태를 NPU 메모리에 유지 (Zero-copy)

- RKNN의 `set_io_mem` API를 통해 NPU 내부 메모리 버퍼를 직접 재사용
- 캐시 출력 텐서 → 다음 청크 입력으로 CPU 경유 없이 NPU 내에서 재사용
- 가능하다면 전송 오버헤드 거의 제거 가능
- 구현 복잡도 높음, RKNN C API 필요 (Python 바인딩 미확인)

### 방향 3: Encoder 분리

Encoder를 두 부분으로 분리:
1. **캐시 의존 없는 부분** (Conv, linear 등) → RKNN으로 실행
2. **캐시 업데이트 부분** (CumSum 포함) → ONNX로 실행

전송 대상 텐서를 크게 줄일 수 있지만 ONNX 그래프 분리 작업 필요.

### 방향 4: CumSum 패치 후 pure RKNN

ONNX 그래프에서 CumSum 노드를 RKNN 지원 ops로 교체:
- 참고: `rknn-wakeword/fix_rknn_graph.py` (ReduceMean 교체 패턴)
- CumSum → 동등한 Conv/Matmul 표현으로 변환
- 성공 시 pure RKNN 추론 가능
- **단, 전송 오버헤드(60ms)는 여전히 존재** → ONNX보다 여전히 느릴 가능성

### 방향별 예상 효과

| 방향 | 기대 효과 | 난이도 |
|------|-----------|--------|
| 1. 텐서 concat | 전송 횟수 감소, 속도 약간 개선 | 중 |
| 2. Zero-copy NPU 메모리 | 전송 오버헤드 대폭 감소 | 높음 |
| 3. Encoder 분리 | 전송 데이터 감소 | 중 |
| 4. CumSum 패치 | Pure RKNN 가능, 전송은 여전히 문제 | 중 |
| **2+4 조합** | **전송 제거 + 정확한 캐시 → RKNN 최속** | 높음 |

---

## 현재 권장 사항

```python
# 최적: ONNX INT8, 4-thread
from inference_onnx import ZipformerONNX
model = ZipformerONNX(use_int8=True)
stats = model.transcribe('audio.wav')
# RTF ≈ 0.130, CER 19.95%
```

RKNN으로 더 빠르게 만들려면 **방향 2 (Zero-copy)** 또는 **방향 1+4 조합** 추진 필요.
