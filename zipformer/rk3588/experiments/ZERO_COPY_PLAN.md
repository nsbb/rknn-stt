# RKNN Zero-Copy 최적화 결과

## 최종 결과 (2026-03-13)

| 방식 | run | total | 비고 |
|------|-----|-------|------|
| inputs_set (rknnlite.api) | - | 52.7ms | 기존 방식 |
| set_io_mem + NON_CACHEABLE | - | 64.3ms | CPU 캐시 우회 → 느림 |
| set_io_mem + CACHEABLE + pt=0 | 38.8ms | 39.6ms | Step 1 |
| **set_io_mem + CACHEABLE + pt=0 (v3)** | **37.0ms** | **37.8ms** | sync 최적화 |
| **set_io_mem + selective pt=1** | **36.6ms** | **37.3ms** | Step 2 (15/35 텐서) |
| **ONNX INT8 4-thread** | - | **35ms** | 목표 |
| NPU 순수 계산 (perf_detail) | 2.6ms | - | 이론 한계 |

**결론: RKNN 37.3ms vs ONNX 35ms — 사실상 동등. 하지만 ONNX를 이기지 못함.**

---

## Step 1 결과: Cacheable 메모리 ✅

**변경:** `rknn_create_mem2(ctx, size, 0)` + `rknn_mem_sync(SYNC_TO_DEVICE)`

| 항목 | NON_CACHEABLE | CACHEABLE |
|------|---------------|-----------|
| write (memmove) | - | 0.41ms |
| sync (flush) | - | 0.25ms |
| run (DMA+NPU) | 52.5ms | 38.8ms |
| read | - | 0.10ms |
| **total** | **64.3ms** | **39.6ms** |

**성과:** 52.7ms → 39.6ms (25% 감소)
**남은 병목:** rknn_run 내부 38.8ms (NPU 2.6ms + DMA/format 변환 36ms)

## Step 2 결과: pass_through=1 (선택적) ⚠️ 효과 미미

### Native Format 호환성 조사

| 텐서 그룹 | 개수 | input fmt | output fmt | 호환 | 비고 |
|-----------|------|-----------|------------|------|------|
| cached_key | 5 | NC1HWC2 INT8 | NC1HWC2 INT8 | ✅ | 크기 동일 |
| cached_val | 5 | NC1HWC2 INT8 | NC1HWC2 INT8 | ✅ | 크기 동일 |
| cached_val2 | 5 | NC1HWC2 INT8 | NC1HWC2 INT8 | ✅ | 크기 동일 |
| cached_conv1 | 5 | NHWC INT8 (24KB) | NC1HWC2 INT8 (368KB) | ❌ | 15x 크기 차이 |
| cached_conv2 | 5 | NHWC INT8 (24KB) | NC1HWC2 INT8 (368KB) | ❌ | 15x 크기 차이 |
| cached_avg | 5 | 5D INT8 (768B) | 3D INT8 (12KB) | ❌ | 차원수 다름 |
| cached_len | 5 | INT64 (16B) | INT64 (64B) | ❌ | 크기 다름 |

**호환 가능: 15/35 (174KB / 540KB = 32%)**

### 벤치마크

| 모드 | run | total |
|------|-----|-------|
| ALL pt=0 | 37.0ms | 37.8ms |
| Selective pt=1 (15텐서) | 36.6ms | 37.3ms |

**성과:** 0.5ms 절감. format 변환 제거 효과가 미미.

### 전체 pt=1 시도 → 실패

모든 캐시에 pt=1 적용 시 **rknn_run = 6397ms** (160배 느림).
비호환 텐서의 잘못된 native format 데이터가 NPU 연산을 파괴.

## Step 3: Output→Input 메모리 공유 → 불가 ❌

cached_conv의 input(NHWC 24KB) ↔ output(NC1HWC2 368KB) 불일치로
double-buffering 자체가 불가능. 15x 크기 차이는 zero-padding (C→C2=16)이 원인.

---

## 이전 시도가 실패한 이유

모든 이전 set_io_mem 테스트에서 `RKNN_FLAG_MEMORY_NON_CACHEABLE`(flag=2) 사용.

```c
RKNN_FLAG_MEMORY_FLAGS_DEFAULT = 0    // = CACHEABLE (기본값!)
RKNN_FLAG_MEMORY_NON_CACHEABLE = 1 << 1  // 이전 테스트가 전부 이걸 사용
```

NON_CACHEABLE → CPU 쓰기 ~33MB/s → 2MB memmove ~60ms.
CACHEABLE → CPU 캐시 대역폭 ~3GB/s → 2MB memmove ~0.7ms.

---

## 근본 원인 분석

```
rknn_run 37ms 내부:
  NPU 순수 계산: 2.6ms
  DMA 전송:     ~34ms ← 여전히 병목
```

set_io_mem + CACHEABLE로 **CPU↔NPU 버퍼 간 memmove는 해결** (0.4ms).
하지만 **NPU 내부에서 rknn_run이 호출될 때 DMA 전송이 발생** (set_io_mem 버퍼 → NPU SRAM).
이 내부 DMA는 CACHEABLE/NON_CACHEABLE과 무관하게 HW bandwidth 한계.

**핵심:** set_io_mem은 CPU→NPU 버퍼 복사를 제거했지만,
NPU가 실행 시 버퍼에서 SRAM으로의 DMA는 여전히 발생.
2MB 데이터의 NPU DMA 대역폭이 ~60MB/s → ~33ms.

---

## 남은 방향

### E: Encoder 그래프 분리 (가장 유망)

cached_conv (1.35MB, 67%)를 별도 서브그래프로 분리하여 CPU에서 처리.
NPU 입력을 0.55MB로 축소 → DMA ~9ms → 총 ~12ms 예상.

**난이도:** 높음 (ONNX 그래프 수정, 두 개 모델 관리)
**리스크:** 분리 지점에서 정확도 손실 가능

### F: C API 직접 사용 (Python ctypes 오버헤드 제거)

현재 ctypes 래핑에 의한 오버헤드가 있을 수 있음.
C/C++로 직접 구현 시 추가 ~2ms 절감 가능?

### 현실적 결론

RKNN 37ms ≈ ONNX 35ms. 추가 최적화 대비 효과가 작음.
**ONNX INT8 사용 권장** — 구현 간단, 성능 최고, 정확도 최고(CER 19.95%).

---

## 테스트 파일

| 파일 | 내용 |
|------|------|
| `test_cacheable_mem.py` | Step 1: CACHEABLE vs NON_CACHEABLE vs inputs_set |
| `test_pass_through.py` | Step 2 첫 시도: 전체 pt=1 → 실패 (6397ms) |
| `test_pt1_selective.py` | Step 2: native format 호환성 조사 |
| `test_pt1_v3.py` | Step 2 최종: 선택적 pt=1 벤치마크 (37.3ms) |
