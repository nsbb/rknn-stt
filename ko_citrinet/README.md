# Korean CitriNet — RK3588 NPU 포팅

NeMo 기반 한국어 CitriNet CTC 모델을 RK3588 NPU에서 실행.

---

## 성능

| 메트릭 | 값 |
|--------|------|
| **추론 지연** | **52.5ms** (3초 오디오, NPU 3코어) |
| **RTF** | **0.0175** (57배 실시간) |
| **ONNX↔RKNN Cosine** | **0.999935** |
| 모델 크기 | 281MB (FP16) |
| 입력 | mel spectrogram `[1, 80, 300]` (3초, 80 mel bins) |
| 출력 | logits `[1, 2049, 38]` (2048 BPE + blank, 38 time steps) |

---

## 인식 결과

```
[57ms] call_elevator.wav: 엘리베이터블러서
[78ms] check_weather.wav: 날씨알려줬서
[53ms] sample.wav:         조명이 켜졌습니다
[97ms] turn_on_light.wav:  불켜서
```

4개 테스트 전부 ONNX 결과와 **100% 일치**.

---

# ⚠️ RKNN 버그 발견 — Squeeze

> ### **RKNN은 Squeeze/Unsqueeze를 잘못 처리한다**
>
> NCHW 텐서에 `Squeeze(axis=2)`를 적용하면,
> RKNN 내부에서 **NHWC 레이아웃의 axis에 적용**하여
> 출력 데이터가 **완전히 뒤섞인다**.
>
> ```
> 입력: [1, 80, 1, 300]  (NCHW)
> 기대: [1, 80, 300]     (axis=2 제거)
> RKNN: 첫 번째 값만 맞고 나머지 전부 틀림 (cosine ≈ -0.01)
> ```
>
> **Reshape으로 교체해도 동일하게 실패.**
>
> ### 해결: 모델 I/O를 3D로 변경하여 Squeeze/Unsqueeze 완전 제거

이 버그 때문에 모델 출력이 **all-blank** (전부 `<blank>` 토큰)으로 나옴.
Random input에서는 cosine 0.997로 정상처럼 보이지만,
**실제 음성 입력에서는 cosine 0.42로 완전히 깨진다.**

---

## 적용한 ONNX 그래프 수정 (4가지)

`python/fix_citrinet_graph.py` 참조.

### Fix 1 — LogSoftmax 제거

RKNN 빌드 시 `swap_transpose_logsoftmax` 규칙에서 `ValueError: -1 is not in list` 크래시 발생.
CTC greedy decoding에는 argmax만 필요하므로 LogSoftmax 자체가 불필요 → 제거.

### Fix 2 — 마스크 SE 블록 → ReduceMean

23개 SE 블록의 마스크 체인 (ConstantOfShape/Equal/Less/Not/Cast/Where/ReduceSum/Div) 제거.
고정 길이 입력(300 frames)에서는 마스크가 항상 all-True이므로 ReduceMean과 동등.
**530 노드 제거** (1513 → 983).

### Fix 3 — ReduceMean → depthwise Conv

RKNN의 알려진 ReduceMean 버그 ([wakeword](../../../rknn-wakeword/docs/RKNN_PORTING_GUIDE.md) 프로젝트에서 발견).
`ReduceMean(axis=-1)` on `[1, C, T]` → depthwise `Conv(kernel=[T], w=1/T, group=C)` 로 교체.
ORT로 중간 텐서 shape를 추출하여 각 블록의 T 값 결정 (300/150/75/38).

### Fix 4 — Squeeze/Unsqueeze 제거 (**핵심**)

위에서 설명한 RKNN Squeeze 버그의 우회.
모델 입력을 `[1, 80, 1, 300]` → `[1, 80, 300]` (3D)으로 변경.
모델 출력을 `[1, 2049, 1, 38]` → `[1, 2049, 38]` (3D)으로 변경.

---

## 모델 구조

**CitriNet** = Jasper 기반 순수 CNN CTC 모델 (Transformer 없음).

```
Input mel [1, 80, 300]
  ↓
Prologue Conv (80 → 1024)
  ↓
22x Jasper Block:
  ├── Depthwise Separable Conv (1D)
  ├── SE (Squeeze-and-Excitation) block
  │     Conv → ReduceMean → FC → ReLU → FC → Sigmoid → Mul
  └── Residual connection + ReLU
  ↓
Decoder Conv (1024 → 2049)
  ↓
Output logits [1, 2049, 38]  →  CTC greedy decode  →  텍스트
```

- 토크나이저: SentencePiece BPE (2048 토큰 + blank)
- 입력: 16kHz 오디오 → 80-dim mel spectrogram (25ms window, 10ms hop)
- 고정 입력: 300 frames = 3초

---

## 사용법

### 변환 (ONNX → RKNN)

```bash
# 1. ONNX 그래프 수정
conda run -n RKNN-Toolkit2 python python/fix_citrinet_graph.py citrinet_npu_v2_fixlen.onnx citrinet_npu_v2_fixlen_fixed.onnx

# 2. RKNN FP16 변환
conda run -n RKNN-Toolkit2 python python/convert_fp16.py citrinet_npu_v2_fixlen_fixed.onnx model/citrinet_fp16.rknn
```

### 추론

```bash
# 단일 파일
conda run -n RKNN-Toolkit2 python python/inference_rknn.py path/to/audio.wav

# 벤치마크
conda run -n RKNN-Toolkit2 python python/inference_rknn.py --bench
```

---

## 파일 구조

```
ko_citrinet/
├── python/
│   ├── fix_citrinet_graph.py   # ONNX 그래프 수정 (4가지 fix)
│   ├── convert_fp16.py         # RKNN FP16 변환
│   └── inference_rknn.py       # NPU 추론 + mel 전처리 + CTC 디코딩
├── model_config_ko.yaml        # NeMo 모델 설정
├── tokenizer.model             # SentencePiece 토크나이저
├── vocab_ko.txt                # 한국어 BPE 어휘
└── model/                      # RKNN 모델 (gitignore)
    └── citrinet_fp16.rknn      # 281MB
```
