# wav2vec2 RK3588 NPU 포팅

한국어 wav2vec2-xls-r-300m 모델을 RK3588 NPU(RKNN FP16)에서 실시간 실행.

## 성능

| 방식 | 속도 (5초 오디오) | RTF | CPU 대비 | 비고 |
|------|:---:|:---:|:---:|------|
| **RKNN FP16 3-core** | **476ms** | **0.095** | **6.9x** | C API set_io_mem |
| RKNN FP16 core0 | 722ms | 0.144 | 4.6x | C API set_io_mem |
| ONNX FP32 CPU 4t | 3291ms | 0.658 | 1.0x | onnxruntime |
| ONNX FP32 CPU 1t | 7047ms | 1.409 | 0.5x | onnxruntime |

> **3코어 NPU로 CPU 대비 6.9배 빠름** (476ms vs 3291ms, RTF 0.095)

### INT8 양자화 시도 결과

| 방식 | 크기 | 결과 |
|------|:---:|------|
| INT8 normal | 320MB | 전 프레임 `<pad>` 출력 (실패) |
| INT8 KL divergence | 320MB | 전 프레임 `<pad>` 출력 (실패) |
| auto_hybrid (cos=0.98) | 570MB | "네네네네네다" (부분 실패) |
| auto_hybrid (cos=0.90) | 446MB | 전 프레임 `<pad>` 출력 (실패) |
| auto_hybrid (cos≥0.995) | - | 빌드 중 segfault (RKNN SDK 한계) |
| quantized_hybrid_level | - | 빌드 2시간+ 후 결과 없음 |

> wav2vec2의 LayerNorm(114개) + Softmax(24개) + GELU(32개)가 INT8에 극도로 민감.
> RKNN SDK의 mixed precision도 이 문제를 해결하지 못함. **FP16이 유일한 실용적 선택지.**

## 모델

| 파일 | 크기 | 설명 |
|------|:---:|------|
| `wav2vec-xls-r-300m_5s.onnx` | 1.2GB | ONNX FP32, 5초 입력 |
| `wav2vec-xls-r-300m_5s_fp16.rknn` | 625MB | RKNN FP16, 5초 입력 |
| `wav2vec-xls-r-300m_20s.onnx` | 1.2GB | ONNX FP32, 20초 입력 |
| `wav2vec-xls-r-300m_20s_fp16.rknn` | 653MB | RKNN FP16, 20초 입력 |

모델 파일은 GitHub 100MB 제한으로 레포에 미포함. 별도 다운로드 필요.

- 원본: [wav2vec2-xls-r-300m-korean](https://huggingface.co/) (HuggingFace)
- 입력: `[1, 80000]` (5초 × 16kHz) 또는 `[1, 320000]` (20초 × 16kHz)
- 출력: `[1, 249, 2617]` (CTC logits, 2617 한국어 토큰)

## 사용법

### RKNN 추론 (NPU)

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
│   └── bench_rknn.py            # C API 벤치마크
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
- INT8 양자화는 RKNN SDK에서 wav2vec2 구조와 호환 불가 (LayerNorm + Softmax + GELU 조합이 양자화에 극도로 민감)
- `remove_reshape=True`는 입력 layout 문제(NCHW)로 사용 불가
