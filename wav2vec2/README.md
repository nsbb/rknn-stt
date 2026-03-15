# wav2vec2 RK3588 NPU 포팅

한국어 wav2vec2-xls-r-300m 모델을 RK3588 NPU(RKNN FP16)에서 실시간 실행.

## 성능

| 방식 | 속도 (5초 오디오) | RTF | 비고 |
|------|:---:|:---:|------|
| **RKNN FP16 NPU** | **720ms** | **0.144** | C API set_io_mem, core0 |
| ONNX FP32 CPU 4t | 3291ms | 0.658 | onnxruntime |
| ONNX FP32 CPU 1t | 7047ms | 1.409 | onnxruntime |

> RKNN NPU가 CPU 대비 **4.6배 빠름** (720ms vs 3291ms)
>
> INT8 양자화는 wav2vec2에서 공식 미지원 (GroupNorm 등 양자화 민감 연산)

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

## 참고

- wav2vec2는 non-streaming (전체 오디오를 한번에 처리)
- Zipformer와 달리 CumSum 등 RKNN 버그 없음
- INT8 양자화 시 출력이 상수값으로 고정됨 — rknn_model_zoo 공식 예제도 FP16만 지원
