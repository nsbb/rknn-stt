# rknn-stt

RK3588 NPU(RKNN)에서 한국어 음성인식(STT) 모델을 실시간으로 실행하기 위한 포팅 프로젝트.

## 현재 성능 (Zipformer)

| 항목 | 값 |
|------|-----|
| 모델 | Zipformer Streaming Transducer (Korean) |
| Encoder 속도 | **27.5ms/chunk** (NPU single core) |
| RTF | **~0.10** (10배 실시간) |
| CER | 22.97% (test_wavs 4파일 기준) |
| 비교: ONNX INT8 CPU | 35ms/chunk, RTF 0.13 |

RKNN이 ONNX CPU 대비 **22% 빠름**.

## 프로젝트 구조

```
rknn-stt/
├── zipformer/                  # Streaming Zipformer Transducer
│   ├── rk3588/                 # RKNN 포팅 코드
│   │   ├── RESULTS.md          # 전체 성능 벤치마크
│   │   ├── WHY_ONNX_BEATS_RKNN.md  # 병목 분석
│   │   ├── fix_cumsum.py       # CumSum → MatMul 패치
│   │   ├── inference_onnx.py   # ONNX 추론
│   │   ├── inference_rknn.py   # RKNN NPU 추론
│   │   ├── fbank.py            # Feature extraction
│   │   ├── build_nocache_static.py  # RKNN 변환
│   │   └── bench_*.py          # 벤치마크 스크립트
│   ├── encoder-epoch-99-avg-1.onnx  # HuggingFace에서 다운로드 필요
│   ├── decoder-epoch-99-avg-1.onnx
│   ├── joiner-epoch-99-avg-1.onnx
│   ├── tokens.txt
│   └── test_wavs/              # 테스트 음성 샘플
└── wav2vec2/                   # (미착수)
```

## 빠른 시작

### 1. Encoder 다운로드

Encoder ONNX는 용량 초과(279MB)로 레포에 미포함:

```bash
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.onnx -P zipformer/
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.int8.onnx -P zipformer/
```

### 2. ONNX 추론 (CPU)

```bash
cd zipformer/rk3588
python inference_onnx.py
```

### 3. RKNN 변환 (RK3588 보드에서)

```bash
# CumSum 패치 적용
python fix_cumsum.py

# RKNN INT8 변환
conda run -n RKNN-Toolkit2 python build_nocache_static.py
```

### 4. RKNN NPU 추론

```bash
python inference_rknn.py
```

## RKNN 포팅 핵심 기법

### CumSum 버그 패치

RKNN의 CumSum 연산이 non-zero 초기 상태에서 잘못 계산되는 버그 발견.
15개 CumSum 노드를 하삼각 행렬 MatMul로 교체하여 해결.

```
CER: 91.89% → 22.97% (INT8)
```

### 속도 최적화 과정

```
rknnlite inputs_set       52.7ms
→ C API set_io_mem        39.2ms  (-13.5ms, zero-copy)
→ remove_reshape=True     30.7ms  (-8.5ms, dispatch 감소)
→ nocache-static 변환     27.5ms  (-3.2ms, 최적)
```

### 병목 분석

RKNN 내부에서 ~4832개 레이어로 분해됨. 레이어당 dispatch overhead ~5.9us.
**실제 NPU 연산은 2.6ms(9%)이고, 나머지 91%가 dispatch overhead.**

SDK 옵션 15가지 시도 (flash-attn, sparse, pruning, 양자화 방법 변경 등) → 모두 속도 무영향.
속도 개선은 모델 아키텍처 변경(레이어 수 감소)으로만 가능.

### 레이어 프루닝 실험

| 구성 | 레이어 수 | RKNN 속도 | CER |
|------|----------|-----------|-----|
| 2,4,3,2,4 (원본) | 15 | 27.5ms | 22.97% |
| 2,3,2,2,3 | 12 | 26.7ms | 93.75%* |
| 1,3,2,1,3 | 10 | 23.3ms | 90.00%* |
| 1,2,2,1,2 | 8 | **19.9ms** | 143.75%* |

*Fine-tuning 없이 weight pruning만 적용. Fine-tuning 시 CER 회복 예상.

## 환경

- **보드**: RK3588 (NPU 1GHz, 3코어)
- **SDK**: RKNN-Toolkit2 2.3.2, librknnrt 2.3.2
- **원본 모델**: [icefall Korean Zipformer](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12)

## 상세 문서

- [RESULTS.md](zipformer/rk3588/RESULTS.md) — 전체 벤치마크 결과
- [WHY_ONNX_BEATS_RKNN.md](zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md) — 병목 분석 상세
