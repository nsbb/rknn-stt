# rknn-stt

RK3588 NPU(RKNN)에서 한국어 음성인식(STT) 모델을 실시간으로 실행하기 위한 포팅 프로젝트.

---

## 모델별 성능 비교

| 항목 | Zipformer | wav2vec2 |
|------|-----------|----------|
| **모델** | Streaming Zipformer Transducer | wav2vec2-xls-r-300m (3억 파라미터) |
| **모델 크기 (ONNX)** | 279MB | 1.2GB |
| **양자화** | INT8 | INT8-KL + FP16 split |
| **처리 방식** | 스트리밍 (320ms/chunk) | 1-shot (5초 고정) |
| **테스트 음성** | 2.7~6.6초 (4개, 평균 ~4초) | 5초 고정 |
| **추론 시간** | **27.5ms/chunk** | **427ms/5초** |
| **RTF** | **0.10** (10배 실시간) | **0.085** (12배 실시간) |
| **CER** | 22.97% | 35.25% |
| **NPU vs CPU** | ONNX 대비 22% 빠름 | CPU 대비 **7.7배** |

### RTF (Real-Time Factor) 설명

RTF = 처리 시간 / 음성 길이. RTF < 1.0이면 실시간보다 빠름.

- **Zipformer**: 320ms 음성 청크를 27.5ms에 처리 → RTF 0.086. E2E(decoder+joiner 포함) RTF ~0.10
- **wav2vec2**: 5초 음성을 427ms에 처리 → RTF 0.085

---

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
│   │   └── build_nocache_static.py  # RKNN 변환
│   ├── encoder-epoch-99-avg-1.onnx  # HuggingFace에서 다운로드 필요
│   ├── decoder-epoch-99-avg-1.onnx
│   ├── joiner-epoch-99-avg-1.onnx
│   ├── tokens.txt
│   └── test_wavs/              # 테스트 음성 (2.7~6.6초)
│
├── wav2vec2/                   # wav2vec2-xls-r-300m
│   ├── python/                 # 추론/변환/벤치 코드
│   │   ├── inference_split_rknn.py  # Split INT8+FP16 추론
│   │   ├── bench_rknn.py       # 속도 벤치마크
│   │   └── convert.py          # RKNN 변환
│   ├── model/                  # 모델 파일 (gitignore)
│   └── README.md               # wav2vec2 상세
```

---

## Zipformer

### 성능

| 항목 | 값 |
|------|-----|
| 양자화 | INT8 |
| Encoder 속도 | **27.5ms/chunk** (320ms 음성, NPU single core) |
| RTF (E2E) | **~0.10** (10배 실시간) |
| CER | 22.97% (test_wavs 4개, 2.7~6.6초) |
| 비교: ONNX INT8 CPU | 35ms/chunk, RTF 0.13 |

RKNN이 ONNX CPU 대비 **22% 빠름**.

### 핵심 포팅 기법

**CumSum 버그 패치**: RKNN의 CumSum이 non-zero 초기 상태에서 오계산 → 15개 CumSum 노드를 하삼각 행렬 MatMul로 교체. CER: 91.89% → 22.97%

**속도 최적화**:
```
rknnlite inputs_set       52.7ms
→ C API set_io_mem        39.2ms  (-13.5ms, zero-copy)
→ remove_reshape=True     30.7ms  (-8.5ms, dispatch 감소)
→ nocache-static 변환     27.5ms  (-3.2ms, 최적)
```

**병목**: RKNN 내부 ~4832개 레이어, 레이어당 dispatch overhead ~5.9us. 실제 NPU 연산 2.6ms(9%), 나머지 91% dispatch overhead. SDK 옵션 15가지 시도 → 모두 무영향. 개선은 모델 아키텍처 변경(레이어 수 감소)으로만 가능.

### 레이어 프루닝 실험

| 구성 | 레이어 수 | RKNN 속도 | CER |
|------|----------|-----------|-----|
| 2,4,3,2,4 (원본) | 15 | 27.5ms | 22.97% |
| 2,3,2,2,3 | 12 | 26.7ms | 93.75%* |
| 1,3,2,1,3 | 10 | 23.3ms | 90.00%* |
| 1,2,2,1,2 | 8 | **19.9ms** | 143.75%* |

*Fine-tuning 없이 weight pruning만 적용. Fine-tuning 시 CER 회복 예상.

---

## wav2vec2

### 성능

| 방식 | 속도 (5초 음성) | RTF | CER (702개) | CPU 대비 |
|------|:-:|:-:|:-:|:-:|
| **Split11 INT8-KL 3-core** | **427ms** | **0.085** | **35.25%** | **7.7x** |
| Split15 INT8-KL 3-core | 404ms | 0.081 | 37.06% | 8.2x |
| Split17 INT8-KL 3-core | 391ms | 0.078 | 37.57% | 8.4x |
| RKNN FP16 3-core | 477ms | 0.095 | 35.96% | 6.9x |
| ONNX FP32 CPU 4t | 3291ms | 0.658 | — | 1.0x |

### 핵심 포팅 기법

**Split INT8+FP16 아키텍처**: ONNX 모델을 4파트로 분할, encoder 전반부만 INT8(KL divergence) 양자화:
```
Part1(FP16) → Part2A(INT8-KL, L0-11) → Part2B(FP16, L12-23) → Part3(FP16)
features       encoder 전반              encoder 후반                lm_head
```

**KL divergence 양자화**: normal INT8 CER 43.90% → KL INT8 CER **35.25%** (8.65pp 개선, FP16 35.96%보다 좋음)

---

## 빠른 시작

### Zipformer

```bash
# Encoder 다운로드 (279MB, 레포 미포함)
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.onnx -P zipformer/
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.int8.onnx -P zipformer/

# ONNX 추론 (CPU)
cd zipformer/rk3588 && python inference_onnx.py

# RKNN 변환 + 추론 (RK3588 보드)
python fix_cumsum.py
conda run -n RKNN-Toolkit2 python build_nocache_static.py
python inference_rknn.py
```

### wav2vec2

```bash
cd wav2vec2/python

# RKNN FP16 추론
conda run -n RKNN-Toolkit2 python bench_rknn.py

# Split INT8+FP16 추론
conda run -n RKNN-Toolkit2 python inference_split_rknn.py --split split11
```

---

## 환경

- **보드**: RK3588 (NPU 1GHz, 3코어)
- **SDK**: RKNN-Toolkit2 2.3.2, librknnrt 2.3.2
- **conda env**: `RKNN-Toolkit2`
- **원본 모델**:
  - Zipformer: [icefall Korean Zipformer](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12)
  - wav2vec2: [wav2vec2-xls-r-300m-korean](https://huggingface.co/)

## 상세 문서

- [zipformer/rk3588/RESULTS.md](zipformer/rk3588/RESULTS.md) — Zipformer 전체 벤치마크
- [zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md](zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md) — Zipformer 병목 분석
- [wav2vec2/README.md](wav2vec2/README.md) — wav2vec2 상세 (INT8-KL, split 아키텍처)
