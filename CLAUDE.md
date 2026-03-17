# CLAUDE.md — rknn-stt

RK 시리즈 NPU에서 한국어 STT 모델을 실행하기 위한 포팅 작업.

## 새 세션 시작 시 필독

1. `../docs/RK3588_NPU_AI_GUIDE.md` — RKNN 범용 버그 목록, 디버깅 방법
2. `../rknn-wakeword/docs/RKNN_PORTING_GUIDE.md` — 실제 포팅 시행착오 히스토리
3. `zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md` — **현재 RKNN이 느린 원인 분석 + 개선 방향**

## 폴더 구조

```
rknn-stt/
├── zipformer/              # Streaming Transducer — 1차 완료, RKNN 성능 개선 필요
│   ├── rk3588/             # 변환/추론/벤치 코드
│   └── rk3676/             # 미래
└── wav2vec2/               # Split INT8-KL, CER 11.50%
```

## 현재 상태

### zipformer — RKNN 최적화 완료 (2026-03-17)

**RKNN KL divergence (정확도 최적):**
- CER: **21.85%** (KL divergence 100-sample 캘리브레이션, normal 22.97% 대비 -1.12pp)
- 모델: `encoder-epoch-99-avg-1-int8-cumfix-kl-100s.rknn`

**RKNN rmreshape + C API (속도 최적):**
- Encoder: **33ms/chunk** (ONNX 35ms보다 빠름!)
- CER: 26.25%, 모델: `encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn`

**ONNX INT8 (정확도 최적):**
- CER: 19.95%, RTF: 0.130 (ONNX INT8 4-thread)
- 추론: `rk3588/inference_onnx.py` (use_int8=True)

### 최적화 이력

| 시도 | Encoder/chunk | 비고 |
|------|--------------|------|
| rknnlite inputs_set | 52.7ms | baseline |
| C API set_io_mem + CACHEABLE | 39.2ms | -13.5ms |
| **remove_reshape=True** | **30.7ms (rknn_run)** | **-8.5ms** |
| + cache 변환 오버헤드 | **33ms (총합)** | **ONNX 35ms < 이김!** |

**핵심 기법:** `rknn.config(remove_reshape=True)` + C API `set_io_mem`
- 경계 Reshape 제거 → NPU dispatch 오버헤드 -8.5ms
- cache NCHW→NHWC: `out.reshape(N,C,H,W).transpose(0,2,3,1)` (~3ms)

### 관련 파일

| 파일 | 내용 |
|------|------|
| `rk3588/RESULTS.md` | 전체 성능 결과 |
| `rk3588/encoder_capi.py` | C API encoder wrapper (rmreshape용) |
| `rk3588/convert_encoder_int8_optarget.py` | rmreshape 변환 스크립트 |
| `rk3588/fix_cumsum.py` | CumSum → MatMul 패치 |
| `rk3588/inference_onnx.py` | ONNX INT8 추론 |
| `rk3588/inference_rknn.py` | RKNN Pure 추론 (rknnlite) |

## Claude 행동 규칙

### 사실 확인 없이 답변 금지

파일명, 모델 이름, 설정값, 수치, 경로 등 구체적 사실에 대해서는 반드시 파일을 읽거나 검색해서 확인한 후 답변할 것. 추측으로 답변 금지. 존재하지 않는 파일명을 지어내지 말 것.

### 커밋에 Co-Authored-By 넣지 말 것

git commit 메시지에 `Co-Authored-By: Claude` 줄을 절대 넣지 말 것.

## 공통 환경

```bash
conda run -n RKNN-Toolkit2 python rk3588/<script>.py
# 개행 금지 (퍼미션 다이얼로그 트리거), && 또는 ; 사용
```
