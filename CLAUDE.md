# CLAUDE.md — rknn-stt

RK 시리즈 NPU에서 한국어 STT 모델을 실행하기 위한 포팅 작업.

## 새 세션 시작 시 필독

1. `../docs/RK3588_NPU_AI_GUIDE.md` — RKNN 범용 버그 목록, 디버깅 방법
2. `../rknn-wakeword/docs/RKNN_PORTING_GUIDE.md` — 실제 포팅 시행착오 히스토리
3. `zipformer/rk3588/WHY_ONNX_BEATS_RKNN.md` — **현재 RKNN이 느린 원인 분석 + 개선 방향**

## 폴더 구조

```
rknn-stt/
├── zipformer/              # Streaming Transducer — 완료
│   ├── rk3588/             # 변환/추론/벤치 코드
│   └── rk3676/             # 미래
├── wav2vec2/               # Split INT8-KL — 완료 (최고 정확도)
├── ko_citrinet/            # CitriNet FP16 — 완료 (최고 속도)
├── eval_results/           # 로컬 테스트셋 평가 결과
├── eval_local_testsets.py  # 평가 스크립트
├── EVALUATION.md           # 3개 모델 비교 문서
└── testset/                # 로컬 테스트셋 (gitignore)
```

## 현재 상태 — 3개 모델 완료 (2026-03-19)

### 로컬 테스트셋 평가 (368샘플: 7F_KSK, 7F_HJY, modelhouse 2m/3m)

| 모델 | CER | 지연 | RTF | 모델 크기 | 최대 입력 |
|------|:---:|:---:|:---:|:---:|:---:|
| **wav2vec2 Split INT8-KL** | **9.0%** | 437ms | 0.087 | 462MB | 5초 |
| zipformer INT8 nocache | 22.97% | 27.5ms | 0.10 | 83MB | 스트리밍 |
| citrinet FP16 | 39.9% | **63ms** | **0.021** | 281MB | 3초 |

> 파이프라인에는 **wav2vec2** 채택 (정확도 최고).

### wav2vec2 — Split INT8-KL (2026-03-15)

- **CER 9.0%** (368 로컬 샘플), 11.78% (702 스마트홈 셋)
- 437ms/5초, RTF 0.087 (11.5배 실시간)
- Split INT8: Encoder Layer 0-11 INT8-KL + Layer 12-23 FP16
- amplitude normalization target=5.0 필수
- 상세: `wav2vec2/README.md`

### zipformer — Streaming Transducer (2026-03-17)

- **CER 21.51%** (KL-100s), 22.97% (standard INT8)
- **27.5ms/chunk** (320ms 음성, RTF 0.10, 10배 실시간)
- CumSum 버그 → MatMul 패치로 해결 (CER 91.89% → 22.97%)
- 상세: `zipformer/README.md`, `zipformer/rk3588/RESULTS.md`

### ko_citrinet — CitriNet FP16 (2026-03-17)

- **63ms/3초** (RTF 0.021, 48배 실시간)
- ONNX↔RKNN cosine 0.999935
- RKNN 버그 4개 수정: LogSoftmax, SE→ReduceMean, ReduceMean→Conv, Squeeze 제거
- 상세: `ko_citrinet/README.md`

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
