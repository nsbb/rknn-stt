# CLAUDE.md — rknn-stt

RK 시리즈 NPU에서 한국어 STT 모델을 실행하기 위한 포팅 작업.

## 새 세션 시작 시 필독

1. `../docs/RK3588_NPU_AI_GUIDE.md` — RKNN 범용 버그 목록, 디버깅 방법
2. `../rknn-wakeword/docs/RKNN_PORTING_GUIDE.md` — 실제 포팅 시행착오 히스토리 (BCResNet-t2 기준이지만 접근법 동일)

## 폴더 구조

```
rknn-stt/
├── zipformer/              # Streaming Transducer (다음 작업)
│   ├── rk3588/             # 보드별 변환/추론
│   └── rk3676/             # 미래
└── wav2vec2/               # 미착수
    ├── rk3588/
    └── rk3676/
```

## 현재 상태

### zipformer — 다음 작업

- 모델: 한국어 Streaming Transducer ASR (KsponSpeech, BPE 5000 vocab)
- ONNX 있음: `zipformer/encoder/decoder/joiner-epoch-99-avg-1.onnx`
- CPU 추론 동작 확인: `zipformer/zipformer_onnx_test.py` (sherpa-onnx)

**시작 방법:**
```bash
# 1. RKNN Model Zoo 파이프라인 파악
cat ../rknn_model_zoo/examples/zipformer/python/zipformer.py

# 2. rk3588 폴더 만들고 변환 시도
mkdir -p zipformer/rk3588
conda run -n RKNN-Toolkit2 python -c "
from rknn.api import RKNN; rknn = RKNN(verbose=True)
rknn.load_onnx('zipformer/encoder-epoch-99-avg-1.onnx')
rknn.build(do_quantization=False)
"

# 3. 상수 출력 확인 → 문제 있으면 ONNX 그래프 수정
```

**예상 이슈:**
- `ReduceMean` → depthwise Conv 교체 (../rknn-wakeword/fix_rknn_graph.py 참조)
- `CumSum`, `Where`, `ConstantOfShape` → NPU 미지원 가능, 조사 필요
- Dynamic shape → x: [1, 103, 80] 고정 (Model Zoo zipformer.py 참조)
- encoder 캐시 출력 NCHW→NHWC 변환 필요

### wav2vec2 — 미착수

## 공통 환경

```bash
conda run -n RKNN-Toolkit2 python <script>.py
# 명령에 개행문자 있으면 퍼미션 다이얼로그 → 한 줄로 작성
```
