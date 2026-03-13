# Introduction

This model is converted from
https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12

See
https://github.com/k2-fsa/icefall/pull/1651
for how it is trained.

Note it uses zipformer v1.

## Encoder 다운로드 (GitHub 100MB 제한 초과)

encoder ONNX는 용량 초과로 이 레포에 포함되지 않는다. 아래에서 직접 다운로드:

```bash
# HuggingFace에서 원본 모델 다운로드
# (sherpa-onnx 래핑 버전 — encoder/decoder/joiner 포함)
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.onnx -P zipformer/
wget https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/encoder-epoch-99-avg-1.int8.onnx -P zipformer/
```

파일 크기: fp32 279MB, int8 122MB


## 추론 코드: zipformer_onnx_test.py
- SherpaOnnxInference객체 init함수 안에 onnx 경로 지정
    - fp32: avg-1.onnx
    - int8: avg-1.int8.onnx
    - 객체 생성 후 함수 호출 시 전달
- test_csv: csv파일 입력 추론 -> 추론 텍스트와 cer 추론시간 새로운 column으로 추가해서 csv 저장함
    - FileName: 파일 경로
    - gt: 정답 텍스트
- test_dir: dir안 .wav or .mp3 파일 일괄 추론 -> 추론 결과 텍스트 .txt로 저장
- test_file: 단일 파일 추론 후 결과 텍스트 출력

## test_wavs: 테스트 음성 파일 샘플
- trans.txt에 정답 텍스트 저장되어있음