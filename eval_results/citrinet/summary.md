# CitriNet FP16 평가 결과

## 모델 정보

| 항목 | 내용 |
|------|------|
| **모델** | CitriNet CTC (NeMo, Jasper-based CNN) |
| **RKNN 파일** | `citrinet_fp16.rknn` |
| **원본** | NeMo CitriNet 한국어 CTC (SentencePiece BPE 2048 tokens) |
| **양자화** | FP16 (양자화 없음) |
| **ONNX 수정** | LogSoftmax 제거, SE→ReduceMean, ReduceMean→Conv, Squeeze 제거 |
| **크기** | 281MB |

## 테스트셋별 결과

| Testset | Samples | CER | Avg ms | RTF |
|---------|--------:|----:|-------:|----:|
| 7F_KSK | 108 | 27.5% | 64 | 0.021 |
| 7F_HJY | 107 | 54.9% | 64 | 0.021 |
| modelhouse_2m | 51 | 37.5% | 63 | 0.021 |
| modelhouse_2m_noheater | 51 | 25.6% | 60 | 0.020 |
| modelhouse_3m | 51 | 51.4% | 61 | 0.020 |
| **TOTAL** | **368** | **39.9%** | **63** | **0.021** |

> RTF (Real-Time Factor) = 처리시간 / 오디오길이. 고정 입력 3초 기준. RTF < 1이면 실시간보다 빠름. 0.021 = **48배 실시간**.
