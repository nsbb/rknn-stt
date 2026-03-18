# wav2vec2 Split INT8-KL 평가 결과

## 모델 정보

| 항목 | 내용 |
|------|------|
| **모델** | wav2vec2 xls-r-300m (Large, 300M params) |
| **RKNN 파일** | `wav2vec2_part1_features_fp16.rknn` + `wav2vec2_part2a_int8_kl.rknn` + `wav2vec2_part2b_fp16.rknn` + `wav2vec2_part3_lmhead_fp16.rknn` |
| **원본** | facebook/wav2vec2-xls-r-300m + 한국어 fine-tuned CTC head |
| **양자화** | Split INT8 (Layer 0-11: INT8-KL, Layer 12-23: FP16) |
| **구성** | Part1(CNN FP16) → Part2A(Encoder L0-11, INT8-KL) → Part2B(Encoder L12-23, FP16) → Part3(LM Head, FP16) |
| **합계 크기** | 462MB (13+160+295+5.2) |

## 테스트셋별 결과

| Testset | Samples | CER | Avg ms |
|---------|--------:|----:|-------:|
| 7F_KSK | 108 | **4.7%** | 437 |
| 7F_HJY | 107 | 9.9% | 438 |
| modelhouse_2m | 51 | 9.6% | 435 |
| modelhouse_2m_noheater | 51 | 6.2% | 436 |
| modelhouse_3m | 51 | 18.5% | 434 |
| **TOTAL** | **368** | **9.0%** | **437** |
