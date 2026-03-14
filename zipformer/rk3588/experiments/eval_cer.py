"""
CER (Character Error Rate) 평가
- ONNX / RKNN-Hybrid / RKNN-Pure 비교
- 4개 test_wavs 기준 (대형 데이터셋 없을 때)
"""
import numpy as np, os, sys, time, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'


def cer(ref, hyp):
    """편집 거리 기반 CER 계산 (공백 제거)"""
    ref = ref.replace(' ', '').replace('.', '').replace('?', '').replace('!', '')
    hyp = hyp.replace(' ', '').replace('.', '').replace('?', '').replace('!', '')
    if not ref: return 0.0
    r, h = list(ref), list(hyp)
    R, H = len(r), len(h)
    d = np.zeros((R+1, H+1), dtype=int)
    for i in range(R+1): d[i, 0] = i
    for j in range(H+1): d[0, j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            d[i, j] = min(
                d[i-1, j] + 1,
                d[i, j-1] + 1,
                d[i-1, j-1] + (0 if r[i-1]==h[j-1] else 1)
            )
    return d[R, H] / R


def load_gt():
    gt = {}
    with open(f'{BASE}/test_wavs/trans.txt') as f:
        for line in f:
            nm, *rest = line.strip().split(' ', 1)
            gt[nm] = rest[0] if rest else ''
    return gt


def run_eval(model_name, transcribe_fn):
    gt = load_gt()
    wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
    results = []
    total_time = 0
    for wav in wavs:
        fname = os.path.basename(wav)
        t0 = time.time()
        stats = transcribe_fn(wav)
        elapsed = time.time() - t0
        text = stats['text']
        ref  = gt.get(fname, '')
        c = cer(ref, text)
        results.append({'file': fname, 'ref': ref, 'hyp': text, 'cer': c, 'time': elapsed})
        total_time += elapsed

    avg_cer = np.mean([r['cer'] for r in results])
    avg_time = np.mean([r['time'] for r in results])

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    for r in results:
        print(f"[{r['file']}] CER={r['cer']*100:.1f}%  ({r['time']*1000:.0f}ms)")
        print(f"  REF: {r['ref']}")
        print(f"  HYP: {r['hyp']}")
    print(f"\n  AVG CER:  {avg_cer*100:.2f}%")
    print(f"  AVG Time: {avg_time*1000:.0f}ms/file")
    return results, avg_cer


if __name__ == '__main__':
    print("Zipformer CER 평가")
    print()

    # 1. ONNX FP32 (기준)
    from inference_onnx import ZipformerONNX
    onnx_fp32 = ZipformerONNX(use_int8=False)
    fp32_results, fp32_cer = run_eval('ONNX FP32 (4-thread)', onnx_fp32.transcribe)

    # 2. ONNX INT8 (권장)
    onnx_int8 = ZipformerONNX(use_int8=True)
    int8_results, int8_cer = run_eval('ONNX INT8 (4-thread)', onnx_int8.transcribe)

    # 3. RKNN Hybrid (RKNN encoder_out + ONNX cache)
    from inference_hybrid import ZipformerHybrid
    hybrid_model = ZipformerHybrid()
    hybrid_results, hybrid_cer = run_eval('RKNN Hybrid (RKNN encoder + ONNX cache)', hybrid_model.transcribe)
    hybrid_model.release()

    # 4. RKNN Pure (전체 RKNN, 캐시 이슈 있음)
    from inference_rknn import ZipformerRKNN
    pure_model = ZipformerRKNN()
    pure_results, pure_cer = run_eval('RKNN Pure (전체 RKNN)', pure_model.transcribe)
    pure_model.release()

    print(f"\n{'='*60}")
    print("최종 비교:")
    print(f"  ONNX FP32    CER: {fp32_cer*100:.2f}%")
    print(f"  ONNX INT8    CER: {int8_cer*100:.2f}%  (FP32 대비 delta: {(int8_cer-fp32_cer)*100:+.2f}%)")
    print(f"  RKNN Hybrid  CER: {hybrid_cer*100:.2f}%  (FP32 대비 delta: {(hybrid_cer-fp32_cer)*100:+.2f}%)")
    print(f"  RKNN Pure    CER: {pure_cer*100:.2f}%   (FP32 대비 delta: {(pure_cer-fp32_cer)*100:+.2f}%)")
