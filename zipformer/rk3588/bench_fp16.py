"""Benchmark FP16 rmreshape model vs INT8 rmreshape baseline."""
import sys, time
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI
import numpy as np

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

models = {
    'INT8-rmreshape': f'{BASE}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn',
    'FP16-rmreshape': f'{BASE}/encoder-epoch-99-avg-1-fp16-cumfix-rmreshape.rknn',
}

for name, path in models.items():
    print(f"\n=== {name} ===")
    try:
        enc = EncoderCAPI(path, core_mask=1)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        continue

    cache = enc.init_cache()
    x = np.zeros(enc._in_shapes[0], dtype=np.float32)

    # Warmup
    for _ in range(5):
        _, cache = enc.run(x, cache)

    # Benchmark
    N = 20
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _, cache = enc.run(x, cache)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  avg: {avg:.1f}ms  min: {mn:.1f}ms  max: {mx:.1f}ms")
    enc.release()

print("\nDone!")
