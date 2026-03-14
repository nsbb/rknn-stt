"""
Benchmark rmreshape model variants using C API.
Measures rknn_run time, cache conversion, and total per-chunk time.
"""
import numpy as np, time, os, sys, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

MODELS = [
    ('rmreshape (baseline)', f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'),
    ('rmreshape-opt2',       f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-opt2.rknn'),
    ('rmreshape-singlecore', f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-singlecore.rknn'),
    ('rmreshape-flash',      f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-flash.rknn'),
    ('rmreshape-sim',        f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-sim.rknn'),
]

N_WARMUP = 5
N_ITER = 20

def bench_model(name, path):
    if not os.path.exists(path):
        print(f"  {name}: SKIP (file not found)")
        return None

    enc = EncoderCAPI(path, core_mask=1)
    cache = enc.init_cache()
    x = np.random.randn(1, 39, 80, 1).astype(np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _, cache = enc.run(x, cache)

    # Benchmark
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _, cache = enc.run(x, cache)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    enc.release()
    avg = np.mean(times)
    std = np.std(times)
    mn = np.min(times)
    print(f"  {name}: avg={avg:.1f}ms  std={std:.1f}ms  min={mn:.1f}ms")
    return avg

if __name__ == '__main__':
    print(f"Benchmarking {len(MODELS)} variants (warmup={N_WARMUP}, iter={N_ITER})")
    print()
    results = {}
    for name, path in MODELS:
        r = bench_model(name, path)
        if r is not None:
            results[name] = r

    print()
    print("=" * 60)
    print("Summary (sorted by avg time):")
    for name, avg in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {avg:.1f}ms  {name}")
