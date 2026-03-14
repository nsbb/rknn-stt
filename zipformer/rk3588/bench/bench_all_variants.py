"""Benchmark all RKNN model variants."""
import sys, time, os, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI
import numpy as np

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

# Find all model variants
models = {}
# Baseline
bl = f'{BASE}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'
if os.path.exists(bl):
    models['baseline-rmreshape'] = bl

# All variants
for f in sorted(glob.glob(f'{BASE}/encoder-int8-cumfix-*.rknn')):
    name = os.path.basename(f).replace('encoder-int8-cumfix-', '').replace('.rknn', '')
    models[name] = f

for name, path in models.items():
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"\n=== {name} ({sz:.1f}MB) ===")
    try:
        enc = EncoderCAPI(path, core_mask=1)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        continue

    cache = enc.init_cache()
    x = np.zeros(enc._in_shapes[0], dtype=np.float32)

    # Warmup
    ok = True
    for _ in range(3):
        try:
            _, cache = enc.run(x, cache)
        except Exception as e:
            print(f"  RUNTIME ERROR: {e}")
            ok = False
            break

    if not ok:
        enc.release()
        continue

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
