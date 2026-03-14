"""Benchmark selective op_target models vs baseline."""
import sys, time
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI
import numpy as np

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

models = {
    'baseline-rmreshape': f'{BASE}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn',
    'opt10cpu': f'{BASE}/encoder-int8-cumfix-rmreshape-opt10cpu.rknn',
    'opt50cpu': f'{BASE}/encoder-int8-cumfix-rmreshape-opt50cpu.rknn',
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
