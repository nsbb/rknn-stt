"""
Detailed timing breakdown of rmreshape encoder.
Measures: write_input, rknn_run, read_output, cache_convert separately.
"""
import numpy as np, time, ctypes, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI, lib, out_nchw_to_in_nhwc

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'
N_WARMUP = 10
N_ITER = 50

enc = EncoderCAPI(MODEL, core_mask=1)
cache = enc.init_cache()
x = np.random.randn(1, 39, 80, 1).astype(np.float32)

# Warmup
for _ in range(N_WARMUP):
    _, cache = enc.run(x, cache)

# Detailed benchmark
t_write = []
t_run = []
t_read = []
t_convert = []
t_total = []

for _ in range(N_ITER):
    t0 = time.perf_counter()

    # Write inputs
    enc._write_input(0, x)
    for i, nm in enumerate(enc._cache_names):
        enc._write_input(i + 1, cache[nm])
    t1 = time.perf_counter()

    # Run
    lib.rknn_run(enc._ctx, None)
    t2 = time.perf_counter()

    # Read outputs
    enc_nchw = enc._read_output(0)
    raw_outs = []
    for i in range(len(enc._cache_names)):
        raw_outs.append(enc._read_output(i + 1))
    t3 = time.perf_counter()

    # Convert cache
    enc_out = out_nchw_to_in_nhwc(enc_nchw, (1, 8, 512, 1)).reshape(1, 8, 512)
    new_cache = {}
    for i, nm in enumerate(enc._cache_names):
        out_arr = raw_outs[i]
        in_shape = enc._in_shapes[i + 1]
        converted = out_nchw_to_in_nhwc(out_arr, in_shape)
        if 'cached_len' in nm:
            converted = converted.astype(np.int64)
        new_cache[nm] = converted
    cache = new_cache
    t4 = time.perf_counter()

    t_write.append((t1 - t0) * 1000)
    t_run.append((t2 - t1) * 1000)
    t_read.append((t3 - t2) * 1000)
    t_convert.append((t4 - t3) * 1000)
    t_total.append((t4 - t0) * 1000)

enc.release()

print(f"=== Detailed timing ({N_ITER} iterations) ===")
print(f"  write_input:  avg={np.mean(t_write):.2f}ms  min={np.min(t_write):.2f}ms")
print(f"  rknn_run:     avg={np.mean(t_run):.2f}ms  min={np.min(t_run):.2f}ms")
print(f"  read_output:  avg={np.mean(t_read):.2f}ms  min={np.min(t_read):.2f}ms")
print(f"  cache_convert:avg={np.mean(t_convert):.2f}ms  min={np.min(t_convert):.2f}ms")
print(f"  TOTAL:        avg={np.mean(t_total):.2f}ms  min={np.min(t_total):.2f}ms")
print(f"\n  rknn_run is {np.mean(t_run)/np.mean(t_total)*100:.1f}% of total")
print(f"  overhead is {(np.mean(t_total)-np.mean(t_run))/np.mean(t_total)*100:.1f}% of total")
