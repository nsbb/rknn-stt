"""
Optimize cache conversion: the 2.8ms overhead.
Current: for each of 35 cache tensors:
  read_output → np.frombuffer → reshape(NCHW) → transpose(0,2,3,1) → contiguous copy

Optimizations:
1. Pre-allocate output buffers (avoid allocation per iteration)
2. Use direct memory view without frombuffer copy
3. Batch small tensors together
4. Use memoryview/ctypes for direct NCHW→NHWC without numpy
"""
import numpy as np, time, ctypes, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI, lib, out_nchw_to_in_nhwc, SYNC_FROM_DEVICE

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'
N_WARMUP = 10
N_ITER = 50

enc = EncoderCAPI(MODEL, core_mask=1)
cache = enc.init_cache()
x = np.random.randn(1, 39, 80, 1).astype(np.float32)

# Warmup
for _ in range(N_WARMUP):
    _, cache = enc.run(x, cache)

# Pre-compute conversion info
cache_info = []
for i, nm in enumerate(enc._cache_names):
    in_shape = enc._in_shapes[i + 1]
    out_shape = enc._out_shapes[i + 1]
    dtype = enc._out_dtypes[i + 1]
    N, H, W, C = in_shape
    is_len = 'cached_len' in nm
    n_elems = 1
    for d in out_shape:
        n_elems *= d
    cache_info.append((nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, i + 1))

# Pre-allocate output arrays
pre_alloc_outs = {}
for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
    pre_alloc_outs[nm] = np.empty(in_shape, dtype=np.int64 if is_len else np.float32)


def original_cache_convert():
    """Original method: full read + transpose + copy."""
    new_cache = {}
    for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
        out_arr = enc._read_output(oidx)
        converted = out_nchw_to_in_nhwc(out_arr, in_shape)
        if is_len:
            converted = converted.astype(np.int64)
        new_cache[nm] = converted
    return new_cache


def optimized_cache_convert():
    """Optimized: sync all at once, read into pre-allocated buffers."""
    new_cache = {}

    # Sync all outputs from device first (batch sync)
    for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
        lib.rknn_mem_sync(enc._ctx, enc._out_mems[oidx], SYNC_FROM_DEVICE)

    # Then read and convert without individual syncs
    for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
        mem = enc._out_mems[oidx]
        buf = (ctypes.c_byte * (n_elems * np.dtype(dtype).itemsize)).from_address(mem.contents.virt_addr)
        arr = np.frombuffer(buf, dtype=dtype).reshape(out_shape)
        # Direct transpose into pre-allocated buffer
        result = np.ascontiguousarray(np.transpose(arr.reshape(N, C, H, W), (0, 2, 3, 1)))
        if is_len:
            new_cache[nm] = result.astype(np.int64)
        else:
            new_cache[nm] = result
    return new_cache


def minimal_cache_convert():
    """Minimal: only sync+read, skip transpose (test sync overhead vs convert overhead)."""
    for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
        lib.rknn_mem_sync(enc._ctx, enc._out_mems[oidx], SYNC_FROM_DEVICE)
    # Just read raw data, no conversion
    results = []
    for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
        mem = enc._out_mems[oidx]
        buf = (ctypes.c_byte * (n_elems * np.dtype(dtype).itemsize)).from_address(mem.contents.virt_addr)
        results.append(np.frombuffer(buf, dtype=dtype).reshape(out_shape).copy())
    return results


# Run inference first to populate output buffers
enc._write_input(0, x)
for i, nm in enumerate(enc._cache_names):
    enc._write_input(i + 1, cache[nm])
lib.rknn_run(enc._ctx, None)

# Benchmark cache conversion methods
print("=== Cache conversion benchmark ===\n")

# Original
times = []
for _ in range(N_ITER):
    t0 = time.perf_counter()
    _ = original_cache_convert()
    times.append((time.perf_counter() - t0) * 1000)
print(f"Original:  avg={np.mean(times):.2f}ms  min={np.min(times):.2f}ms")

# Optimized (batch sync)
times = []
for _ in range(N_ITER):
    t0 = time.perf_counter()
    _ = optimized_cache_convert()
    times.append((time.perf_counter() - t0) * 1000)
print(f"Batch sync: avg={np.mean(times):.2f}ms  min={np.min(times):.2f}ms")

# Minimal (sync+read only)
times = []
for _ in range(N_ITER):
    t0 = time.perf_counter()
    _ = minimal_cache_convert()
    times.append((time.perf_counter() - t0) * 1000)
print(f"Sync+read only: avg={np.mean(times):.2f}ms  min={np.min(times):.2f}ms")

# Pure transpose benchmark
print("\n=== Transpose overhead ===")
# Create test data matching cache sizes
test_arrays = []
for nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx in cache_info:
    test_arrays.append(np.random.randn(*out_shape).astype(np.float32))

times = []
for _ in range(N_ITER):
    t0 = time.perf_counter()
    for arr, (nm, in_shape, out_shape, dtype, N, H, W, C, is_len, n_elems, oidx) in zip(test_arrays, cache_info):
        np.ascontiguousarray(np.transpose(arr.reshape(N, C, H, W), (0, 2, 3, 1)))
    times.append((time.perf_counter() - t0) * 1000)
print(f"35x transpose: avg={np.mean(times):.2f}ms  min={np.min(times):.2f}ms")

enc.release()
