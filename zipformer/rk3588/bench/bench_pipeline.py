"""
Pipeline parallelism: overlap NPU rknn_run with CPU cache conversion.

Strategy:
- Thread 1 (NPU): write inputs → rknn_run → signal done
- Thread 2 (CPU): wait for signal → read outputs → convert cache
- While CPU converts cache for chunk N, NPU starts chunk N+1

Expected savings: cache_convert time (~1.1ms) is fully overlapped with rknn_run.
Additionally, output reading (~0.8ms) can be partially overlapped.

Total expected: rknn_run + write_input ≈ 35 + 0.87 ≈ 36ms (vs 38ms sequential)
"""
import numpy as np, time, ctypes, sys, threading
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import EncoderCAPI, lib, out_nchw_to_in_nhwc

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'
N_WARMUP = 10
N_ITER = 50


def bench_sequential():
    """Baseline: sequential execution."""
    enc = EncoderCAPI(MODEL, core_mask=1)
    cache = enc.init_cache()
    x = np.random.randn(1, 39, 80, 1).astype(np.float32)

    for _ in range(N_WARMUP):
        _, cache = enc.run(x, cache)

    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _, cache = enc.run(x, cache)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    enc.release()
    return np.mean(times), np.min(times)


def bench_pipelined():
    """Pipelined: overlap cache conversion with next rknn_run."""
    enc = EncoderCAPI(MODEL, core_mask=1)
    cache = enc.init_cache()
    x = np.random.randn(1, 39, 80, 1).astype(np.float32)

    for _ in range(N_WARMUP):
        _, cache = enc.run(x, cache)

    # Pre-allocate for double buffering
    times = []

    for _ in range(N_ITER):
        t0 = time.perf_counter()

        # Step 1: Write inputs (must be before rknn_run)
        enc._write_input(0, x)
        for i, nm in enumerate(enc._cache_names):
            enc._write_input(i + 1, cache[nm])

        # Step 2: rknn_run
        lib.rknn_run(enc._ctx, None)

        # Step 3: Read all outputs into raw buffers (fast memcpy)
        enc_nchw = enc._read_output(0)
        raw_outs = []
        for i in range(len(enc._cache_names)):
            raw_outs.append(enc._read_output(i + 1))

        # Step 4: Convert cache on CPU (this is what we want to overlap)
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

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    enc.release()
    return np.mean(times), np.min(times)


def bench_optimized_cache():
    """Optimized cache conversion: avoid unnecessary copies."""
    enc = EncoderCAPI(MODEL, core_mask=1)
    cache = enc.init_cache()
    x = np.random.randn(1, 39, 80, 1).astype(np.float32)

    for _ in range(N_WARMUP):
        _, cache = enc.run(x, cache)

    # Pre-allocate cache buffers
    cache_buffers = {}
    for i, nm in enumerate(enc._cache_names):
        shape = enc._in_shapes[i + 1]
        dtype = enc._in_dtypes[i + 1]
        cache_buffers[nm] = np.zeros(shape, dtype=dtype)

    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()

        # Write inputs
        enc._write_input(0, x)
        for i, nm in enumerate(enc._cache_names):
            enc._write_input(i + 1, cache[nm])

        # Run
        lib.rknn_run(enc._ctx, None)

        # Read and convert in-place where possible
        enc_nchw = enc._read_output(0)
        enc_out = out_nchw_to_in_nhwc(enc_nchw, (1, 8, 512, 1)).reshape(1, 8, 512)

        for i, nm in enumerate(enc._cache_names):
            out_arr = enc._read_output(i + 1)
            in_shape = enc._in_shapes[i + 1]
            N, H, W, C = in_shape
            # In-place conversion into pre-allocated buffer
            np.transpose(out_arr.reshape(N, C, H, W), (0, 2, 3, 1), out=cache_buffers[nm] if cache_buffers[nm].shape == (N,H,W,C) else None)
            if cache_buffers[nm].shape == (N,H,W,C):
                cache[nm] = cache_buffers[nm]
            else:
                cache[nm] = out_nchw_to_in_nhwc(out_arr, in_shape)
            if 'cached_len' in nm:
                cache[nm] = cache[nm].astype(np.int64)

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    enc.release()
    return np.mean(times), np.min(times)


def bench_skip_cache_convert():
    """Test: what if we skip cache conversion entirely (wrong results, just for timing)."""
    enc = EncoderCAPI(MODEL, core_mask=1)
    cache = enc.init_cache()
    x = np.random.randn(1, 39, 80, 1).astype(np.float32)

    for _ in range(N_WARMUP):
        _, cache = enc.run(x, cache)

    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()

        enc._write_input(0, x)
        for i, nm in enumerate(enc._cache_names):
            enc._write_input(i + 1, cache[nm])

        lib.rknn_run(enc._ctx, None)

        # Only read encoder_out, skip cache
        enc_nchw = enc._read_output(0)

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    enc.release()
    return np.mean(times), np.min(times)


if __name__ == '__main__':
    print("=== Pipeline parallelism benchmark ===\n")

    avg, mn = bench_sequential()
    print(f"Sequential (baseline):    avg={avg:.1f}ms  min={mn:.1f}ms")

    avg2, mn2 = bench_pipelined()
    print(f"Pipelined (same logic):   avg={avg2:.1f}ms  min={mn2:.1f}ms")

    avg3, mn3 = bench_skip_cache_convert()
    print(f"Skip cache (timing only): avg={avg3:.1f}ms  min={mn3:.1f}ms")
    print(f"  → cache overhead = {avg - avg3:.1f}ms")
