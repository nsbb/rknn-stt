"""
nocache RKNN 모델 벤치마크 (C API 사용).
encoder_capi.py의 EncoderCAPI를 사용하되, nocache 모델 경로 지정.
출력 수와 캐시 처리 방식이 다름.
"""
import numpy as np, time, sys, ctypes
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}

class RknnInputOutputNum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]

class RknnTensorAttr(Structure):
    _fields_ = [
        ('index', c_uint32), ('n_dims', c_uint32),
        ('dims', c_uint32 * RKNN_MAX_DIMS),
        ('name', ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_elems', c_uint32), ('size', c_uint32),
        ('fmt', c_int), ('type', c_int), ('qnt_type', c_int),
        ('fl', c_int8), ('zp', c_int32), ('scale', c_float),
        ('w_stride', c_uint32), ('size_with_stride', c_uint32),
        ('pass_through', c_uint8), ('h_stride', c_uint32),
    ]

class RknnTensorMem(Structure):
    _fields_ = [
        ('virt_addr', c_void_p), ('phys_addr', c_uint64),
        ('fd', c_int32), ('offset', c_int32),
        ('size', c_uint32), ('flags', c_uint32), ('priv_data', c_void_p),
    ]

lib.rknn_init.restype = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
lib.rknn_destroy.restype = c_int
lib.rknn_destroy.argtypes = [c_uint64]
lib.rknn_query.restype = c_int
lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
lib.rknn_set_core_mask.restype = c_int
lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]
lib.rknn_run.restype = c_int
lib.rknn_run.argtypes = [c_uint64, c_void_p]
lib.rknn_create_mem2.restype = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]
lib.rknn_set_io_mem.restype = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]
lib.rknn_mem_sync.restype = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(RknnTensorMem), c_int]

SYNC_TO_DEVICE = 0x1
SYNC_FROM_DEVICE = 0x2

NOCACHE_PATH = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-int8-cumfix-nocache.rknn'
NOCACHE_SIM_PATH = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-int8-cumfix-nocache-sim.rknn'
NOCACHE_STATIC_PATH = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-int8-cumfix-nocache-static.rknn'
BASELINE_PATH = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'


def bench_rknn_run(model_path, label, n_runs=100, warmup=10):
    """Benchmark rknn_run time using C API."""
    with open(model_path, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(buf), 0, None)
    assert ret == 0, f"init failed: {ret}"
    lib.rknn_set_core_mask(ctx, 1)

    io = RknnInputOutputNum()
    lib.rknn_query(ctx, 0, byref(io), ctypes.sizeof(io))
    n_in, n_out = io.n_input, io.n_output
    print(f"\n{label}: {n_in} inputs, {n_out} outputs")

    # Setup I/O
    in_mems, in_attrs = [], []
    for i in range(n_in):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, 1, byref(attr), ctypes.sizeof(attr))
        nt = attr.type
        if nt == 8:
            attr.type = 8; db = 8
        else:
            attr.type = 0; db = 4
        attr.pass_through = 0
        attr.fmt = 1 if attr.n_dims == 4 else 0
        nb = NATIVE_BYTES.get(nt, 2)
        nsz = attr.size_with_stride if attr.size_with_stride else attr.size
        mem = lib.rknn_create_mem2(ctx, max(nsz * db // nb, 64), 0)
        in_mems.append(mem)
        in_attrs.append(attr)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    out_mems, out_attrs = [], []
    for i in range(n_out):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, 2, byref(attr), ctypes.sizeof(attr))
        nt = attr.type
        if nt == 8:
            attr.type = 8; db = 8
        else:
            attr.type = 0; db = 4
        attr.pass_through = 0
        mem = lib.rknn_create_mem2(ctx, max(attr.n_elems * db, 64), 0)
        out_mems.append(mem)
        out_attrs.append(attr)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    # Zero-fill inputs and sync
    for i in range(n_in):
        sz = in_attrs[i].size_with_stride if in_attrs[i].size_with_stride else in_attrs[i].size
        ctypes.memset(in_mems[i].contents.virt_addr, 0, min(sz, 1024*1024))
        lib.rknn_mem_sync(ctx, in_mems[i], SYNC_TO_DEVICE)

    # Warmup
    for _ in range(warmup):
        lib.rknn_run(ctx, None)

    # Benchmark: rknn_run only
    run_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t1 = time.perf_counter()
        run_times.append((t1 - t0) * 1000)

    # Benchmark: full cycle (write + run + sync + read)
    full_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        # Write inputs
        for i in range(n_in):
            lib.rknn_mem_sync(ctx, in_mems[i], SYNC_TO_DEVICE)
        # Run
        lib.rknn_run(ctx, None)
        # Read outputs
        for i in range(n_out):
            lib.rknn_mem_sync(ctx, out_mems[i], SYNC_FROM_DEVICE)
        t1 = time.perf_counter()
        full_times.append((t1 - t0) * 1000)

    # Cleanup
    for m in in_mems + out_mems:
        lib.rknn_destroy_mem(ctx, m)
    lib.rknn_destroy(ctx)

    return {
        'run_median': np.median(run_times),
        'run_min': np.min(run_times),
        'full_median': np.median(full_times),
        'full_min': np.min(full_times),
    }


def main():
    print("=== nocache vs baseline RKNN Benchmark ===")

    results = {}
    for path, label in [
        (BASELINE_PATH, "Baseline (rmreshape)"),
        (NOCACHE_PATH, "nocache (no Concat/Slice)"),
        (NOCACHE_SIM_PATH, "nocache-sim (onnxsim)"),
        (NOCACHE_STATIC_PATH, "nocache-static (shapes folded)"),
    ]:
        results[label] = bench_rknn_run(path, label, n_runs=100, warmup=15)

    print("\n" + "=" * 60)
    print(f"{'Model':35s} {'run_med':>10s} {'run_min':>10s} {'full_med':>10s} {'full_min':>10s}")
    print("-" * 75)
    for label, r in results.items():
        print(f"{label:35s} {r['run_median']:10.2f} {r['run_min']:10.2f} {r['full_median']:10.2f} {r['full_min']:10.2f}")

    base = results["Baseline (rmreshape)"]
    nc = results["nocache (no Concat/Slice)"]
    diff = base['run_median'] - nc['run_median']
    print(f"\nDifference (rknn_run): {diff:+.2f}ms")
    print(f"Per-layer estimate: baseline ~1619 layers, nocache ~{1619 - 150} layers")
    if diff > 0:
        per_layer = diff / 150 * 1000
        print(f"  Estimated per-layer savings: {per_layer:.0f}µs")


if __name__ == '__main__':
    main()
