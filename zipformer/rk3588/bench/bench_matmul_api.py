"""
RKNN MatMul API 벤치마크 — per-call dispatch overhead 측정.
C++ demo 참고하여 정확한 struct layout 사용.
"""
import ctypes, numpy as np, time
from ctypes import (c_int, c_int8, c_int16, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_char_p, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

# ── rknn_matmul_type enum ──
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4

# ── Structures (matching C header exactly) ──
class RknnTensorMem(Structure):
    _fields_ = [
        ('virt_addr', c_void_p), ('phys_addr', c_uint64),
        ('fd', c_int32), ('offset', c_int32),
        ('size', c_uint32), ('flags', c_uint32), ('priv_data', c_void_p),
    ]

class RknnMatmulInfo(Structure):
    """rknn_matmul_info — must match C struct exactly including reserved bytes"""
    _fields_ = [
        ('M', c_int32),
        ('K', c_int32),
        ('N', c_int32),
        ('type', c_int32),              # rknn_matmul_type enum
        ('B_layout', c_int16),
        ('B_quant_type', c_int16),
        ('AC_layout', c_int16),
        ('AC_quant_type', c_int16),
        ('iommu_domain_id', c_int32),
        ('group_size', c_int16),
        ('reserved', c_int8 * 34),
    ]

class RknnMatmulTensorAttr(Structure):
    """rknn_matmul_tensor_attr"""
    _fields_ = [
        ('name', ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_dims', c_uint32),
        ('dims', c_uint32 * RKNN_MAX_DIMS),
        ('size', c_uint32),
        ('type', c_int32),              # rknn_tensor_type
    ]

class RknnMatmulIOAttr(Structure):
    """rknn_matmul_io_attr — contains A, B, C tensor attrs"""
    _fields_ = [
        ('A', RknnMatmulTensorAttr),
        ('B', RknnMatmulTensorAttr),
        ('C', RknnMatmulTensorAttr),
    ]

# ── Function signatures ──
lib.rknn_matmul_create.restype = c_int
lib.rknn_matmul_create.argtypes = [POINTER(c_uint64), POINTER(RknnMatmulInfo), POINTER(RknnMatmulIOAttr)]
lib.rknn_matmul_destroy.restype = c_int
lib.rknn_matmul_destroy.argtypes = [c_uint64]
lib.rknn_matmul_run.restype = c_int
lib.rknn_matmul_run.argtypes = [c_uint64]
lib.rknn_matmul_set_io_mem.restype = c_int
lib.rknn_matmul_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnMatmulTensorAttr)]
lib.rknn_matmul_set_core_mask.restype = c_int
lib.rknn_matmul_set_core_mask.argtypes = [c_uint64, c_int]

# rknn_create_mem (not rknn_create_mem2)
lib.rknn_create_mem.restype = POINTER(RknnTensorMem)
lib.rknn_create_mem.argtypes = [c_uint64, c_uint32]
lib.rknn_destroy_mem.restype = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]


def bench_matmul(M, K, N, matmul_type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
                 warmup=20, iterations=200):
    """Run MatMul API benchmark."""
    # Zero-initialize info struct (like memset in C++)
    info = RknnMatmulInfo()
    ctypes.memset(byref(info), 0, ctypes.sizeof(info))
    info.M = M
    info.K = K
    info.N = N
    info.type = matmul_type
    info.B_layout = 0      # normal layout
    info.AC_layout = 0     # normal layout

    io_attr = RknnMatmulIOAttr()
    ctypes.memset(byref(io_attr), 0, ctypes.sizeof(io_attr))
    ctx = c_uint64(0)

    ret = lib.rknn_matmul_create(byref(ctx), byref(info), byref(io_attr))
    if ret != 0:
        print(f"  rknn_matmul_create FAILED: {ret} (M={M}, K={K}, N={N})")
        return None

    lib.rknn_matmul_set_core_mask(ctx, 1)  # Core 0

    # Create memory using io_attr sizes
    mem_A = lib.rknn_create_mem(ctx, io_attr.A.size)
    mem_B = lib.rknn_create_mem(ctx, io_attr.B.size)
    mem_C = lib.rknn_create_mem(ctx, io_attr.C.size)

    if not mem_A or not mem_B or not mem_C:
        print(f"  rknn_create_mem FAILED")
        lib.rknn_matmul_destroy(ctx)
        return None

    # Set IO memory with proper tensor attrs
    ret_a = lib.rknn_matmul_set_io_mem(ctx, mem_A, byref(io_attr.A))
    ret_b = lib.rknn_matmul_set_io_mem(ctx, mem_B, byref(io_attr.B))
    ret_c = lib.rknn_matmul_set_io_mem(ctx, mem_C, byref(io_attr.C))

    if ret_a != 0 or ret_b != 0 or ret_c != 0:
        print(f"  set_io_mem FAILED: A={ret_a}, B={ret_b}, C={ret_c}")
        lib.rknn_matmul_destroy(ctx)
        return None

    # Fill A and B with random fp16 data
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    ctypes.memmove(mem_A.contents.virt_addr, A.ctypes.data, A.nbytes)
    ctypes.memmove(mem_B.contents.virt_addr, B.ctypes.data, B.nbytes)

    # Warmup
    for _ in range(warmup):
        ret = lib.rknn_matmul_run(ctx)
        if ret != 0:
            print(f"  rknn_matmul_run FAILED during warmup: {ret}")
            lib.rknn_destroy_mem(ctx, mem_A)
            lib.rknn_destroy_mem(ctx, mem_B)
            lib.rknn_destroy_mem(ctx, mem_C)
            lib.rknn_matmul_destroy(ctx)
            return None

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        lib.rknn_matmul_run(ctx)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # µs

    # Cleanup
    lib.rknn_destroy_mem(ctx, mem_A)
    lib.rknn_destroy_mem(ctx, mem_B)
    lib.rknn_destroy_mem(ctx, mem_C)
    lib.rknn_matmul_destroy(ctx)

    return times


def bench_sequential(count, M, K, N, warmup=10, iterations=50):
    """Benchmark N sequential matmul_run calls (simulate encoder)."""
    info = RknnMatmulInfo()
    ctypes.memset(byref(info), 0, ctypes.sizeof(info))
    info.M = M; info.K = K; info.N = N
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32

    io_attr = RknnMatmulIOAttr()
    ctypes.memset(byref(io_attr), 0, ctypes.sizeof(io_attr))
    ctx = c_uint64(0)

    ret = lib.rknn_matmul_create(byref(ctx), byref(info), byref(io_attr))
    if ret != 0:
        print(f"  Sequential create FAILED: {ret}")
        return None

    lib.rknn_matmul_set_core_mask(ctx, 1)
    mem_A = lib.rknn_create_mem(ctx, io_attr.A.size)
    mem_B = lib.rknn_create_mem(ctx, io_attr.B.size)
    mem_C = lib.rknn_create_mem(ctx, io_attr.C.size)
    lib.rknn_matmul_set_io_mem(ctx, mem_A, byref(io_attr.A))
    lib.rknn_matmul_set_io_mem(ctx, mem_B, byref(io_attr.B))
    lib.rknn_matmul_set_io_mem(ctx, mem_C, byref(io_attr.C))

    # Warmup
    for _ in range(warmup):
        lib.rknn_matmul_run(ctx)

    # Benchmark
    times = []
    run_fn = lib.rknn_matmul_run
    for _ in range(iterations):
        t0 = time.perf_counter()
        for _ in range(count):
            run_fn(ctx)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    lib.rknn_destroy_mem(ctx, mem_A)
    lib.rknn_destroy_mem(ctx, mem_B)
    lib.rknn_destroy_mem(ctx, mem_C)
    lib.rknn_matmul_destroy(ctx)
    return times


def main():
    print("RKNN MatMul API Benchmark")
    print("=" * 70)

    # Verify struct sizes
    print(f"RknnMatmulInfo size: {ctypes.sizeof(RknnMatmulInfo)} bytes")
    print(f"RknnMatmulIOAttr size: {ctypes.sizeof(RknnMatmulIOAttr)} bytes")
    print(f"RknnMatmulTensorAttr size: {ctypes.sizeof(RknnMatmulTensorAttr)} bytes")

    # Test cases matching zipformer encoder MatMul sizes
    # RK3588 alignment: K aligned 32B, N aligned 16B (fp16)
    test_cases = [
        (8, 384, 192,  "Attn Q/K proj"),
        (8, 384, 96,   "Attn V proj"),
        (8, 192, 384,  "Attn out proj"),
        (8, 384, 1536, "FFN expand"),
        (8, 1536, 384, "FFN contract"),
        (8, 384, 384,  "General 384x384"),
        (1, 384, 192,  "1-frame Q/K"),
        (1, 384, 384,  "1-frame general"),
        (64, 32, 32,   "Attn scores big"),
        (32, 32, 32,   "Attn scores small"),
    ]

    print(f"\n{'Description':25s} {'M':>4s} {'K':>5s} {'N':>5s} {'median µs':>10s} {'min µs':>8s} {'p95 µs':>8s}")
    print("-" * 70)

    for M, K, N, desc in test_cases:
        times = bench_matmul(M, K, N, warmup=30, iterations=300)
        if times is not None:
            med = np.median(times)
            mn = np.min(times)
            p95 = np.percentile(times, 95)
            print(f"{desc:25s} {M:4d} {K:5d} {N:5d} {med:10.1f} {mn:8.1f} {p95:8.1f}")

    # Sequential calls simulation
    print("\n\n=== Sequential MatMul calls (simulating encoder) ===")
    for count, desc in [(75, "75 calls (exMatMul)"), (168, "168 calls (all MatMul+Conv)"), (257, "257 calls (ONNX MatMul)")]:
        times = bench_sequential(count, 8, 384, 192, warmup=10, iterations=50)
        if times is not None:
            med = np.median(times)
            mn = np.min(times)
            per_call = med / count * 1000
            print(f"  {desc}: median={med:.2f}ms, min={mn:.2f}ms ({per_call:.0f}µs/call)")

    # Compare: graph dispatch baseline
    print(f"\n--- 비교 ---")
    print(f"  Graph dispatch: 1619 layers × 19µs = ~31ms")
    print(f"  (위 MatMul API 결과와 비교)")


if __name__ == '__main__':
    main()
