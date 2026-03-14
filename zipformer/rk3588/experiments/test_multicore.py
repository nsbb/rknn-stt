"""
멀티코어 NPU 테스트
현재: CORE_0 단독 → 37ms
테스트: CORE_0_1_2 (3코어), CORE_0_1 (2코어), CORE_AUTO
"""
import ctypes, numpy as np, time
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEM_CACHEABLE = 0
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}

# Core masks
CORE_0 = 1
CORE_0_1_2 = 7
CORE_0_1 = 3
CORE_AUTO = 0

CORES = [
    ("CORE_0", CORE_0),
    ("CORE_0_1_2", CORE_0_1_2),
    ("CORE_0_1", CORE_0_1),
    ("CORE_AUTO", CORE_AUTO),
]

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

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix.rknn'


def bench_core(core_name, core_mask):
    with open(MODEL, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(md), 0, None)
    if ret != 0:
        print(f"  {core_name}: init failed ret={ret}")
        return None

    ret = lib.rknn_set_core_mask(ctx, core_mask)
    if ret != 0:
        print(f"  {core_name}: set_core_mask({core_mask}) failed ret={ret}")
        lib.rknn_destroy(ctx)
        return None

    io = RknnInputOutputNum()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
    n_in, n_out = io.n_input, io.n_output

    in_mems = []
    in_attrs = []
    for i in range(n_in):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        native_b = NATIVE_BYTES.get(native_t, 2)
        if native_t == RKNN_TENSOR_INT64:
            attr.type = RKNN_TENSOR_INT64; desired_b = 8
        else:
            attr.type = RKNN_TENSOR_FLOAT32; desired_b = 4
        attr.pass_through = 0
        attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
        native_sz = attr.size_with_stride if attr.size_with_stride else attr.size
        sz = native_sz * desired_b // native_b
        in_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  {core_name}: set_io_mem input {i} failed ret={ret}")
            lib.rknn_destroy(ctx)
            return None

    out_mems = []
    out_attrs = []
    for i in range(n_out):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        native_b = NATIVE_BYTES.get(native_t, 2)
        if native_t == RKNN_TENSOR_INT64:
            attr.type = RKNN_TENSOR_INT64; desired_b = 8
        else:
            attr.type = RKNN_TENSOR_FLOAT32; desired_b = 4
        attr.pass_through = 0
        sz = attr.n_elems * desired_b
        out_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  {core_name}: set_io_mem output {i} failed ret={ret}")
            lib.rknn_destroy(ctx)
            return None

    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup
    for _ in range(3):
        ret = lib.rknn_run(ctx, None)
        if ret != 0:
            print(f"  {core_name}: rknn_run failed ret={ret}")
            lib.rknn_destroy(ctx)
            return None
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    N = 20
    times_run = []
    times_total = []

    for _ in range(N):
        t0 = time.perf_counter()
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)

        t1 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t2 = time.perf_counter()

        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        t3 = time.perf_counter()

        times_run.append((t2-t1)*1000)
        times_total.append((t3-t0)*1000)

    for m in in_mems + out_mems:
        lib.rknn_destroy_mem(ctx, m)
    lib.rknn_destroy(ctx)

    return np.median(times_run), np.median(times_total)


print("=" * 60)
print("Multi-core NPU Benchmark")
print("=" * 60)

results = {}
for name, mask in CORES:
    print(f"\n  Testing {name} (mask={mask})...")
    r = bench_core(name, mask)
    if r:
        print(f"  {name}: run={r[0]:.1f}ms  total={r[1]:.1f}ms")
        results[name] = r
    else:
        print(f"  {name}: FAILED")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for name, (run, total) in results.items():
    print(f"  {name:>15}: run={run:.1f}ms  total={total:.1f}ms")
print(f"  {'ONNX INT8':>15}: total=35ms")
