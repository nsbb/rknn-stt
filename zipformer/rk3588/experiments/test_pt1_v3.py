"""
Step 2 v3: pass_through=1 정밀 테스트
- 4D 텐서 NHWC format 사용 (test_cacheable_mem.py와 동일)
- size_with_stride 기반 크기 계산
- 3가지 모드: A) all pt=0, B) compatible pt=1, C) 비교
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
RKNN_QUERY_NATIVE_INPUT_ATTR = 8
RKNN_QUERY_NATIVE_OUTPUT_ATTR = 9
RKNN_NPU_CORE_0 = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEM_CACHEABLE = 0
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

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

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix.rknn'

IN_NAMES = [
    'x','cached_len_0','cached_len_1','cached_len_2','cached_len_3','cached_len_4',
    'cached_avg_0','cached_avg_1','cached_avg_2','cached_avg_3','cached_avg_4',
    'cached_key_0','cached_key_1','cached_key_2','cached_key_3','cached_key_4',
    'cached_val_0','cached_val_1','cached_val_2','cached_val_3','cached_val_4',
    'cached_val2_0','cached_val2_1','cached_val2_2','cached_val2_3','cached_val2_4',
    'cached_conv1_0','cached_conv1_1','cached_conv1_2','cached_conv1_3','cached_conv1_4',
    'cached_conv2_0','cached_conv2_1','cached_conv2_2','cached_conv2_3','cached_conv2_4',
]

# Compatible indices: cached_key(11-15), cached_val(16-20), cached_val2(21-25)
# These have NC1HWC2 in both input and output with matching sizes
COMPAT_INDICES = set(range(11, 26))


def load_model():
    with open(MODEL, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    assert lib.rknn_init(byref(ctx), md, len(md), 0, None) == 0
    assert lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0) == 0
    io = RknnInputOutputNum()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
    return ctx, io.n_input, io.n_output, md


def calc_pt0_size(attr):
    """Calculate buffer size for pt=0 (same logic as test_cacheable_mem.py)."""
    native_t = attr.type
    native_b = NATIVE_BYTES.get(native_t, 2)
    if native_t == RKNN_TENSOR_INT64:
        desired_b = 8
    else:
        desired_b = 4  # float32
    native_sz = attr.size_with_stride if attr.size_with_stride else attr.size
    return native_sz * desired_b // native_b


def bench_mode(ctx, n_in, n_out, pt1_set, label):
    """Benchmark with given pt1_set."""
    print(f"\n--- {label} ---")
    in_mems = []
    in_attrs_list = []

    for i in range(n_in):
        attr = RknnTensorAttr()
        attr.index = i

        if i in pt1_set:
            # pass_through=1: native format
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.pass_through = 1
            sz = attr.size_with_stride if attr.size_with_stride else attr.size
        else:
            # pass_through=0: runtime converts (same as test_cacheable_mem.py)
            lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            native_t = attr.type
            native_b = NATIVE_BYTES.get(native_t, 2)
            if native_t == RKNN_TENSOR_INT64:
                attr.type = RKNN_TENSOR_INT64
                desired_b = 8
            else:
                attr.type = RKNN_TENSOR_FLOAT32
                desired_b = 4
            attr.pass_through = 0
            # NHWC for 4D, NCHW for others
            attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
            native_sz = attr.size_with_stride if attr.size_with_stride else attr.size
            sz = native_sz * desired_b // native_b

        in_attrs_list.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 fail input {i}"
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem input {i} ({IN_NAMES[i]}): ret={ret}, pt={attr.pass_through}")
            for m in in_mems:
                lib.rknn_destroy_mem(ctx, m)
            return None

    out_mems = []
    out_attrs_list = []

    for i in range(n_out):
        attr = RknnTensorAttr()
        attr.index = i

        if i == 0:
            # encoder_out: always pt=0 float32
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            native_t = attr.type
            native_b = NATIVE_BYTES.get(native_t, 2)
            attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            sz = attr.n_elems * 4
        elif i in pt1_set:
            # Compatible cache output: pt=1 native
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.pass_through = 1
            sz = attr.size_with_stride if attr.size_with_stride else attr.size
        else:
            # Incompatible cache output: pt=0 float32/int64
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            native_t = attr.type
            native_b = NATIVE_BYTES.get(native_t, 2)
            if native_t == RKNN_TENSOR_INT64:
                attr.type = RKNN_TENSOR_INT64
                desired_b = 8
            else:
                attr.type = RKNN_TENSOR_FLOAT32
                desired_b = 4
            attr.pass_through = 0
            sz = attr.n_elems * desired_b

        out_attrs_list.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 fail output {i}"
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem output {i}: ret={ret}")
            for m in in_mems + out_mems:
                lib.rknn_destroy_mem(ctx, m)
            return None

    # Initialize all to zero
    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Verify: run once, check output
    ret = lib.rknn_run(ctx, None)
    if ret != 0:
        print(f"  rknn_run FAILED: ret={ret}")
        for m in in_mems + out_mems:
            lib.rknn_destroy_mem(ctx, m)
        return None

    lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
    enc = np.empty(out_attrs_list[0].n_elems, dtype=np.float32)
    ctypes.memmove(enc.ctypes.data, out_mems[0].contents.virt_addr, enc.nbytes)
    print(f"  Verify: enc_out[:5]={enc[:5]}, range=[{enc.min():.4f},{enc.max():.4f}], nz={np.count_nonzero(enc)}/{len(enc)}")

    # Copy cache out → in
    for j in range(1, n_out):
        lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
        copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
        ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
        lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup 2 more
    for _ in range(2):
        ret = lib.rknn_run(ctx, None)
        assert ret == 0, f"warmup failed: {ret}"
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Benchmark
    N = 20
    times_write = []
    times_run = []
    times_copy = []
    times_read = []
    times_total = []

    for _ in range(N):
        t0 = time.perf_counter()

        # Write x (zeros)
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)

        # For pt=0 cache: need to sync (already in buffer from previous copy)
        for j in range(1, n_in):
            if j not in pt1_set:
                lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t1 = time.perf_counter()

        ret = lib.rknn_run(ctx, None)
        assert ret == 0
        t2 = time.perf_counter()

        # Copy cache out → in
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            if j in pt1_set:
                lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t3 = time.perf_counter()

        # Read encoder_out
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        t4 = time.perf_counter()

        times_write.append((t1-t0)*1000)
        times_run.append((t2-t1)*1000)
        times_copy.append((t3-t2)*1000)
        times_read.append((t4-t3)*1000)
        times_total.append((t4-t0)*1000)

    print(f"  Results (median of {N}):")
    print(f"    write+sync:  {np.median(times_write):.2f}ms")
    print(f"    run:         {np.median(times_run):.2f}ms")
    print(f"    copy cache:  {np.median(times_copy):.2f}ms")
    print(f"    read enc:    {np.median(times_read):.2f}ms")
    print(f"    TOTAL:       {np.median(times_total):.2f}ms")

    for m in in_mems + out_mems:
        lib.rknn_destroy_mem(ctx, m)

    return {
        'run': np.median(times_run),
        'copy': np.median(times_copy),
        'total': np.median(times_total),
    }


# ─── Main ────────────────────────────────────────────────────────
print("=" * 60)
print("pass_through=1 v3 (NHWC fix)")
print("=" * 60)

# Mode A: ALL pt=0 (baseline)
ctx, n_in, n_out, md = load_model()
result_a = bench_mode(ctx, n_in, n_out, set(), "Mode A: ALL pt=0 (baseline)")
lib.rknn_destroy(ctx)

# Mode B: Compatible pt=1 (cached_key/val/val2)
ctx, n_in, n_out, md = load_model()
result_b = bench_mode(ctx, n_in, n_out, COMPAT_INDICES, "Mode B: Selective pt=1 (key/val/val2)")
lib.rknn_destroy(ctx)

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
if result_a:
    print(f"  A) ALL pt=0:       run={result_a['run']:.1f}ms  copy={result_a['copy']:.1f}ms  total={result_a['total']:.1f}ms")
else:
    print(f"  A) ALL pt=0:       FAILED")
if result_b:
    print(f"  B) Selective pt=1: run={result_b['run']:.1f}ms  copy={result_b['copy']:.1f}ms  total={result_b['total']:.1f}ms")
else:
    print(f"  B) Selective pt=1: FAILED")
print(f"  ref) test_cacheable_mem pt=0: 39.6ms")
print(f"  ref) inputs_set:              45.0ms")
print(f"  ref) ONNX INT8:               35ms")
