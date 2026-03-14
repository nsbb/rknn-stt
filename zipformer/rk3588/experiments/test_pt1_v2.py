"""
Step 2 v2: pass_through=1 정밀 테스트
- 모든 return value 확인
- encoder_out 실제 값 출력하여 정상 동작 검증
- 3가지 모드 비교: A) all pt=0, B) compatible pt=1 only, C) all pt=1
"""
import ctypes, numpy as np, time, sys
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

FMT_NAMES = {0:'NCHW', 1:'NHWC', 2:'NC1HWC2', 3:'UNDEF'}
TYPE_NAMES = {0:'FP32', 1:'FP16', 2:'INT8', 3:'UINT8', 8:'INT64'}

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

IN_DTYPES = [
    'float32','int64','int64','int64','int64','int64',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
]

# Compatible cache indices (NC1HWC2 both in/out, same size)
# cached_key: 11-15, cached_val: 16-20, cached_val2: 21-25
COMPAT_INDICES = set(range(11, 26))


def load_model():
    with open(MODEL, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(md), 0, None)
    assert ret == 0, f"rknn_init failed: {ret}"
    ret = lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)
    assert ret == 0, f"set_core_mask failed: {ret}"
    io = RknnInputOutputNum()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
    return ctx, io.n_input, io.n_output, md


def query_all_attrs(ctx, n_in, n_out):
    in_norm = []
    in_nat = []
    out_norm = []
    out_nat = []
    for i in range(n_in):
        a = RknnTensorAttr(); a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(a), ctypes.sizeof(a))
        in_norm.append(a)
        b = RknnTensorAttr(); b.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(b), ctypes.sizeof(b))
        in_nat.append(b)
    for i in range(n_out):
        a = RknnTensorAttr(); a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(a), ctypes.sizeof(a))
        out_norm.append(a)
        b = RknnTensorAttr(); b.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(b), ctypes.sizeof(b))
        out_nat.append(b)
    return in_norm, in_nat, out_norm, out_nat


def bench_mode(ctx, n_in, n_out, in_norm, in_nat, out_norm, out_nat, pt1_set, label):
    """
    Benchmark with given pt1_set (indices that use pass_through=1).
    """
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
            na = in_nat[i]
            sz = na.size_with_stride if na.size_with_stride else na.size
        else:
            # pass_through=0: runtime converts
            lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            if IN_DTYPES[i] == 'int64':
                attr.type = RKNN_TENSOR_INT64
                sz = attr.n_elems * 8
            else:
                attr.type = RKNN_TENSOR_FLOAT32
                sz = attr.n_elems * 4
            attr.fmt = RKNN_TENSOR_NCHW
            attr.pass_through = 0

        in_attrs_list.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        if not mem:
            print(f"  FAIL create_mem2 input {i} ({IN_NAMES[i]})")
            return None
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem input {i} ({IN_NAMES[i]}): ret={ret}, pt={attr.pass_through}, sz={sz}")
            # Cleanup
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
            attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            sz = attr.n_elems * 4
        elif i in pt1_set:
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.pass_through = 1
            na = out_nat[i]
            sz = na.size_with_stride if na.size_with_stride else na.size
        else:
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            if i <= 5:  # cached_len outputs
                attr.type = RKNN_TENSOR_INT64
                sz = attr.n_elems * 8
            else:
                attr.type = RKNN_TENSOR_FLOAT32
                sz = attr.n_elems * 4
            attr.pass_through = 0

        out_attrs_list.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        if not mem:
            print(f"  FAIL create_mem2 output {i}")
            for m in in_mems + out_mems:
                lib.rknn_destroy_mem(ctx, m)
            return None
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem output {i}: ret={ret}")
            for m in in_mems + out_mems:
                lib.rknn_destroy_mem(ctx, m)
            return None

    # Initialize inputs to zero
    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Run once, check return value and output
    ret = lib.rknn_run(ctx, None)
    if ret != 0:
        print(f"  rknn_run FAILED: ret={ret}")
        for m in in_mems + out_mems:
            lib.rknn_destroy_mem(ctx, m)
        return None

    # Read encoder_out to verify
    lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
    enc_size = out_attrs_list[0].n_elems
    enc = np.empty(enc_size, dtype=np.float32)
    ctypes.memmove(enc.ctypes.data, out_mems[0].contents.virt_addr, enc.nbytes)
    print(f"  Verify: encoder_out[:5] = {enc[:5]}")
    print(f"  Verify: encoder_out range = [{enc.min():.4f}, {enc.max():.4f}], nonzero={np.count_nonzero(enc)}/{len(enc)}")

    if np.all(enc == 0):
        print(f"  WARNING: all zeros output!")

    # Copy cache out → in for warmup state
    for j in range(1, n_out):
        lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
        copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
        ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
        lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup 2 more runs
    for _ in range(2):
        ret = lib.rknn_run(ctx, None)
        assert ret == 0, f"warmup rknn_run failed: {ret}"
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Benchmark
    N = 20
    times_run = []
    times_copy = []
    times_total = []

    for _ in range(N):
        t0 = time.perf_counter()

        # Write x (zeros, for consistent benchmark)
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)

        t1 = time.perf_counter()
        ret = lib.rknn_run(ctx, None)
        assert ret == 0
        t2 = time.perf_counter()

        # Copy cache out → in
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t3 = time.perf_counter()

        # Read encoder_out
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        t4 = time.perf_counter()

        times_run.append((t2 - t1) * 1000)
        times_copy.append((t3 - t2) * 1000)
        times_total.append((t4 - t0) * 1000)

    med_run = np.median(times_run)
    med_copy = np.median(times_copy)
    med_total = np.median(times_total)

    print(f"  Results (N={N}):")
    print(f"    run:        {med_run:.2f}ms")
    print(f"    copy cache: {med_copy:.2f}ms")
    print(f"    total:      {med_total:.2f}ms")

    # Cleanup
    for m in in_mems + out_mems:
        lib.rknn_destroy_mem(ctx, m)

    return med_run, med_copy, med_total


# ─── Main ────────────────────────────────────────────────────────
print("=" * 60)
print("pass_through=1 v2 (return value checking)")
print("=" * 60)

# Mode A: ALL pt=0
ctx, n_in, n_out, md = load_model()
in_norm, in_nat, out_norm, out_nat = query_all_attrs(ctx, n_in, n_out)
result_a = bench_mode(ctx, n_in, n_out, in_norm, in_nat, out_norm, out_nat, set(), "Mode A: ALL pt=0")
lib.rknn_destroy(ctx)

# Mode B: Compatible only pt=1 (cached_key/val/val2)
ctx, n_in, n_out, md = load_model()
in_norm, in_nat, out_norm, out_nat = query_all_attrs(ctx, n_in, n_out)
result_b = bench_mode(ctx, n_in, n_out, in_norm, in_nat, out_norm, out_nat, COMPAT_INDICES, "Mode B: Compatible pt=1 (15 tensors)")
lib.rknn_destroy(ctx)

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
if result_a:
    print(f"  A) ALL pt=0:        run={result_a[0]:.1f}ms  copy={result_a[1]:.1f}ms  total={result_a[2]:.1f}ms")
if result_b:
    print(f"  B) Selective pt=1:  run={result_b[0]:.1f}ms  copy={result_b[1]:.1f}ms  total={result_b[2]:.1f}ms")
print(f"  Previous test_cacheable_mem:  39.6ms")
print(f"  ONNX INT8:                    35ms")
