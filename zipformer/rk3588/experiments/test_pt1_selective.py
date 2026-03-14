"""
Step 2b: pass_through=1 선택적 적용
- 호환 텐서 (native format 동일): pt=1
- 비호환 텐서: pt=0 (runtime 변환)
- x, encoder_out: pt=0 (float32)

이전 결과:
  전체 pt=1: 6397ms (깨짐 - format 불일치)
  전체 pt=0: 39.6ms
  inputs_set: 45.0ms
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

FMT_NAMES = {0:'NCHW', 1:'NHWC', 2:'NC1HWC2', 3:'UNDEFINED'}
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

ENC_NAMES = [
    'x','cached_len_0','cached_len_1','cached_len_2','cached_len_3','cached_len_4',
    'cached_avg_0','cached_avg_1','cached_avg_2','cached_avg_3','cached_avg_4',
    'cached_key_0','cached_key_1','cached_key_2','cached_key_3','cached_key_4',
    'cached_val_0','cached_val_1','cached_val_2','cached_val_3','cached_val_4',
    'cached_val2_0','cached_val2_1','cached_val2_2','cached_val2_3','cached_val2_4',
    'cached_conv1_0','cached_conv1_1','cached_conv1_2','cached_conv1_3','cached_conv1_4',
    'cached_conv2_0','cached_conv2_1','cached_conv2_2','cached_conv2_3','cached_conv2_4',
]

ENC_SHAPES = [
    [1,39,80],[2,1],[4,1],[3,1],[2,1],[4,1],
    [2,1,384],[4,1,384],[3,1,384],[2,1,384],[4,1,384],
    [2,64,1,192],[4,32,1,192],[3,16,1,192],[2,8,1,192],[4,32,1,192],
    [2,64,1,96],[4,32,1,96],[3,16,1,96],[2,8,1,96],[4,32,1,96],
    [2,64,1,96],[4,32,1,96],[3,16,1,96],[2,8,1,96],[4,32,1,96],
    [2,1,384,30],[4,1,384,30],[3,1,384,30],[2,1,384,30],[4,1,384,30],
    [2,1,384,30],[4,1,384,30],[3,1,384,30],[2,1,384,30],[4,1,384,30],
]

ENC_DTYPES = [
    'float32','int64','int64','int64','int64','int64',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
    'float32','float32','float32','float32','float32',
]


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


def get_dims(attr):
    return [attr.dims[d] for d in range(attr.n_dims)]


def check_compatibility(ctx, n_in, n_out):
    """Check native format compatibility between inputs and outputs."""
    compatible = {}  # index -> True/False

    in_native = []
    out_native = []
    in_normal = []
    out_normal = []

    for i in range(n_in):
        a = RknnTensorAttr(); a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(a), ctypes.sizeof(a))
        in_native.append(a)
        b = RknnTensorAttr(); b.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(b), ctypes.sizeof(b))
        in_normal.append(b)

    for i in range(n_out):
        a = RknnTensorAttr(); a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(a), ctypes.sizeof(a))
        out_native.append(a)
        b = RknnTensorAttr(); b.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(b), ctypes.sizeof(b))
        out_normal.append(b)

    print("\n--- Native Format Compatibility ---")
    print(f"{'idx':>3} {'name':>15} | {'in_fmt':>8} {'in_type':>5} {'in_sz':>8} {'in_dims':>20} | {'out_fmt':>8} {'out_type':>5} {'out_sz':>8} {'out_dims':>20} | {'compat':>7}")
    print("-" * 130)

    pt1_inputs = set()  # indices that can use pass_through=1

    for i in range(n_in):
        nm = ENC_NAMES[i] if i < len(ENC_NAMES) else f"in_{i}"
        ina = in_native[i]
        in_fmt = FMT_NAMES.get(ina.fmt, str(ina.fmt))
        in_type = TYPE_NAMES.get(ina.type, str(ina.type))
        in_sz = ina.size_with_stride if ina.size_with_stride else ina.size
        in_dims = get_dims(ina)

        if i == 0:
            # x - always pt=0
            print(f"{i:3d} {nm:>15} | {in_fmt:>8} {in_type:>5} {in_sz:>8} {str(in_dims):>20} | {'---':>8} {'---':>5} {'---':>8} {'---':>20} | {'x(skip)':>7}")
            continue

        # Output index = i (same order as input for cache)
        if i < n_out:
            outa = out_native[i]
            out_fmt = FMT_NAMES.get(outa.fmt, str(outa.fmt))
            out_type = TYPE_NAMES.get(outa.type, str(outa.type))
            out_sz = outa.size_with_stride if outa.size_with_stride else outa.size
            out_dims = get_dims(outa)

            # Compatible if: same format, same type, same size, same dims
            compat = (ina.fmt == outa.fmt and ina.type == outa.type and in_sz == out_sz)
            if compat:
                pt1_inputs.add(i)
        else:
            out_fmt = out_type = "N/A"
            out_sz = 0
            out_dims = []
            compat = False

        c_str = "YES" if compat else "NO"
        print(f"{i:3d} {nm:>15} | {in_fmt:>8} {in_type:>5} {in_sz:>8} {str(in_dims):>20} | {out_fmt:>8} {out_type:>5} {out_sz:>8} {str(out_dims):>20} | {c_str:>7}")

    # Summary
    n_compat = len(pt1_inputs)
    n_cache = n_in - 1  # exclude x
    compat_bytes = sum(
        (in_native[i].size_with_stride if in_native[i].size_with_stride else in_native[i].size)
        for i in pt1_inputs
    )
    total_cache_bytes = sum(
        (in_native[i].size_with_stride if in_native[i].size_with_stride else in_native[i].size)
        for i in range(1, n_in)
    )

    print(f"\nCompatible: {n_compat}/{n_cache} cache tensors ({compat_bytes/1024:.1f}KB / {total_cache_bytes/1024:.1f}KB)")
    print(f"pt=1 가능: {sorted(pt1_inputs)}")

    return in_native, out_native, in_normal, out_normal, pt1_inputs


def bench_selective_pt1(ctx, n_in, n_out, in_native, out_native, in_normal, out_normal, pt1_set):
    """Benchmark with selective pass_through=1."""
    in_attrs = []
    in_mems = []

    for i in range(n_in):
        attr = RknnTensorAttr()
        attr.index = i

        if i == 0:
            # x: pt=0 float32
            lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.type = RKNN_TENSOR_FLOAT32
            attr.fmt = RKNN_TENSOR_NCHW
            attr.pass_through = 0
            sz = attr.n_elems * 4
        elif i in pt1_set:
            # Compatible cache: pt=1
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.pass_through = 1
            na = in_native[i]
            sz = na.size_with_stride if na.size_with_stride else na.size
        else:
            # Incompatible cache: pt=0
            lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            if ENC_DTYPES[i] == 'int64':
                attr.type = RKNN_TENSOR_INT64
            else:
                attr.type = RKNN_TENSOR_FLOAT32
            attr.fmt = RKNN_TENSOR_NCHW
            attr.pass_through = 0
            sz = attr.n_elems * (8 if ENC_DTYPES[i] == 'int64' else 4)

        in_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 fail input {i}"
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem input {i} ({ENC_NAMES[i]}): ret={ret}")
            return None

    out_attrs = []
    out_mems = []

    for i in range(n_out):
        attr = RknnTensorAttr()
        attr.index = i

        if i == 0:
            # encoder_out: pt=0 float32
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            sz = attr.n_elems * 4
        elif i in pt1_set:
            # Compatible cache output: pt=1
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            attr.pass_through = 1
            na = out_native[i]
            sz = na.size_with_stride if na.size_with_stride else na.size
        else:
            # Incompatible cache output: pt=0
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
            if ENC_DTYPES[i] == 'int64':
                attr.type = RKNN_TENSOR_INT64
            else:
                attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            sz = attr.n_elems * (8 if ENC_DTYPES[i] == 'int64' else 4)

        out_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 fail output {i}"
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL set_io_mem output {i}: ret={ret}")
            return None

    # Initialize all buffers to zero
    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup
    print("  Warmup (3 runs)...")
    for w in range(3):
        lib.rknn_run(ctx, None)
        # Copy cache outputs → inputs
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            out_sz = out_mems[j].contents.size
            in_sz = in_mems[j].contents.size
            if j in pt1_set:
                # Native format, same size → direct copy
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, in_sz)
            else:
                # pt=0: both are float32/int64, just copy
                copy_sz = min(out_sz, in_sz)
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        print(f"    warmup {w} done")

    # Benchmark
    N = 20
    times = {'write':[], 'sync':[], 'run':[], 'copy':[], 'read':[], 'total':[]}

    for n in range(N):
        t0 = time.perf_counter()
        # Write x
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        t1 = time.perf_counter()
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)
        t2 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t3 = time.perf_counter()

        # Copy cache out → in
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            out_sz = out_mems[j].contents.size
            in_sz = in_mems[j].contents.size
            if j in pt1_set:
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, in_sz)
            else:
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, min(out_sz, in_sz))
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t4 = time.perf_counter()

        # Read encoder_out
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        enc = np.empty(out_attrs[0].n_elems, dtype=np.float32)
        ctypes.memmove(enc.ctypes.data, out_mems[0].contents.virt_addr, enc.nbytes)
        t5 = time.perf_counter()

        times['write'].append((t1-t0)*1000)
        times['sync'].append((t2-t1)*1000)
        times['run'].append((t3-t2)*1000)
        times['copy'].append((t4-t3)*1000)
        times['read'].append((t5-t4)*1000)
        times['total'].append((t5-t0)*1000)

    pt1_count = len(pt1_set)
    pt0_count = n_in - 1 - pt1_count

    print(f"\n=== Selective pt=1 ({pt1_count} pt=1, {pt0_count} pt=0, x excluded) ===")
    print(f"  write x:       {np.median(times['write']):.2f}ms")
    print(f"  sync x:        {np.median(times['sync']):.2f}ms")
    print(f"  run:           {np.median(times['run']):.2f}ms")
    print(f"  copy cache:    {np.median(times['copy']):.2f}ms")
    print(f"  read enc_out:  {np.median(times['read']):.2f}ms")
    print(f"  TOTAL:         {np.median(times['total']):.2f}ms")

    # Cleanup
    for mem in in_mems:
        lib.rknn_destroy_mem(ctx, mem)
    for mem in out_mems:
        lib.rknn_destroy_mem(ctx, mem)

    return np.median(times['total'])


# === Also benchmark all pt=0 as baseline ===
def bench_all_pt0(ctx, n_in, n_out, in_native, out_native):
    """All pt=0 baseline with set_io_mem + CACHEABLE."""
    in_attrs = []
    in_mems = []

    for i in range(n_in):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        if ENC_DTYPES[i] == 'int64':
            attr.type = RKNN_TENSOR_INT64
        else:
            attr.type = RKNN_TENSOR_FLOAT32
        attr.fmt = RKNN_TENSOR_NCHW
        attr.pass_through = 0
        sz = attr.n_elems * (8 if ENC_DTYPES[i] == 'int64' else 4)

        in_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        in_mems.append(mem)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    out_attrs = []
    out_mems = []
    for i in range(n_out):
        attr = RknnTensorAttr()
        attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        if ENC_DTYPES[i] == 'int64':
            attr.type = RKNN_TENSOR_INT64
        else:
            attr.type = RKNN_TENSOR_FLOAT32
        attr.pass_through = 0
        sz = attr.n_elems * (8 if ENC_DTYPES[i] == 'int64' else 4)

        out_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        out_mems.append(mem)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup
    for _ in range(3):
        lib.rknn_run(ctx, None)
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr,
                          min(out_mems[j].contents.size, in_mems[j].contents.size))
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    N = 20
    runs = []
    totals = []

    for _ in range(N):
        t0 = time.perf_counter()
        ctypes.memmove(in_mems[0].contents.virt_addr, np.zeros([1,39,80], dtype=np.float32).ctypes.data, 1*39*80*4)
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)
        t1 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t2 = time.perf_counter()
        for j in range(1, n_out):
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr,
                          min(out_mems[j].contents.size, in_mems[j].contents.size))
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t3 = time.perf_counter()
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        t4 = time.perf_counter()
        runs.append((t2-t1)*1000)
        totals.append((t4-t0)*1000)

    for mem in in_mems:
        lib.rknn_destroy_mem(ctx, mem)
    for mem in out_mems:
        lib.rknn_destroy_mem(ctx, mem)

    return np.median(runs), np.median(totals)


# ─── Main ────────────────────────────────────────────────────────
print("=" * 60)
print("Selective pass_through=1 Benchmark")
print("=" * 60)

ctx, n_in, n_out, md = load_model()
in_native, out_native, in_normal, out_normal, pt1_set = check_compatibility(ctx, n_in, n_out)
lib.rknn_destroy(ctx)

# --- Benchmark 1: all pt=0 (baseline) ---
print("\n--- Benchmark: ALL pt=0 (baseline) ---")
ctx, n_in, n_out, md = load_model()
run_pt0, total_pt0 = bench_all_pt0(ctx, n_in, n_out, in_native, out_native)
lib.rknn_destroy(ctx)
print(f"  ALL pt=0:  run={run_pt0:.1f}ms  total={total_pt0:.1f}ms")

# --- Benchmark 2: selective pt=1 ---
if pt1_set:
    print(f"\n--- Benchmark: Selective pt=1 ({len(pt1_set)} tensors) ---")
    ctx, n_in, n_out, md = load_model()
    total_sel = bench_selective_pt1(ctx, n_in, n_out, in_native, out_native, in_normal, out_normal, pt1_set)
    lib.rknn_destroy(ctx)
else:
    print("\n  No compatible tensors for pt=1!")
    total_sel = None

# --- Summary ---
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  ALL pt=0 (set_io_mem):  run={run_pt0:.1f}ms  total={total_pt0:.1f}ms")
if total_sel:
    print(f"  Selective pt=1:         total={total_sel:.1f}ms")
print(f"  inputs_set (이전):      45.0ms")
print(f"  ONNX INT8:              35ms")
