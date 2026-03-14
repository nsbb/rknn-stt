"""
Advanced zero-copy optimizations:
A) Disable flush flags (rknn_run skips cache flush/invalidation)
B) SRAM allocation for small tensors
C) rknn_create_mem_from_fd for compatible cache tensors (true zero-copy)
D) All combined
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
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEM_CACHEABLE = 0
RKNN_FLAG_MEM_TRY_SRAM = 4  # RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

# Init flags
RKNN_FLAG_DISABLE_FLUSH_INPUT = 0x4000
RKNN_FLAG_DISABLE_FLUSH_OUTPUT = 0x8000
RKNN_FLAG_ENABLE_SRAM = 0x800
RKNN_FLAG_MODEL_ZERO_COPY = 0x10000

NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}
RKNN_NPU_CORE_0 = 1

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
lib.rknn_create_mem_from_fd.restype = POINTER(RknnTensorMem)
lib.rknn_create_mem_from_fd.argtypes = [c_uint64, c_int32, c_void_p, c_uint32, c_int32]

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

# Compatible: cached_key(11-15), cached_val(16-20), cached_val2(21-25)
COMPAT = set(range(11, 26))


def load_model_buf():
    with open(MODEL, 'rb') as f:
        return f.read()


def init_ctx(buf, extra_flags=0):
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(md), extra_flags, None)
    if ret != 0:
        print(f"  init failed ret={ret} flags=0x{extra_flags:x}")
        return None, None, None, md
    lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)
    io = RknnInputOutputNum()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
    return ctx, io.n_input, io.n_output, md


def setup_io_baseline(ctx, n_in, n_out, mem_flag=RKNN_FLAG_MEM_CACHEABLE):
    """Standard set_io_mem with pt=0, same as working test_cacheable_mem.py approach."""
    in_mems = []
    in_attrs = []
    for i in range(n_in):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        native_b = NATIVE_BYTES.get(native_t, 2)
        if native_t == 8:  # INT64
            attr.type = 8; desired_b = 8
        else:
            attr.type = 0; desired_b = 4  # FLOAT32
        attr.pass_through = 0
        attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
        native_sz = attr.size_with_stride if attr.size_with_stride else attr.size
        sz = native_sz * desired_b // native_b
        in_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), mem_flag)
        if not mem:
            print(f"  create_mem2 fail input {i}")
            return None, None, None, None
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  set_io_mem input {i} ({IN_NAMES[i]}) fail ret={ret}")
            return None, None, None, None

    out_mems = []
    out_attrs = []
    for i in range(n_out):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        if native_t == 8:
            attr.type = 8; desired_b = 8
        else:
            attr.type = 0; desired_b = 4
        attr.pass_through = 0
        sz = attr.n_elems * desired_b
        out_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), mem_flag)
        out_mems.append(mem)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    return in_mems, in_attrs, out_mems, out_attrs


def setup_io_with_mem_sharing(ctx, n_in, n_out):
    """
    Compatible cache tensors: output mem shared as input via rknn_create_mem_from_fd.
    Incompatible: normal separate allocation.
    All pt=0 for now (sharing the buffer, not the format).
    """
    # First allocate outputs
    out_mems = []
    out_attrs = []
    for i in range(n_out):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        if native_t == 8:
            attr.type = 8; desired_b = 8
        else:
            attr.type = 0; desired_b = 4
        attr.pass_through = 0
        sz = attr.n_elems * desired_b
        out_attrs.append(attr)
        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  set_io_mem output {i} fail ret={ret}")
            return None, None, None, None, set()

    # Now allocate inputs - for compatible tensors, try to share output buffer
    in_mems = []
    in_attrs = []
    shared_indices = set()

    for i in range(n_in):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(attr), ctypes.sizeof(attr))
        native_t = attr.type
        native_b = NATIVE_BYTES.get(native_t, 2)
        if native_t == 8:
            attr.type = 8; desired_b = 8
        else:
            attr.type = 0; desired_b = 4
        attr.pass_through = 0
        attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
        native_sz = attr.size_with_stride if attr.size_with_stride else attr.size
        in_sz = native_sz * desired_b // native_b
        in_attrs.append(attr)

        # Try to share output buffer for compatible cache tensors
        if i in COMPAT and i < n_out:
            out_mem = out_mems[i]
            out_sz = out_mem.contents.size
            if out_sz >= in_sz:
                # Share output memory as input
                mem = lib.rknn_create_mem_from_fd(
                    ctx,
                    out_mem.contents.fd,
                    out_mem.contents.virt_addr,
                    out_sz,
                    0  # offset
                )
                if mem:
                    ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
                    if ret == 0:
                        in_mems.append(mem)
                        shared_indices.add(i)
                        continue
                    else:
                        print(f"  shared set_io_mem input {i} fail ret={ret}, falling back")
                        lib.rknn_destroy_mem(ctx, mem)

        # Fallback: normal allocation
        mem = lib.rknn_create_mem2(ctx, max(in_sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  set_io_mem input {i} ({IN_NAMES[i]}) fail ret={ret}")
            return None, None, None, None, set()

    return in_mems, in_attrs, out_mems, out_attrs, shared_indices


def bench(ctx, n_in, n_out, in_mems, out_mems, shared_set=set(), do_own_sync=True, label=""):
    """Run benchmark. shared_set = indices where input shares output buffer (skip copy)."""
    # Init all to zero
    for i in range(n_in):
        ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
        if do_own_sync:
            lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup
    for _ in range(3):
        ret = lib.rknn_run(ctx, None)
        if ret != 0:
            print(f"  {label}: rknn_run fail ret={ret}")
            return None
        for j in range(1, n_out):
            if j in shared_set:
                # Shared buffer: output IS input, just sync
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
                    lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
            else:
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
                copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

    # Bench
    N = 20
    times_run = []
    times_copy = []
    times_total = []

    for _ in range(N):
        t0 = time.perf_counter()
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        if do_own_sync:
            lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)

        t1 = time.perf_counter()
        ret = lib.rknn_run(ctx, None)
        assert ret == 0, f"run fail {ret}"
        t2 = time.perf_counter()

        for j in range(1, n_out):
            if j in shared_set:
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
                    lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
            else:
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
                copy_sz = min(out_mems[j].contents.size, in_mems[j].contents.size)
                ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
                if do_own_sync:
                    lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)

        if do_own_sync:
            lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        t3 = time.perf_counter()

        times_run.append((t2-t1)*1000)
        times_copy.append((t3-t2)*1000)
        times_total.append((t3-t0)*1000)

    return {
        'run': np.median(times_run),
        'copy': np.median(times_copy),
        'total': np.median(times_total),
    }


def cleanup(ctx, in_mems, out_mems):
    for m in in_mems:
        lib.rknn_destroy_mem(ctx, m)
    for m in out_mems:
        lib.rknn_destroy_mem(ctx, m)
    lib.rknn_destroy(ctx)


# ─── Main ────────────────────────────────────────────────────────
print("=" * 60)
print("Advanced Zero-Copy Optimizations")
print("=" * 60)

buf = load_model_buf()
results = {}

# Test A: Baseline (no extra flags)
print("\n--- A) Baseline (no extra init flags) ---")
ctx, n_in, n_out, md = init_ctx(buf, 0)
if ctx:
    io = setup_io_baseline(ctx, n_in, n_out)
    if io[0]:
        r = bench(ctx, n_in, n_out, io[0], io[2], label="baseline")
        if r:
            results['A_baseline'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, io[0], io[2])
    else:
        lib.rknn_destroy(ctx)

# Test B: Disable flush flags
print("\n--- B) Disable flush flags ---")
flags_b = RKNN_FLAG_DISABLE_FLUSH_INPUT | RKNN_FLAG_DISABLE_FLUSH_OUTPUT
ctx, n_in, n_out, md = init_ctx(buf, flags_b)
if ctx:
    io = setup_io_baseline(ctx, n_in, n_out)
    if io[0]:
        r = bench(ctx, n_in, n_out, io[0], io[2], do_own_sync=True, label="disable_flush")
        if r:
            results['B_disable_flush'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, io[0], io[2])
    else:
        lib.rknn_destroy(ctx)

# Test C: Enable SRAM
print("\n--- C) Enable SRAM ---")
ctx, n_in, n_out, md = init_ctx(buf, RKNN_FLAG_ENABLE_SRAM)
if ctx:
    io = setup_io_baseline(ctx, n_in, n_out)
    if io[0]:
        r = bench(ctx, n_in, n_out, io[0], io[2], label="sram")
        if r:
            results['C_sram'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, io[0], io[2])
    else:
        lib.rknn_destroy(ctx)

# Test D: Disable flush + SRAM
print("\n--- D) Disable flush + SRAM ---")
flags_d = RKNN_FLAG_DISABLE_FLUSH_INPUT | RKNN_FLAG_DISABLE_FLUSH_OUTPUT | RKNN_FLAG_ENABLE_SRAM
ctx, n_in, n_out, md = init_ctx(buf, flags_d)
if ctx:
    io = setup_io_baseline(ctx, n_in, n_out)
    if io[0]:
        r = bench(ctx, n_in, n_out, io[0], io[2], label="flush+sram")
        if r:
            results['D_flush_sram'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, io[0], io[2])
    else:
        lib.rknn_destroy(ctx)

# Test E: Memory sharing (rknn_create_mem_from_fd)
print("\n--- E) Memory sharing (compatible tensors) ---")
ctx, n_in, n_out, md = init_ctx(buf, 0)
if ctx:
    result = setup_io_with_mem_sharing(ctx, n_in, n_out)
    in_mems, in_attrs, out_mems, out_attrs, shared = result[0], result[1], result[2], result[3], result[4]
    if in_mems:
        print(f"  Shared {len(shared)} tensors: {sorted(shared)}")
        r = bench(ctx, n_in, n_out, in_mems, out_mems, shared_set=shared, label="mem_share")
        if r:
            results['E_mem_share'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, in_mems, out_mems)
    else:
        lib.rknn_destroy(ctx)

# Test F: All combined
print("\n--- F) All combined (flush + SRAM + mem sharing) ---")
flags_f = RKNN_FLAG_DISABLE_FLUSH_INPUT | RKNN_FLAG_DISABLE_FLUSH_OUTPUT | RKNN_FLAG_ENABLE_SRAM
ctx, n_in, n_out, md = init_ctx(buf, flags_f)
if ctx:
    result = setup_io_with_mem_sharing(ctx, n_in, n_out)
    in_mems, in_attrs, out_mems, out_attrs, shared = result[0], result[1], result[2], result[3], result[4]
    if in_mems:
        print(f"  Shared {len(shared)} tensors: {sorted(shared)}")
        r = bench(ctx, n_in, n_out, in_mems, out_mems, shared_set=shared, label="all")
        if r:
            results['F_all'] = r
            print(f"  run={r['run']:.1f}ms  copy={r['copy']:.1f}ms  total={r['total']:.1f}ms")
        cleanup(ctx, in_mems, out_mems)
    else:
        lib.rknn_destroy(ctx)

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for k, v in results.items():
    print(f"  {k:>20}: run={v['run']:.1f}ms  copy={v['copy']:.1f}ms  total={v['total']:.1f}ms")
print(f"  {'ONNX INT8':>20}: total=35ms")
best = min(results.values(), key=lambda x: x['total']) if results else None
if best:
    print(f"\n  Best total: {best['total']:.1f}ms (ONNX diff: {best['total']-35:.1f}ms)")
