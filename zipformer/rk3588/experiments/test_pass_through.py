"""
Step 2: pass_through=1 테스트
cache 텐서를 native format(INT8)으로 직접 전달하여 format 변환 제거.

비교:
  A) set_io_mem + CACHEABLE + pass_through=0 (전 단계 결과: 39.6ms)
  B) set_io_mem + CACHEABLE + pass_through=1 (cache만, x는 pt=0 유지)
  C) rknn_inputs_set 기준선 (45ms)
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

NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}  # 3=U8

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

ENC_SCHEMA = [
    ('x',[1,39,80],'float32'),('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),
    ('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),
    ('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),
    ('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),
    ('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),
    ('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),
    ('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),
    ('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
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


def query_native_attrs(ctx, n_in, n_out):
    """Query native input/output attrs"""
    in_native = []
    for i in range(n_in):
        a = RknnTensorAttr()
        a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(a), ctypes.sizeof(a))
        in_native.append(a)
    out_native = []
    for i in range(n_out):
        a = RknnTensorAttr()
        a.index = i
        lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(a), ctypes.sizeof(a))
        out_native.append(a)
    return in_native, out_native


def bench_pass_through(ctx, n_in, n_out, in_native, out_native):
    """
    pass_through=1 for all cache inputs (native INT8).
    x: pass_through=0 (float32, runtime converts).
    All outputs: pass_through=1 (native format).
    """
    in_attrs = (RknnTensorAttr * n_in)()
    in_mems = []

    for i in range(n_in):
        na = in_native[i]
        in_attrs[i].index = i

        if i == 0:
            # x: pass_through=0, float32 → runtime converts
            lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr))
            attr = in_attrs[i]
            attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            attr.fmt = RKNN_TENSOR_NCHW
            # size for float32: n_elems * 4
            sz = attr.n_elems * 4
        elif ENC_SCHEMA[i][2] == 'int64':
            # cached_len: pass_through=1, native INT64
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr))
            attr = in_attrs[i]
            attr.pass_through = 1
            sz = na.size_with_stride if na.size_with_stride else na.size
        else:
            # cache tensors: pass_through=1, native INT8
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr))
            attr = in_attrs[i]
            attr.pass_through = 1
            sz = na.size_with_stride if na.size_with_stride else na.size

        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 failed input {i}"
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            nm = ENC_SCHEMA[i][0]
            print(f"  FAIL input {i} ({nm}): ret={ret} sz={sz} pt={attr.pass_through}")
            return None

    # Output: all pass_through=1 (native format), except encoder_out (pt=0 for float32)
    out_attrs = (RknnTensorAttr * n_out)()
    out_mems = []

    for i in range(n_out):
        if i == 0:
            # encoder_out: pass_through=0, want float32
            out_attrs[i].index = i
            lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(out_attrs[i]), ctypes.sizeof(RknnTensorAttr))
            attr = out_attrs[i]
            attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            sz = attr.n_elems * 4
        else:
            # cache outputs: pass_through=1, native format
            out_attrs[i].index = i
            lib.rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, byref(out_attrs[i]), ctypes.sizeof(RknnTensorAttr))
            attr = out_attrs[i]
            attr.pass_through = 1
            na = out_native[i]
            sz = na.size_with_stride if na.size_with_stride else na.size

        mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
        assert mem, f"create_mem2 failed output {i}"
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        if ret != 0:
            print(f"  FAIL output {i}: ret={ret} sz={sz} pt={attr.pass_through}")
            return None

    # Write zeros to all input buffers (initial cache = 0)
    for i in range(n_in):
        mem = in_mems[i]
        if i == 0:
            # x: write float32
            x = np.zeros([1, 39, 80], dtype=np.float32)
            ctypes.memmove(mem.contents.virt_addr, x.ctypes.data, x.nbytes)
        else:
            # cache: write zeros in native format
            ctypes.memset(mem.contents.virt_addr, 0, mem.contents.size)
        lib.rknn_mem_sync(ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE)

    # Warmup
    N_WARM = 3
    N_BENCH = 20
    for _ in range(N_WARM):
        lib.rknn_run(ctx, None)

    # Benchmark
    times_write = []
    times_sync_in = []
    times_run = []
    times_copy_cache = []
    times_read = []
    times_total = []

    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        # Write x only (cache stays from previous run for now)
        x = np.zeros([1, 39, 80], dtype=np.float32)
        ctypes.memmove(in_mems[0].contents.virt_addr, x.ctypes.data, x.nbytes)
        t1 = time.perf_counter()

        # Sync x to device
        lib.rknn_mem_sync(ctx, in_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE)
        t2 = time.perf_counter()

        # Run
        lib.rknn_run(ctx, None)
        t3 = time.perf_counter()

        # Copy cache outputs → cache inputs for next iteration
        # For matching native formats: direct memcpy
        # For non-matching: need conversion (skip for now, just measure time)
        for j in range(1, n_out):
            out_sz = out_mems[j].contents.size
            in_sz = in_mems[j].contents.size
            copy_sz = min(out_sz, in_sz)
            lib.rknn_mem_sync(ctx, out_mems[j], RKNN_MEMORY_SYNC_FROM_DEVICE)
            ctypes.memmove(in_mems[j].contents.virt_addr, out_mems[j].contents.virt_addr, copy_sz)
            lib.rknn_mem_sync(ctx, in_mems[j], RKNN_MEMORY_SYNC_TO_DEVICE)
        t4 = time.perf_counter()

        # Read encoder_out
        lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        enc = np.empty(out_attrs[0].n_elems, dtype=np.float32)
        ctypes.memmove(enc.ctypes.data, out_mems[0].contents.virt_addr, enc.nbytes)
        t5 = time.perf_counter()

        times_write.append((t1 - t0) * 1000)
        times_sync_in.append((t2 - t1) * 1000)
        times_run.append((t3 - t2) * 1000)
        times_copy_cache.append((t4 - t3) * 1000)
        times_read.append((t5 - t4) * 1000)
        times_total.append((t5 - t0) * 1000)

    print("\n=== pass_through=1 (cache native INT8) ===")
    print(f"  write x:       {np.median(times_write):.2f}ms")
    print(f"  sync x:        {np.median(times_sync_in):.2f}ms")
    print(f"  run:           {np.median(times_run):.2f}ms")
    print(f"  copy cache:    {np.median(times_copy_cache):.2f}ms")
    print(f"  read enc_out:  {np.median(times_read):.2f}ms")
    print(f"  TOTAL:         {np.median(times_total):.2f}ms")
    print(f"  (run 만:       {np.median(times_run):.2f}ms)")

    # 정리
    for mem in in_mems:
        lib.rknn_destroy_mem(ctx, mem)
    for mem in out_mems:
        lib.rknn_destroy_mem(ctx, mem)

    return np.median(times_total)


# ─── 실행 ────────────────────────────────────────────────────────
print("=" * 60)
print("pass_through=1 Benchmark")
print("=" * 60)

ctx, n_in, n_out, md = load_model()
in_native, out_native = query_native_attrs(ctx, n_in, n_out)

# 입출력 native format 요약
total_in = sum(a.size_with_stride if a.size_with_stride else a.size for a in in_native)
total_out = sum(a.size_with_stride if a.size_with_stride else a.size for a in out_native)
print(f"Native input total: {total_in:,} bytes ({total_in/1024:.0f}KB)")
print(f"Native output total: {total_out:,} bytes ({total_out/1024:.0f}KB)")

t = bench_pass_through(ctx, n_in, n_out, in_native, out_native)
lib.rknn_destroy(ctx)

if t:
    print(f"\n  pass_through=1: {t:.1f}ms")
    print(f"  이전 pt=0:      39.6ms")
    print(f"  inputs_set:     45.0ms")
    print(f"  ONNX INT8:      35ms")
