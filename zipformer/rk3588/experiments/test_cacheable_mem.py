"""
Step 1: Cacheable 메모리 + mem_sync 테스트
이전 실패 원인: RKNN_FLAG_MEMORY_NON_CACHEABLE(flag=2) 사용 → CPU 쓰기 ~33MB/s
수정: flag=0 (CACHEABLE default) + rknn_mem_sync → CPU 캐시 대역폭 ~3GB/s 기대

비교:
  A) set_io_mem + CACHEABLE (pass_through=0, float32)
  B) set_io_mem + NON_CACHEABLE (이전 방식, 비교용)
  C) rknn_inputs_set (현재 방식, 기준선)
"""
import ctypes, numpy as np, sys, time
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_SUCC = 0
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_NPU_CORE_0 = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1

RKNN_FLAG_MEM_CACHEABLE = 0  # default = cacheable
RKNN_FLAG_MEM_NON_CACHEABLE = 1 << 1  # = 2
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 8: 8}

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

class RknnInput(Structure):
    _fields_ = [('index', c_uint32), ('buf', c_void_p), ('size', c_uint32),
                ('pass_through', c_uint8), ('type', c_int), ('fmt', c_int)]

class RknnOutput(Structure):
    _fields_ = [('want_float', c_uint8), ('is_prealloc', c_uint8),
                ('index', c_uint32), ('buf', c_void_p), ('size', c_uint32)]

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
lib.rknn_inputs_set.restype = c_int
lib.rknn_inputs_set.argtypes = [c_uint64, c_uint32, POINTER(RknnInput)]
lib.rknn_outputs_get.restype = c_int
lib.rknn_outputs_get.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput), c_void_p]
lib.rknn_outputs_release.restype = c_int
lib.rknn_outputs_release.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput)]

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

def nchw2nhwc(a):
    return np.ascontiguousarray(np.transpose(a, (0,2,3,1)))

# 입력 데이터 준비 (한 번만)
input_arrays = []
for nm, sh, dt in ENC_SCHEMA:
    if dt == 'int64':
        arr = np.zeros(sh, dtype=np.int64)
    else:
        arr = np.zeros(sh, dtype=np.float32)
    if len(sh) == 4:
        arr = nchw2nhwc(arr)
    input_arrays.append(np.ascontiguousarray(arr))


def load_model():
    with open(MODEL, 'rb') as f:
        buf = f.read()
    model_data = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    assert lib.rknn_init(byref(ctx), model_data, len(model_data), 0, None) == 0
    assert lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0) == 0
    io_num = RknnInputOutputNum()
    assert lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io_num), ctypes.sizeof(io_num)) == 0
    return ctx, io_num.n_input, io_num.n_output, model_data


def bench_set_io_mem(ctx, n_in, n_out, flag, label):
    """set_io_mem 방식 벤치마크"""
    # 입력 버퍼 설정
    in_attrs = (RknnTensorAttr * n_in)()
    in_mems = []
    for i in range(n_in):
        in_attrs[i].index = i
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr))
        attr = in_attrs[i]
        native_t = attr.type
        native_b = NATIVE_BYTES.get(native_t, 2)
        if native_t == RKNN_TENSOR_INT64:
            attr.type = RKNN_TENSOR_INT64; desired_b = 8
        else:
            attr.type = RKNN_TENSOR_FLOAT32; desired_b = 4
        attr.pass_through = 0
        attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
        native_szs = attr.size_with_stride if attr.size_with_stride else attr.size
        sz = native_szs * desired_b // native_b
        mem = lib.rknn_create_mem2(ctx, sz, flag)
        assert mem, f"create_mem2 failed input {i}"
        in_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        assert ret == 0, f"set_io_mem input {i} failed: {ret}"

    # 출력 버퍼 설정
    out_attrs = (RknnTensorAttr * n_out)()
    out_mems = []
    for i in range(n_out):
        out_attrs[i].index = i
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(out_attrs[i]), ctypes.sizeof(RknnTensorAttr))
        attr = out_attrs[i]
        native_t = attr.type
        if native_t == RKNN_TENSOR_INT64:
            attr.type = RKNN_TENSOR_INT64; desired_b = 8
        else:
            attr.type = RKNN_TENSOR_FLOAT32; desired_b = 4
        attr.pass_through = 0
        sz = attr.n_elems * desired_b
        mem = lib.rknn_create_mem2(ctx, sz, flag)
        assert mem, f"create_mem2 failed output {i}"
        out_mems.append(mem)
        ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
        assert ret == 0, f"set_io_mem output {i} failed: {ret}"

    # 벤치마크: memmove + sync + run + sync
    N_WARM = 3
    N_BENCH = 20

    for _ in range(N_WARM):
        for i in range(n_in):
            arr = input_arrays[i]
            ctypes.memmove(in_mems[i].contents.virt_addr, arr.ctypes.data, arr.nbytes)
        if flag == RKNN_FLAG_MEM_CACHEABLE:
            for i in range(n_in):
                lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)
        lib.rknn_run(ctx, None)
        if flag == RKNN_FLAG_MEM_CACHEABLE:
            lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)

    times_total = []
    times_write = []
    times_sync = []
    times_run = []
    times_read = []

    for _ in range(N_BENCH):
        # write
        t0 = time.perf_counter()
        for i in range(n_in):
            arr = input_arrays[i]
            ctypes.memmove(in_mems[i].contents.virt_addr, arr.ctypes.data, arr.nbytes)
        t1 = time.perf_counter()

        # sync to device (cacheable only)
        if flag == RKNN_FLAG_MEM_CACHEABLE:
            for i in range(n_in):
                lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)
        t2 = time.perf_counter()

        # run
        lib.rknn_run(ctx, None)
        t3 = time.perf_counter()

        # sync from device + read encoder_out
        if flag == RKNN_FLAG_MEM_CACHEABLE:
            lib.rknn_mem_sync(ctx, out_mems[0], RKNN_MEMORY_SYNC_FROM_DEVICE)
        enc_out = np.empty(out_attrs[0].n_elems, dtype=np.float32)
        ctypes.memmove(enc_out.ctypes.data, out_mems[0].contents.virt_addr, enc_out.nbytes)
        t4 = time.perf_counter()

        times_write.append((t1-t0)*1000)
        times_sync.append((t2-t1)*1000)
        times_run.append((t3-t2)*1000)
        times_read.append((t4-t3)*1000)
        times_total.append((t4-t0)*1000)

    print(f"\n=== {label} ===")
    print(f"  write:   {np.median(times_write):.2f}ms")
    print(f"  sync:    {np.median(times_sync):.2f}ms")
    print(f"  run:     {np.median(times_run):.2f}ms")
    print(f"  read:    {np.median(times_read):.2f}ms")
    print(f"  TOTAL:   {np.median(times_total):.2f}ms")

    # 정리
    for mem in in_mems:
        lib.rknn_destroy_mem(ctx, mem)
    for mem in out_mems:
        lib.rknn_destroy_mem(ctx, mem)

    return np.median(times_total)


def bench_inputs_set(ctx, n_in, n_out, label):
    """rknn_inputs_set 방식 벤치마크 (기준선)"""
    N_WARM = 3
    N_BENCH = 20

    for _ in range(N_WARM):
        inputs = (RknnInput * n_in)()
        for i in range(n_in):
            arr = input_arrays[i]
            inputs[i].index = i
            inputs[i].buf = arr.ctypes.data
            inputs[i].size = arr.nbytes
            inputs[i].pass_through = 0
            inputs[i].type = RKNN_TENSOR_INT64 if ENC_SCHEMA[i][2] == 'int64' else RKNN_TENSOR_FLOAT32
            inputs[i].fmt = RKNN_TENSOR_NHWC if len(ENC_SCHEMA[i][1]) == 4 else RKNN_TENSOR_NCHW
        lib.rknn_inputs_set(ctx, n_in, inputs)
        lib.rknn_run(ctx, None)
        outputs = (RknnOutput * n_out)()
        for i in range(n_out):
            outputs[i].want_float = 1
            outputs[i].is_prealloc = 0
        lib.rknn_outputs_get(ctx, n_out, outputs, None)
        lib.rknn_outputs_release(ctx, n_out, outputs)

    times_total = []
    times_set = []
    times_run = []
    times_get = []

    for _ in range(N_BENCH):
        inputs = (RknnInput * n_in)()
        for i in range(n_in):
            arr = input_arrays[i]
            inputs[i].index = i
            inputs[i].buf = arr.ctypes.data
            inputs[i].size = arr.nbytes
            inputs[i].pass_through = 0
            inputs[i].type = RKNN_TENSOR_INT64 if ENC_SCHEMA[i][2] == 'int64' else RKNN_TENSOR_FLOAT32
            inputs[i].fmt = RKNN_TENSOR_NHWC if len(ENC_SCHEMA[i][1]) == 4 else RKNN_TENSOR_NCHW

        t0 = time.perf_counter()
        lib.rknn_inputs_set(ctx, n_in, inputs)
        t1 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t2 = time.perf_counter()
        outputs = (RknnOutput * n_out)()
        for i in range(n_out):
            outputs[i].want_float = 1
            outputs[i].is_prealloc = 0
        lib.rknn_outputs_get(ctx, n_out, outputs, None)
        t3 = time.perf_counter()
        lib.rknn_outputs_release(ctx, n_out, outputs)

        times_set.append((t1-t0)*1000)
        times_run.append((t2-t1)*1000)
        times_get.append((t3-t2)*1000)
        times_total.append((t3-t0)*1000)

    print(f"\n=== {label} ===")
    print(f"  set:     {np.median(times_set):.2f}ms")
    print(f"  run:     {np.median(times_run):.2f}ms")
    print(f"  get:     {np.median(times_get):.2f}ms")
    print(f"  TOTAL:   {np.median(times_total):.2f}ms")

    return np.median(times_total)


# ─── 실행 ────────────────────────────────────────────────────────
print("=" * 60)
print("RKNN Cacheable vs Non-Cacheable Memory Benchmark")
print("=" * 60)

# Method C: inputs_set (기준선)
ctx, n_in, n_out, md = load_model()
t_c = bench_inputs_set(ctx, n_in, n_out, "C: rknn_inputs_set (기준선)")
lib.rknn_destroy(ctx)

# Method A: set_io_mem + CACHEABLE
ctx, n_in, n_out, md = load_model()
t_a = bench_set_io_mem(ctx, n_in, n_out, RKNN_FLAG_MEM_CACHEABLE, "A: set_io_mem + CACHEABLE + mem_sync")
lib.rknn_destroy(ctx)

# Method B: set_io_mem + NON_CACHEABLE (이전 방식)
ctx, n_in, n_out, md = load_model()
t_b = bench_set_io_mem(ctx, n_in, n_out, RKNN_FLAG_MEM_NON_CACHEABLE, "B: set_io_mem + NON_CACHEABLE (이전)")
lib.rknn_destroy(ctx)

print("\n" + "=" * 60)
print("결과 비교")
print("=" * 60)
print(f"  A (CACHEABLE):      {t_a:.1f}ms")
print(f"  B (NON_CACHEABLE):  {t_b:.1f}ms")
print(f"  C (inputs_set):     {t_c:.1f}ms")
print(f"  ONNX INT8 기준:     35ms")
print(f"\n  A vs C 개선:        {t_c - t_a:.1f}ms ({t_c/t_a:.2f}x)")
