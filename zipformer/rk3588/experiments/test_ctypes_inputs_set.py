"""
rknn_inputs_set + rknn_run + rknn_outputs_get vs set_io_mem + rknn_run 비교
"""
import ctypes, numpy as np, sys, time
from ctypes import c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64, c_float, c_void_p, POINTER, Structure, byref

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS     = 16
RKNN_MAX_NAME_LEN = 256
RKNN_SUCC         = 0
RKNN_QUERY_IN_OUT_NUM  = 0
RKNN_QUERY_INPUT_ATTR  = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_NPU_CORE_0   = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT64   = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEMORY_NON_CACHEABLE = 1 << 1
NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 8: 8}

class RknnInputOutputNum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]

class RknnTensorAttr(Structure):
    _fields_ = [
        ('index',           c_uint32),
        ('n_dims',          c_uint32),
        ('dims',            c_uint32 * RKNN_MAX_DIMS),
        ('name',            ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_elems',         c_uint32),
        ('size',            c_uint32),
        ('fmt',             c_int),
        ('type',            c_int),
        ('qnt_type',        c_int),
        ('fl',              c_int8),
        ('zp',              c_int32),
        ('scale',           c_float),
        ('w_stride',        c_uint32),
        ('size_with_stride', c_uint32),
        ('pass_through',    c_uint8),
        ('h_stride',        c_uint32),
    ]

class RknnTensorMem(Structure):
    _fields_ = [
        ('virt_addr',  c_void_p),
        ('phys_addr',  c_uint64),
        ('fd',         c_int32),
        ('offset',     c_int32),
        ('size',       c_uint32),
        ('flags',      c_uint32),
        ('priv_data',  c_void_p),
    ]

class RknnInput(Structure):
    _fields_ = [
        ('index',        c_uint32),
        ('buf',          c_void_p),
        ('size',         c_uint32),
        ('pass_through', c_uint8),
        ('type',         c_int),
        ('fmt',          c_int),
    ]

class RknnOutput(Structure):
    _fields_ = [
        ('want_float',  c_uint8),
        ('is_prealloc', c_uint8),
        ('index',       c_uint32),
        ('buf',         c_void_p),
        ('size',        c_uint32),
    ]

lib.rknn_init.restype  = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
lib.rknn_destroy.restype  = c_int
lib.rknn_destroy.argtypes = [c_uint64]
lib.rknn_query.restype  = c_int
lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
lib.rknn_set_core_mask.restype  = c_int
lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]
lib.rknn_run.restype  = c_int
lib.rknn_run.argtypes = [c_uint64, c_void_p]
lib.rknn_inputs_set.restype  = c_int
lib.rknn_inputs_set.argtypes = [c_uint64, c_uint32, POINTER(RknnInput)]
lib.rknn_outputs_get.restype  = c_int
lib.rknn_outputs_get.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput), c_void_p]
lib.rknn_outputs_release.restype  = c_int
lib.rknn_outputs_release.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput)]
lib.rknn_create_mem2.restype  = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype  = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]
lib.rknn_set_io_mem.restype  = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1.rknn'

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

def nchw2nhwc(a): return np.ascontiguousarray(np.transpose(a, (0,2,3,1)))

def make_state():
    s = {}
    for nm, sh, dt in ENC_SCHEMA:
        if dt == 'int64':
            s[nm] = np.zeros(sh, dtype=np.int64)
        else:
            s[nm] = (np.random.randn(*sh)*0.1).astype(np.float32)
    return s

def pack_inputs(state):
    inp = []
    for nm, sh, dt in ENC_SCHEMA:
        a = state[nm]
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inp.append(np.ascontiguousarray(a))
    return inp

def load_model():
    with open(MODEL, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    assert lib.rknn_init(byref(ctx), md, len(md), 0, None) == 0
    assert lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0) == 0
    io = RknnInputOutputNum()
    assert lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io)) == 0
    return ctx, md, io.n_input, io.n_output

N_WARM  = 3
N_BENCH = 15

# ═══════════════════════════════════════════════════════════════════
# Method 1: rknn_inputs_set + rknn_run + rknn_outputs_get
# ═══════════════════════════════════════════════════════════════════
print("=== Method 1: rknn_inputs_set + rknn_run + rknn_outputs_get ===")
ctx1, md1, n_in, n_out = load_model()

rin_arr  = (RknnInput  * n_in)()
rout_arr = (RknnOutput * n_out)()

for i in range(n_out):
    rout_arr[i].want_float  = 1
    rout_arr[i].is_prealloc = 0
    rout_arr[i].index       = i

# keep numpy arrays alive
np_inputs = None

def set_and_run_m1(state):
    global np_inputs
    np_inputs = pack_inputs(state)
    for i, arr in enumerate(np_inputs):
        rin_arr[i].index        = i
        rin_arr[i].buf          = arr.ctypes.data_as(c_void_p)
        rin_arr[i].size         = arr.nbytes
        rin_arr[i].pass_through = 0
        rin_arr[i].type         = RKNN_TENSOR_INT64 if arr.dtype == np.int64 else RKNN_TENSOR_FLOAT32
        rin_arr[i].fmt          = RKNN_TENSOR_NHWC if arr.ndim == 4 else RKNN_TENSOR_NCHW
    assert lib.rknn_inputs_set(ctx1, n_in, rin_arr) == 0
    assert lib.rknn_run(ctx1, None) == 0
    assert lib.rknn_outputs_get(ctx1, n_out, rout_arr, None) == 0
    lib.rknn_outputs_release(ctx1, n_out, rout_arr)

state = make_state()
for _ in range(N_WARM):
    set_and_run_m1(make_state())

times1_set, times1_run, times1_get = [], [], []
for _ in range(N_BENCH):
    np_inputs = pack_inputs(make_state())
    for i, arr in enumerate(np_inputs):
        rin_arr[i].index        = i
        rin_arr[i].buf          = arr.ctypes.data_as(c_void_p)
        rin_arr[i].size         = arr.nbytes
        rin_arr[i].pass_through = 0
        rin_arr[i].type         = RKNN_TENSOR_INT64 if arr.dtype == np.int64 else RKNN_TENSOR_FLOAT32
        rin_arr[i].fmt          = RKNN_TENSOR_NHWC if arr.ndim == 4 else RKNN_TENSOR_NCHW

    t0 = time.perf_counter()
    lib.rknn_inputs_set(ctx1, n_in, rin_arr)
    t1 = time.perf_counter()
    lib.rknn_run(ctx1, None)
    t2 = time.perf_counter()
    lib.rknn_outputs_get(ctx1, n_out, rout_arr, None)
    t3 = time.perf_counter()
    lib.rknn_outputs_release(ctx1, n_out, rout_arr)

    times1_set.append((t1-t0)*1000)
    times1_run.append((t2-t1)*1000)
    times1_get.append((t3-t2)*1000)

total1 = [s+r+g for s,r,g in zip(times1_set, times1_run, times1_get)]
print(f"  inputs_set: median={np.median(times1_set):.2f}ms")
print(f"  rknn_run:   median={np.median(times1_run):.2f}ms")
print(f"  outputs_get: median={np.median(times1_get):.2f}ms")
print(f"  TOTAL:      median={np.median(total1):.2f}ms")
lib.rknn_destroy(ctx1)

# ═══════════════════════════════════════════════════════════════════
# Method 2: set_io_mem (pre-allocated) + rknn_run
# ═══════════════════════════════════════════════════════════════════
print("\n=== Method 2: set_io_mem + rknn_run ===")
ctx2, md2, n_in, n_out = load_model()

in_attrs2 = (RknnTensorAttr * n_in)()
out_attrs2 = (RknnTensorAttr * n_out)()
in_mems2  = []
out_mems2 = []

for i in range(n_in):
    in_attrs2[i].index = i
    lib.rknn_query(ctx2, RKNN_QUERY_INPUT_ATTR, byref(in_attrs2[i]), ctypes.sizeof(RknnTensorAttr))
    attr = in_attrs2[i]
    native_t = attr.type
    native_b = NATIVE_BYTES.get(native_t, 2)
    desired_b = 8 if native_t == RKNN_TENSOR_INT64 else 4
    attr.type = RKNN_TENSOR_INT64 if native_t == RKNN_TENSOR_INT64 else RKNN_TENSOR_FLOAT32
    attr.pass_through = 0
    attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
    native_szs = attr.size_with_stride if attr.size_with_stride else attr.size
    sz = native_szs * desired_b // native_b
    mem = lib.rknn_create_mem2(ctx2, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
    in_mems2.append(mem)
    assert lib.rknn_set_io_mem(ctx2, mem, byref(attr)) == 0

for i in range(n_out):
    out_attrs2[i].index = i
    lib.rknn_query(ctx2, RKNN_QUERY_OUTPUT_ATTR, byref(out_attrs2[i]), ctypes.sizeof(RknnTensorAttr))
    attr = out_attrs2[i]
    native_t = attr.type
    desired_b = 8 if native_t == RKNN_TENSOR_INT64 else 4
    attr.type = RKNN_TENSOR_INT64 if native_t == RKNN_TENSOR_INT64 else RKNN_TENSOR_FLOAT32
    attr.pass_through = 0
    sz = attr.n_elems * desired_b
    mem = lib.rknn_create_mem2(ctx2, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
    out_mems2.append(mem)
    assert lib.rknn_set_io_mem(ctx2, mem, byref(attr)) == 0

print("  Buffers set up")

def copy_inputs_m2(np_inputs):
    for i, arr in enumerate(np_inputs):
        flat = np.ascontiguousarray(arr)
        mem  = in_mems2[i]
        nbytes = min(flat.nbytes, mem.contents.size)
        ctypes.memmove(mem.contents.virt_addr, flat.ctypes.data, nbytes)

for _ in range(N_WARM):
    copy_inputs_m2(pack_inputs(make_state()))
    lib.rknn_run(ctx2, None)

times2_copy, times2_run, times2_read = [], [], []
for _ in range(N_BENCH):
    np_inp = pack_inputs(make_state())

    t0 = time.perf_counter()
    copy_inputs_m2(np_inp)
    t1 = time.perf_counter()
    lib.rknn_run(ctx2, None)
    t2 = time.perf_counter()
    # 출력 읽기 (encoder_out만)
    enc_out_mem = out_mems2[0]
    n = out_attrs2[0].n_elems
    arr = np.empty(n, dtype=np.float32)
    ctypes.memmove(arr.ctypes.data, enc_out_mem.contents.virt_addr, n*4)
    t3 = time.perf_counter()

    times2_copy.append((t1-t0)*1000)
    times2_run.append((t2-t1)*1000)
    times2_read.append((t3-t2)*1000)

total2 = [c+r+rd for c,r,rd in zip(times2_copy, times2_run, times2_read)]
print(f"  copy_inputs: median={np.median(times2_copy):.2f}ms")
print(f"  rknn_run:    median={np.median(times2_run):.2f}ms")
print(f"  read_output: median={np.median(times2_read):.2f}ms")
print(f"  TOTAL:       median={np.median(total2):.2f}ms")

for mem in in_mems2: lib.rknn_destroy_mem(ctx2, mem)
for mem in out_mems2: lib.rknn_destroy_mem(ctx2, mem)
lib.rknn_destroy(ctx2)

# ═══════════════════════════════════════════════════════════════════
# Method 3: rknnlite.api inference() (기준)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Method 3: rknnlite.api inference() ===")
from rknnlite.api import RKNNLite
enc_r = RKNNLite(verbose=False)
enc_r.load_rknn(MODEL)
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

for _ in range(N_WARM):
    enc_r.inference(inputs=pack_inputs(make_state()))

times3 = []
for _ in range(N_BENCH):
    inp = pack_inputs(make_state())
    t0 = time.perf_counter()
    enc_r.inference(inputs=inp)
    times3.append((time.perf_counter()-t0)*1000)

print(f"  TOTAL: median={np.median(times3):.2f}ms  min={min(times3):.2f}ms")
enc_r.release()

print("\n=== 요약 ===")
print(f"Method 1 (inputs_set+run+get): {np.median(total1):.2f}ms  (set={np.median(times1_set):.2f}, run={np.median(times1_run):.2f}, get={np.median(times1_get):.2f})")
print(f"Method 2 (set_io_mem+run):     {np.median(total2):.2f}ms  (copy={np.median(times2_copy):.2f}, run={np.median(times2_run):.2f})")
print(f"Method 3 (rknnlite.api):       {np.median(times3):.2f}ms")
