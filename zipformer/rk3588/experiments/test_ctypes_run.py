"""
ctypes zero-copy 단계별 테스트 — set_io_mem 성공 후 rknn_run 확인
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
RKNN_NPU_CORE_0        = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8    = 2
RKNN_TENSOR_INT64   = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEMORY_NON_CACHEABLE = 1 << 1
RKNN_MEMORY_SYNC_TO_DEVICE   = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

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
lib.rknn_create_mem2.restype  = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype  = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]
lib.rknn_set_io_mem.restype  = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]
lib.rknn_mem_sync.restype  = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(RknnTensorMem), c_int]

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1.rknn'

# ─── init ─────────────────────────────────────────────────────────
with open(MODEL, 'rb') as f:
    buf = f.read()
model_data = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
del buf

ctx = c_uint64(0)
assert lib.rknn_init(byref(ctx), model_data, len(model_data), 0, None) == 0
assert lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0) == 0

io_num = RknnInputOutputNum()
assert lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io_num), ctypes.sizeof(io_num)) == 0
n_in, n_out = io_num.n_input, io_num.n_output
print(f"n_in={n_in}, n_out={n_out}")

# ─── query + set_io_mem for all inputs ────────────────────────────
in_attrs = (RknnTensorAttr * n_in)()
in_mems  = []

print("\nSetting up input buffers ...")
for i in range(n_in):
    in_attrs[i].index = i
    assert lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,
                          byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr)) == 0
    attr = in_attrs[i]
    nm = attr.name.decode()
    native_t = attr.type
    native_b = NATIVE_BYTES.get(native_t, 2)

    if native_t == RKNN_TENSOR_INT64:
        desired_b = 8
        attr.type = RKNN_TENSOR_INT64
    else:
        desired_b = 4
        attr.type = RKNN_TENSOR_FLOAT32

    attr.pass_through = 0
    attr.fmt = RKNN_TENSOR_NHWC if attr.n_dims == 4 else RKNN_TENSOR_NCHW
    native_szs = attr.size_with_stride if attr.size_with_stride else attr.size
    sz = native_szs * desired_b // native_b

    mem = lib.rknn_create_mem2(ctx, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
    assert mem, f"create_mem2 failed for input {i}"
    in_mems.append(mem)

    ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
    if ret != 0:
        print(f"  FAIL input {i} ({nm}): ret={ret}, sz={sz}, n_elems={attr.n_elems}, native_b={native_b}, szs={native_szs}")
        sys.exit(1)
    if i % 5 == 0:
        print(f"  [{i}] {nm}: sz={sz} OK")

print("All input buffers set OK")

# ─── query + set_io_mem for all outputs ───────────────────────────
out_attrs = (RknnTensorAttr * n_out)()
out_mems  = []

print("\nSetting up output buffers ...")
for i in range(n_out):
    out_attrs[i].index = i
    assert lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
                          byref(out_attrs[i]), ctypes.sizeof(RknnTensorAttr)) == 0
    attr = out_attrs[i]
    nm = attr.name.decode()
    native_t = attr.type

    if native_t == RKNN_TENSOR_INT64:
        desired_b = 8
        attr.type = RKNN_TENSOR_INT64
    else:
        desired_b = 4
        attr.type = RKNN_TENSOR_FLOAT32

    attr.pass_through = 0
    # 출력은 packed 형태로 반환 → n_elems * desired_bytes
    sz = attr.n_elems * desired_b

    mem = lib.rknn_create_mem2(ctx, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
    assert mem, f"create_mem2 failed for output {i}"
    out_mems.append(mem)

    ret = lib.rknn_set_io_mem(ctx, mem, byref(attr))
    if ret != 0:
        print(f"  FAIL output {i} ({nm}): ret={ret}, sz={sz}")
        sys.exit(1)
    if i % 5 == 0:
        print(f"  [{i}] {nm}: sz={sz} OK")

print("All output buffers set OK")

# ─── 입력 데이터 쓰기 ───────────────────────────────────────────────
print("\nWriting input data to DMA buffers ...")
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

def nchw2nhwc(a): return np.transpose(a, (0,2,3,1))

for i, (nm, sh, dt) in enumerate(ENC_SCHEMA):
    if dt == 'int64':
        arr = np.zeros(sh, dtype=np.int64)
    else:
        arr = (np.random.randn(*sh) * 0.1).astype(np.float32)
    if len(sh) == 4:
        arr = nchw2nhwc(arr)
    flat = np.ascontiguousarray(arr)
    mem = in_mems[i]
    nbytes = flat.nbytes
    buf_sz = mem.contents.size
    if nbytes > buf_sz:
        print(f"  WARNING: [{i}] {nm} data={nbytes} > buf={buf_sz}")
        # write only buf_sz bytes
        nbytes = buf_sz
    ctypes.memmove(mem.contents.virt_addr, flat.ctypes.data, nbytes)

print("Input data written OK")

# ─── rknn_run ────────────────────────────────────────────────────
print("\nRunning rknn_run ...")
N_WARM = 3
N_BENCH = 10

for _ in range(N_WARM):
    ret = lib.rknn_run(ctx, None)
    print(f"  warmup rknn_run ret={ret}")

times = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    ret = lib.rknn_run(ctx, None)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
    if ret != 0:
        print(f"  rknn_run failed: ret={ret}")
        break

print(f"\nrknn_run times: median={np.median(times):.2f}ms  min={min(times):.2f}ms")

# ─── 출력 읽기 ───────────────────────────────────────────────────
print("\nReading outputs ...")
enc_out_mem = out_mems[0]
enc_out_attr = out_attrs[0]
n_elems = enc_out_attr.n_elems
arr = np.empty(n_elems, dtype=np.float32)
ctypes.memmove(arr.ctypes.data, enc_out_mem.contents.virt_addr, n_elems * 4)
dims = list(enc_out_attr.dims[:enc_out_attr.n_dims])
enc_out = arr.reshape(dims)
print(f"  encoder_out shape: {enc_out.shape}, max={enc_out.max():.4f}, min={enc_out.min():.4f}")

# ─── 정리 ───────────────────────────────────────────────────────
print("\nCleaning up ...")
for mem in in_mems:
    lib.rknn_destroy_mem(ctx, mem)
for mem in out_mems:
    lib.rknn_destroy_mem(ctx, mem)
lib.rknn_destroy(ctx)
print("Done")
