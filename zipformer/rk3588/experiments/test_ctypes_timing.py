"""
rknn_run 내부 타이밍 분석
- 같은 데이터 반복 vs 새 데이터
- RKNN_QUERY_PERF_RUN으로 NPU 실제 계산 시간 확인
- pass_through=1 (float16 NHWC) 시도
"""
import ctypes, numpy as np, sys, time
from ctypes import c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64, c_float, c_void_p, c_int64, POINTER, Structure, byref

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS     = 16
RKNN_MAX_NAME_LEN = 256
RKNN_SUCC         = 0
RKNN_QUERY_IN_OUT_NUM  = 0
RKNN_QUERY_INPUT_ATTR  = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_RUN    = 4
RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR  = 10
RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR = 11
RKNN_NPU_CORE_0   = 1
RKNN_FLAG_COLLECT_PERF_MASK = 0x00000008
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT64   = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 8: 8}

class RknnInputOutputNum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]

class RknnTensorAttr(Structure):
    _fields_ = [
        ('index', c_uint32), ('n_dims', c_uint32), ('dims', c_uint32*RKNN_MAX_DIMS),
        ('name', ctypes.c_char*RKNN_MAX_NAME_LEN), ('n_elems', c_uint32), ('size', c_uint32),
        ('fmt', c_int), ('type', c_int), ('qnt_type', c_int), ('fl', c_int8),
        ('zp', c_int32), ('scale', c_float), ('w_stride', c_uint32), ('size_with_stride', c_uint32),
        ('pass_through', c_uint8), ('h_stride', c_uint32),
    ]

class RknnInput(Structure):
    _fields_ = [('index', c_uint32), ('buf', c_void_p), ('size', c_uint32),
                ('pass_through', c_uint8), ('type', c_int), ('fmt', c_int)]

class RknnOutput(Structure):
    _fields_ = [('want_float', c_uint8), ('is_prealloc', c_uint8),
                ('index', c_uint32), ('buf', c_void_p), ('size', c_uint32)]

class RknnPerfRun(Structure):
    _fields_ = [('run_duration', c_int64)]

lib.rknn_init.restype = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
lib.rknn_destroy.restype = c_int; lib.rknn_destroy.argtypes = [c_uint64]
lib.rknn_query.restype = c_int; lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
lib.rknn_set_core_mask.restype = c_int; lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]
lib.rknn_run.restype = c_int; lib.rknn_run.argtypes = [c_uint64, c_void_p]
lib.rknn_inputs_set.restype = c_int; lib.rknn_inputs_set.argtypes = [c_uint64, c_uint32, POINTER(RknnInput)]
lib.rknn_outputs_get.restype = c_int; lib.rknn_outputs_get.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput), c_void_p]
lib.rknn_outputs_release.restype = c_int; lib.rknn_outputs_release.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput)]

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

def make_inputs_fp32():
    inp = []
    for nm, sh, dt in ENC_SCHEMA:
        a = np.zeros(sh, dtype=np.int64) if dt=='int64' else (np.random.randn(*sh)*0.1).astype(np.float32)
        if len(sh)==4: a = nchw2nhwc(a)
        inp.append(np.ascontiguousarray(a))
    return inp

def make_inputs_fp16():
    """float16 버전 (int64는 그대로)"""
    inp = []
    for nm, sh, dt in ENC_SCHEMA:
        if dt=='int64':
            a = np.zeros(sh, dtype=np.int64)
        else:
            a = (np.random.randn(*sh)*0.1).astype(np.float16)
            if len(sh)==4: a = nchw2nhwc(a)
        inp.append(np.ascontiguousarray(a))
    return inp

def load_model(perf_mode=False):
    with open(MODEL,'rb') as f: buf = f.read()
    md = (ctypes.c_uint8*len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    flag = RKNN_FLAG_COLLECT_PERF_MASK if perf_mode else 0
    assert lib.rknn_init(byref(ctx), md, len(md), flag, None) == 0
    assert lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0) == 0
    io = RknnInputOutputNum()
    assert lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io)) == 0
    return ctx, md, io.n_input, io.n_output

N_WARM=3; N_BENCH=15

# ─── 공통 벤치마크 함수 ─────────────────────────────────────────────
def bench_method(ctx, n_in, n_out, make_fn, input_type, pass_through=0):
    rin  = (RknnInput  * n_in)()
    rout = (RknnOutput * n_out)()
    for i in range(n_out):
        rout[i].want_float=1; rout[i].is_prealloc=0; rout[i].index=i

    _inputs = None  # 살아있도록 유지

    def one_run(inps):
        nonlocal _inputs
        _inputs = inps
        for i, arr in enumerate(inps):
            rin[i].index = i
            rin[i].buf   = arr.ctypes.data_as(c_void_p)
            rin[i].size  = arr.nbytes
            rin[i].pass_through = pass_through
            rin[i].type  = RKNN_TENSOR_INT64 if arr.dtype==np.int64 else input_type
            rin[i].fmt   = RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW
        lib.rknn_inputs_set(ctx, n_in, rin)
        lib.rknn_run(ctx, None)
        lib.rknn_outputs_get(ctx, n_out, rout, None)
        lib.rknn_outputs_release(ctx, n_out, rout)

    for _ in range(N_WARM):
        one_run(make_fn())

    ts, tr, tg = [], [], []
    for _ in range(N_BENCH):
        inps = make_fn()
        _inputs = inps
        for i, arr in enumerate(inps):
            rin[i].index=i; rin[i].buf=arr.ctypes.data_as(c_void_p)
            rin[i].size=arr.nbytes; rin[i].pass_through=pass_through
            rin[i].type=RKNN_TENSOR_INT64 if arr.dtype==np.int64 else input_type
            rin[i].fmt=RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW

        t0=time.perf_counter()
        lib.rknn_inputs_set(ctx, n_in, rin)
        t1=time.perf_counter()
        lib.rknn_run(ctx, None)
        t2=time.perf_counter()
        lib.rknn_outputs_get(ctx, n_out, rout, None)
        t3=time.perf_counter()
        lib.rknn_outputs_release(ctx, n_out, rout)
        ts.append((t1-t0)*1000); tr.append((t2-t1)*1000); tg.append((t3-t2)*1000)

    return ts, tr, tg


# ═══════════════════════════════════════════════════════════════════
# Test 1: float32 NHWC, pass_through=0 (기준)
# ═══════════════════════════════════════════════════════════════════
print("=== Test 1: float32 NHWC, pass_through=0 ===")
ctx1, md1, n_in, n_out = load_model()
ts1, tr1, tg1 = bench_method(ctx1, n_in, n_out, make_inputs_fp32, RKNN_TENSOR_FLOAT32, 0)
t1 = [s+r+g for s,r,g in zip(ts1,tr1,tg1)]
print(f"  set={np.median(ts1):.1f}ms  run={np.median(tr1):.1f}ms  get={np.median(tg1):.1f}ms  TOTAL={np.median(t1):.1f}ms")
lib.rknn_destroy(ctx1)

# ═══════════════════════════════════════════════════════════════════
# Test 2: float16 NHWC, pass_through=0 (type 변환만 없앰)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 2: float16 NHWC, pass_through=0 ===")
ctx2, md2, n_in, n_out = load_model()
ts2, tr2, tg2 = bench_method(ctx2, n_in, n_out, make_inputs_fp16, RKNN_TENSOR_FLOAT16, 0)
t2 = [s+r+g for s,r,g in zip(ts2,tr2,tg2)]
print(f"  set={np.median(ts2):.1f}ms  run={np.median(tr2):.1f}ms  get={np.median(tg2):.1f}ms  TOTAL={np.median(t2):.1f}ms")
lib.rknn_destroy(ctx2)

# ═══════════════════════════════════════════════════════════════════
# Test 3: float16 NHWC, pass_through=1 (변환 없음)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 3: float16 NHWC, pass_through=1 (no conversion) ===")
ctx3, md3, n_in, n_out = load_model()
ts3, tr3, tg3 = bench_method(ctx3, n_in, n_out, make_inputs_fp16, RKNN_TENSOR_FLOAT16, 1)
t3 = [s+r+g for s,r,g in zip(ts3,tr3,tg3)]
print(f"  set={np.median(ts3):.1f}ms  run={np.median(tr3):.1f}ms  get={np.median(tg3):.1f}ms  TOTAL={np.median(t3):.1f}ms")
lib.rknn_destroy(ctx3)

# ═══════════════════════════════════════════════════════════════════
# Test 4: 같은 입력 반복 (캐시 효과 측정)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 4: 같은 float32 입력 반복 (no cache update) ===")
ctx4, md4, n_in, n_out = load_model()
fixed_inputs = make_inputs_fp32()
rin4  = (RknnInput  * n_in)()
rout4 = (RknnOutput * n_out)()
for i in range(n_out): rout4[i].want_float=1; rout4[i].is_prealloc=0; rout4[i].index=i

for i, arr in enumerate(fixed_inputs):
    rin4[i].index=i; rin4[i].buf=arr.ctypes.data_as(c_void_p)
    rin4[i].size=arr.nbytes; rin4[i].pass_through=0
    rin4[i].type=RKNN_TENSOR_INT64 if arr.dtype==np.int64 else RKNN_TENSOR_FLOAT32
    rin4[i].fmt=RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW

for _ in range(N_WARM):
    lib.rknn_inputs_set(ctx4, n_in, rin4)
    lib.rknn_run(ctx4, None)
    lib.rknn_outputs_get(ctx4, n_out, rout4, None)
    lib.rknn_outputs_release(ctx4, n_out, rout4)

ts4, tr4, tg4 = [], [], []
for _ in range(N_BENCH):
    t0=time.perf_counter()
    lib.rknn_inputs_set(ctx4, n_in, rin4)
    t1=time.perf_counter()
    lib.rknn_run(ctx4, None)
    t2=time.perf_counter()
    lib.rknn_outputs_get(ctx4, n_out, rout4, None)
    t3=time.perf_counter()
    lib.rknn_outputs_release(ctx4, n_out, rout4)
    ts4.append((t1-t0)*1000); tr4.append((t2-t1)*1000); tg4.append((t3-t2)*1000)

t4 = [s+r+g for s,r,g in zip(ts4,tr4,tg4)]
print(f"  set={np.median(ts4):.1f}ms  run={np.median(tr4):.1f}ms  get={np.median(tg4):.1f}ms  TOTAL={np.median(t4):.1f}ms")

# RKNN_QUERY_PERF_RUN: NPU 실제 계산 시간
perf = RknnPerfRun()
ret = lib.rknn_query(ctx4, RKNN_QUERY_PERF_RUN, byref(perf), ctypes.sizeof(perf))
print(f"  NPU perf_run: {perf.run_duration/1000:.2f}ms (ret={ret})")
lib.rknn_destroy(ctx4)

# ═══════════════════════════════════════════════════════════════════
# Test 5: PERF 모드로 float32 실행, NPU 시간 확인
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 5: PERF 모드 (NPU 내부 시간 측정) ===")
ctx5, md5, n_in, n_out = load_model(perf_mode=True)
rin5  = (RknnInput  * n_in)()
rout5 = (RknnOutput * n_out)()
for i in range(n_out): rout5[i].want_float=1; rout5[i].is_prealloc=0; rout5[i].index=i

inputs5 = make_inputs_fp32()
for i, arr in enumerate(inputs5):
    rin5[i].index=i; rin5[i].buf=arr.ctypes.data_as(c_void_p)
    rin5[i].size=arr.nbytes; rin5[i].pass_through=0
    rin5[i].type=RKNN_TENSOR_INT64 if arr.dtype==np.int64 else RKNN_TENSOR_FLOAT32
    rin5[i].fmt=RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW

lib.rknn_inputs_set(ctx5, n_in, rin5)
lib.rknn_run(ctx5, None)
lib.rknn_outputs_get(ctx5, n_out, rout5, None)
lib.rknn_outputs_release(ctx5, n_out, rout5)

perf5 = RknnPerfRun()
ret = lib.rknn_query(ctx5, RKNN_QUERY_PERF_RUN, byref(perf5), ctypes.sizeof(perf5))
print(f"  NPU perf_run: {perf5.run_duration/1000:.2f}ms (ret={ret})")
lib.rknn_destroy(ctx5)

print("\n=== 요약 ===")
print(f"Test 1 (fp32 NHWC, pt=0): {np.median(t1):.1f}ms")
print(f"Test 2 (fp16 NHWC, pt=0): {np.median(t2):.1f}ms")
print(f"Test 3 (fp16 NHWC, pt=1): {np.median(t3):.1f}ms")
print(f"Test 4 (fp32 NHWC, pt=0, fixed inputs): {np.median(t4):.1f}ms")
