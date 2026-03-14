"""
INT8 vs FP16 encoder 속도 비교
- FP16: pass_through=1, float16 입력 (Test 3 방식)
- INT8: pass_through=0, float32 입력 (안전한 방식)
"""
import ctypes, numpy as np, time
from ctypes import c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64, c_float, c_void_p, POINTER, Structure, byref

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_QUERY_IN_OUT_NUM=0; RKNN_QUERY_INPUT_ATTR=1; RKNN_QUERY_OUTPUT_ATTR=2
RKNN_NPU_CORE_0=1
RKNN_TENSOR_FLOAT32=0; RKNN_TENSOR_FLOAT16=1; RKNN_TENSOR_INT64=8
RKNN_TENSOR_NCHW=0; RKNN_TENSOR_NHWC=1
RKNN_MAX_DIMS=16; RKNN_MAX_NAME_LEN=256
NATIVE_BYTES={0:4,1:2,2:1,8:8}

class Num(Structure): _fields_=[('n_input',c_uint32),('n_output',c_uint32)]
class Attr(Structure):
    _fields_=[('index',c_uint32),('n_dims',c_uint32),('dims',c_uint32*RKNN_MAX_DIMS),
              ('name',ctypes.c_char*RKNN_MAX_NAME_LEN),('n_elems',c_uint32),('size',c_uint32),
              ('fmt',c_int),('type',c_int),('qnt_type',c_int),('fl',c_int8),
              ('zp',c_int32),('scale',c_float),('w_stride',c_uint32),('size_with_stride',c_uint32),
              ('pass_through',c_uint8),('h_stride',c_uint32)]
class Inp(Structure):
    _fields_=[('index',c_uint32),('buf',c_void_p),('size',c_uint32),
              ('pass_through',c_uint8),('type',c_int),('fmt',c_int)]
class Out(Structure):
    _fields_=[('want_float',c_uint8),('is_prealloc',c_uint8),
              ('index',c_uint32),('buf',c_void_p),('size',c_uint32)]

lib.rknn_init.restype=c_int; lib.rknn_init.argtypes=[POINTER(c_uint64),c_void_p,c_uint32,c_uint32,c_void_p]
lib.rknn_destroy.restype=c_int; lib.rknn_destroy.argtypes=[c_uint64]
lib.rknn_query.restype=c_int; lib.rknn_query.argtypes=[c_uint64,c_int,c_void_p,c_uint32]
lib.rknn_set_core_mask.restype=c_int; lib.rknn_set_core_mask.argtypes=[c_uint64,c_int]
lib.rknn_run.restype=c_int; lib.rknn_run.argtypes=[c_uint64,c_void_p]
lib.rknn_inputs_set.restype=c_int; lib.rknn_inputs_set.argtypes=[c_uint64,c_uint32,POINTER(Inp)]
lib.rknn_outputs_get.restype=c_int; lib.rknn_outputs_get.argtypes=[c_uint64,c_uint32,POINTER(Out),c_void_p]
lib.rknn_outputs_release.restype=c_int; lib.rknn_outputs_release.argtypes=[c_uint64,c_uint32,POINTER(Out)]

BASE='/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR=f'{BASE}/rk3588'
FP16_MODEL=f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn'
INT8_MODEL=f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8.rknn'

SCHEMA=[
    ('x',[1,39,80],'float32'),
    ('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),
    ('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
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

def nchw2nhwc(a): return np.ascontiguousarray(np.transpose(a,(0,2,3,1)))

def make_inps(dtype=np.float32):
    res=[]
    for nm,sh,dt in SCHEMA:
        if dt=='int64': a=np.zeros(sh,dtype=np.int64)
        else:
            a=(np.random.randn(*sh)*0.1).astype(dtype)
            if len(sh)==4: a=nchw2nhwc(a)
        res.append(np.ascontiguousarray(a))
    return res

def run_bench(model_path, input_dtype, pass_through_val, n_warm=3, n_bench=15, label=""):
    with open(model_path,'rb') as f: buf=f.read()
    md=(ctypes.c_uint8*len(buf)).from_buffer_copy(buf)
    ctx=c_uint64(0)
    assert lib.rknn_init(byref(ctx),md,len(md),0,None)==0
    assert lib.rknn_set_core_mask(ctx,RKNN_NPU_CORE_0)==0
    io=Num()
    assert lib.rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,byref(io),ctypes.sizeof(io))==0
    n_in,n_out=io.n_input,io.n_output

    rin  = (Inp * n_in)()
    rout = (Out * n_out)()
    for i in range(n_out): rout[i].want_float=1; rout[i].is_prealloc=0; rout[i].index=i

    def do_run(inps):
        for i,arr in enumerate(inps):
            rin[i].index=i
            rin[i].buf=arr.ctypes.data_as(c_void_p)
            rin[i].size=arr.nbytes
            rin[i].pass_through=pass_through_val
            if arr.dtype==np.int64: rin[i].type=RKNN_TENSOR_INT64
            elif arr.dtype==np.float16: rin[i].type=RKNN_TENSOR_FLOAT16
            else: rin[i].type=RKNN_TENSOR_FLOAT32
            rin[i].fmt=RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW
        r=lib.rknn_inputs_set(ctx,n_in,rin)
        if r!=0: print(f"  inputs_set ret={r}")
        r=lib.rknn_run(ctx,None)
        if r!=0: print(f"  rknn_run ret={r}")
        r=lib.rknn_outputs_get(ctx,n_out,rout,None)
        if r!=0: print(f"  outputs_get ret={r}")
        lib.rknn_outputs_release(ctx,n_out,rout)

    for _ in range(n_warm): do_run(make_inps(input_dtype))

    times=[]
    for _ in range(n_bench):
        inps=make_inps(input_dtype)
        for i,arr in enumerate(inps):
            rin[i].index=i; rin[i].buf=arr.ctypes.data_as(c_void_p); rin[i].size=arr.nbytes
            rin[i].pass_through=pass_through_val
            if arr.dtype==np.int64: rin[i].type=RKNN_TENSOR_INT64
            elif arr.dtype==np.float16: rin[i].type=RKNN_TENSOR_FLOAT16
            else: rin[i].type=RKNN_TENSOR_FLOAT32
            rin[i].fmt=RKNN_TENSOR_NHWC if arr.ndim==4 else RKNN_TENSOR_NCHW
        t0=time.perf_counter()
        lib.rknn_inputs_set(ctx,n_in,rin)
        lib.rknn_run(ctx,None)
        lib.rknn_outputs_get(ctx,n_out,rout,None)
        times.append((time.perf_counter()-t0)*1000)
        lib.rknn_outputs_release(ctx,n_out,rout)

    m=float(np.median(times))
    print(f"  {label}: median={m:.2f}ms  min={min(times):.2f}ms")
    lib.rknn_destroy(ctx)
    return m

print("=== ctypes 방식 비교 ===")
t1=run_bench(FP16_MODEL, np.float16, 1, label="FP16 encoder (pt=1, fp16)")
t2=run_bench(INT8_MODEL, np.float32, 0, label="INT8 encoder (pt=0, fp32)")
t3=run_bench(FP16_MODEL, np.float32, 0, label="FP16 encoder (pt=0, fp32)")

print("\n=== rknnlite.api 기준 ===")
from rknnlite.api import RKNNLite

def bench_lite(path, label):
    enc=RKNNLite(verbose=False); enc.load_rknn(path); enc.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    def mk():
        return [nchw2nhwc((np.random.randn(*sh)*0.1).astype(np.float32)) if len(sh)==4
                else np.zeros(sh,dtype=np.int64) if dt=='int64'
                else (np.random.randn(*sh)*0.1).astype(np.float32)
                for _,sh,dt in SCHEMA]
    for _ in range(3): enc.inference(inputs=mk())
    ts=[]
    for _ in range(15):
        i=mk(); t0=time.perf_counter(); enc.inference(inputs=i); ts.append((time.perf_counter()-t0)*1000)
    m=float(np.median(ts))
    print(f"  {label}: median={m:.2f}ms")
    enc.release()
    return m

t4=bench_lite(FP16_MODEL,"FP16 rknnlite (fp32 NHWC)")
t5=bench_lite(INT8_MODEL,"INT8 rknnlite (fp32 NHWC)")

print(f"\n=== 요약 ===")
print(f"FP16 ctypes pt=1 fp16: {t1:.1f}ms  (rknnlite: {t4:.1f}ms, speedup: {t4/t1:.2f}x)")
print(f"INT8 ctypes pt=0 fp32: {t2:.1f}ms  (rknnlite: {t5:.1f}ms, speedup: {t5/t2:.2f}x)")
print(f"FP16 ctypes pt=0 fp32: {t3:.1f}ms")
print(f"ONNX INT8 4-thread baseline: 35ms")
print(f"Best RKNN: {min(t1,t2,t3):.1f}ms")
