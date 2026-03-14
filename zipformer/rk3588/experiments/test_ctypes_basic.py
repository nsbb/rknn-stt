"""
ctypes RKNN 기본 동작 확인 — 단계별 디버깅
"""
import ctypes, numpy as np, sys
from ctypes import c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64, c_float, c_void_p, POINTER, Structure, byref

lib = ctypes.CDLL('/usr/lib/librknnrt.so')
print("Library loaded OK")

RKNN_MAX_DIMS     = 16
RKNN_MAX_NAME_LEN = 256
RKNN_SUCC         = 0

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

print(f"sizeof RknnTensorAttr = {ctypes.sizeof(RknnTensorAttr)}")
print(f"sizeof RknnTensorMem  = {ctypes.sizeof(RknnTensorMem)}")

# 함수 시그니처
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

RKNN_QUERY_IN_OUT_NUM  = 0
RKNN_QUERY_INPUT_ATTR  = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_NPU_CORE_0        = 1
RKNN_FLAG_MEMORY_NON_CACHEABLE = 1 << 1

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1.rknn'

# ─── Step 1: 모델 로드 ─────────────────────────────────────────────
print("\nStep 1: rknn_init ...")
with open(MODEL, 'rb') as f:
    model_buf = f.read()

# ctypes 버퍼로 변환 (Python이 GC하지 않도록 변수에 보관)
model_data = (ctypes.c_uint8 * len(model_buf)).from_buffer_copy(model_buf)
del model_buf  # 원본은 해제

ctx = c_uint64(0)
ret = lib.rknn_init(byref(ctx), model_data, len(model_data), 0, None)
print(f"  rknn_init ret={ret}, ctx={ctx.value:#x}")
if ret != RKNN_SUCC:
    print("FAIL")
    sys.exit(1)
print("  OK")

# ─── Step 2: Core mask ────────────────────────────────────────────
print("\nStep 2: rknn_set_core_mask ...")
ret = lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)
print(f"  ret={ret}")

# ─── Step 3: Query io_num ─────────────────────────────────────────
print("\nStep 3: rknn_query IN_OUT_NUM ...")
io_num = RknnInputOutputNum()
ret = lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io_num), ctypes.sizeof(io_num))
print(f"  ret={ret}, n_inputs={io_num.n_input}, n_outputs={io_num.n_output}")

n_in  = io_num.n_input
n_out = io_num.n_output

# ─── Step 4: Query input attrs ────────────────────────────────────
print("\nStep 4: Query input attrs ...")
in_attrs = (RknnTensorAttr * n_in)()
for i in range(n_in):
    in_attrs[i].index = i
    ret = lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, byref(in_attrs[i]), ctypes.sizeof(RknnTensorAttr))
    if i < 3:
        nm  = in_attrs[i].name.decode()
        sz  = in_attrs[i].size
        szs = in_attrs[i].size_with_stride
        fmt = in_attrs[i].fmt
        typ = in_attrs[i].type
        print(f"  [{i}] {nm}: size={sz}, size_with_stride={szs}, fmt={fmt}, type={typ}")
print(f"  ... total {n_in} inputs")

# ─── Step 5: Query output attrs ───────────────────────────────────
print("\nStep 5: Query output attrs ...")
out_attrs = (RknnTensorAttr * n_out)()
for i in range(n_out):
    out_attrs[i].index = i
    ret = lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, byref(out_attrs[i]), ctypes.sizeof(RknnTensorAttr))
    if i < 3:
        nm = out_attrs[i].name.decode()
        sz = out_attrs[i].size
        ne = out_attrs[i].n_elems
        print(f"  [{i}] {nm}: size={sz}, n_elems={ne}")
print(f"  ... total {n_out} outputs")

# ─── Step 6: Create mem for first input only ──────────────────────
print("\nStep 6: rknn_create_mem2 (input 0) ...")
sz0  = in_attrs[0].size_with_stride or in_attrs[0].size
mem0 = lib.rknn_create_mem2(ctx, sz0, RKNN_FLAG_MEMORY_NON_CACHEABLE)
print(f"  mem0={mem0}, virt_addr={mem0.contents.virt_addr if mem0 else None}")

# ─── Step 7: set_io_mem for first input ───────────────────────────
print("\nStep 7: rknn_set_io_mem (input 0, pass_through=0, NHWC) ...")
in_attrs[0].pass_through = 0  # 변환 허용
in_attrs[0].fmt  = 1          # NHWC
in_attrs[0].type = 0          # FLOAT32
ret = lib.rknn_set_io_mem(ctx, mem0, byref(in_attrs[0]))
print(f"  ret={ret}")

print("\nBasic steps OK. Cleaning up ...")
if mem0:
    lib.rknn_destroy_mem(ctx, mem0)
lib.rknn_destroy(ctx)
print("Done")
