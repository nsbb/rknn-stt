"""
RKNN C API perf_detail: 38ms rknn_run의 실제 내부 분해.
RKNN_FLAG_COLLECT_PERF_MASK로 init → RKNN_QUERY_PERF_DETAIL/PERF_RUN 쿼리.
"""
import ctypes, numpy as np, time
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_char_p, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 3
RKNN_QUERY_PERF_RUN = 4
RKNN_NPU_CORE_0 = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_FLAG_MEM_CACHEABLE = 0
RKNN_FLAG_COLLECT_PERF = 0x8
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}

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

class RknnPerfDetail(Structure):
    _fields_ = [('perf_data', c_char_p), ('data_len', c_uint64)]

class RknnPerfRun(Structure):
    _fields_ = [('run_duration', c_int32 * 2)]  # int64_t as 2x int32 for alignment

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

print("=" * 60)
print("RKNN perf_detail analysis")
print("=" * 60)

# Load with perf collection enabled
with open(MODEL, 'rb') as f:
    buf = f.read()
md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
ctx = c_uint64(0)
ret = lib.rknn_init(byref(ctx), md, len(md), RKNN_FLAG_COLLECT_PERF, None)
assert ret == 0, f"init failed: {ret}"
lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)

io = RknnInputOutputNum()
lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
n_in, n_out = io.n_input, io.n_output
print(f"Inputs: {n_in}, Outputs: {n_out}")

# Setup IO with set_io_mem (same as baseline)
in_mems = []
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
    sz = native_sz * desired_b // native_b
    mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
    in_mems.append(mem)
    lib.rknn_set_io_mem(ctx, mem, byref(attr))

out_mems = []
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
    mem = lib.rknn_create_mem2(ctx, max(sz, 64), RKNN_FLAG_MEM_CACHEABLE)
    out_mems.append(mem)
    lib.rknn_set_io_mem(ctx, mem, byref(attr))

# Zero init + sync
for i in range(n_in):
    ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
    lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

# Run once to get perf data
print("\nRunning inference...")
t0 = time.perf_counter()
ret = lib.rknn_run(ctx, None)
t1 = time.perf_counter()
print(f"rknn_run ret={ret}, wall time={((t1-t0)*1000):.1f}ms")

# Sync outputs
for i in range(n_out):
    lib.rknn_mem_sync(ctx, out_mems[i], RKNN_MEMORY_SYNC_FROM_DEVICE)

# Query PERF_RUN
perf_run = RknnPerfRun()
ret = lib.rknn_query(ctx, RKNN_QUERY_PERF_RUN, byref(perf_run), ctypes.sizeof(perf_run))
if ret == 0:
    # Combine two int32 into int64
    run_us = perf_run.run_duration[0] + (perf_run.run_duration[1] << 32)
    print(f"\nPERF_RUN: {run_us} us ({run_us/1000:.2f} ms)")
else:
    print(f"\nPERF_RUN query failed: ret={ret}")

# Query PERF_DETAIL
perf_detail = RknnPerfDetail()
ret = lib.rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, byref(perf_detail), ctypes.sizeof(perf_detail))
if ret == 0 and perf_detail.perf_data:
    detail_str = perf_detail.perf_data.decode('utf-8', errors='replace')
    print(f"\nPERF_DETAIL ({perf_detail.data_len} bytes):")
    # Print first 5000 chars
    print(detail_str[:5000])
    if len(detail_str) > 5000:
        print(f"  ... ({len(detail_str)} total chars)")
        # Also print summary/totals at end
        lines = detail_str.strip().split('\n')
        if len(lines) > 10:
            print("\n  ... last 10 lines:")
            for line in lines[-10:]:
                print(f"  {line}")
else:
    print(f"\nPERF_DETAIL query failed: ret={ret}")

# Second run to see consistency
print("\nRunning 2nd inference...")
for i in range(1, n_out):
    copy_sz = min(out_mems[i].contents.size, in_mems[i].contents.size)
    ctypes.memmove(in_mems[i].contents.virt_addr, out_mems[i].contents.virt_addr, copy_sz)
    lib.rknn_mem_sync(ctx, in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)

t0 = time.perf_counter()
ret = lib.rknn_run(ctx, None)
t1 = time.perf_counter()
print(f"2nd rknn_run wall time={((t1-t0)*1000):.1f}ms")

for i in range(n_out):
    lib.rknn_mem_sync(ctx, out_mems[i], RKNN_MEMORY_SYNC_FROM_DEVICE)

ret = lib.rknn_query(ctx, RKNN_QUERY_PERF_RUN, byref(perf_run), ctypes.sizeof(perf_run))
if ret == 0:
    run_us = perf_run.run_duration[0] + (perf_run.run_duration[1] << 32)
    print(f"2nd PERF_RUN: {run_us} us ({run_us/1000:.2f} ms)")

# Cleanup
for m in in_mems + out_mems:
    lib.rknn_destroy_mem(ctx, m)
lib.rknn_destroy(ctx)
