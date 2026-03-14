"""
Query RKNN model internal info using C API.
Check layer count, input/output details, and model metadata.
"""
import ctypes, numpy as np, sys, struct
from ctypes import c_int, c_uint32, c_uint64, c_void_p, c_char_p, POINTER, byref, Structure

sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from encoder_capi import lib, RknnTensorAttr, RknnInputOutputNum, RKNN_MAX_DIMS, RKNN_MAX_NAME_LEN

# RKNN_QUERY constants
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 3
RKNN_QUERY_PERF_RUN = 4
RKNN_QUERY_SDK_VERSION = 5
RKNN_QUERY_MEM_SIZE = 6
RKNN_QUERY_CUSTOM_STRING = 7

class RknnPerfDetail(Structure):
    _fields_ = [('run_duration', c_uint64)]  # simplified

class RknnPerfRun(Structure):
    _fields_ = [('run_duration', c_uint64)]

class RknnSdkVersion(Structure):
    _fields_ = [
        ('api_version', ctypes.c_char * 256),
        ('drv_version', ctypes.c_char * 256),
    ]

class RknnMemSize(Structure):
    _fields_ = [
        ('total_weight_size', c_uint32),
        ('total_internal_size', c_uint32),
        ('total_dma_allocated_size', c_uint64),
        ('total_sram_size', c_uint32),
        ('free_sram_size', c_uint32),
        ('reserved', c_uint32 * 10),
    ]

MODELS = {
    'rmreshape': '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn',
    'rmreshape-sim': '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-sim.rknn',
    'cumfix (no rmreshape)': '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix.rknn',
}

for name, path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"  File: {path}")

    with open(path, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(buf), 0, None)
    if ret != 0:
        print(f"  rknn_init FAILED: {ret}")
        continue

    # Query I/O count
    io = RknnInputOutputNum()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, byref(io), ctypes.sizeof(io))
    print(f"  Inputs: {io.n_input}, Outputs: {io.n_output}")

    # Query SDK version
    ver = RknnSdkVersion()
    ret = lib.rknn_query(ctx, RKNN_QUERY_SDK_VERSION, byref(ver), ctypes.sizeof(ver))
    if ret == 0:
        print(f"  API: {ver.api_version.decode()}, DRV: {ver.drv_version.decode()}")

    # Query memory size
    mem = RknnMemSize()
    ret = lib.rknn_query(ctx, RKNN_QUERY_MEM_SIZE, byref(mem), ctypes.sizeof(mem))
    if ret == 0:
        print(f"  Weight: {mem.total_weight_size/1024/1024:.1f}MB, Internal: {mem.total_internal_size/1024/1024:.1f}MB")
        print(f"  DMA: {mem.total_dma_allocated_size/1024/1024:.1f}MB, SRAM: {mem.total_sram_size/1024:.1f}KB (free: {mem.free_sram_size/1024:.1f}KB)")

    # Try perf_run (rknn_run timing from driver)
    perf = RknnPerfRun()
    ret = lib.rknn_query(ctx, RKNN_QUERY_PERF_RUN, byref(perf), ctypes.sizeof(perf))
    if ret == 0:
        print(f"  Perf run: {perf.run_duration}us")

    print(f"  File size: {len(buf)/1024/1024:.1f}MB")

    lib.rknn_destroy(ctx)
