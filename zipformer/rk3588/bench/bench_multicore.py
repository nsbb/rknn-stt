"""
Multi-core NPU benchmark using direct C API.
RK3588 has 3 NPU cores. Test core_mask 1,2,4,3,7.
"""
import numpy as np, time, ctypes
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
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

lib.rknn_init.restype = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
lib.rknn_destroy.restype = c_int; lib.rknn_destroy.argtypes = [c_uint64]
lib.rknn_query.restype = c_int; lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
lib.rknn_set_core_mask.restype = c_int; lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]
lib.rknn_run.restype = c_int; lib.rknn_run.argtypes = [c_uint64, c_void_p]
lib.rknn_create_mem2.restype = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]
lib.rknn_set_io_mem.restype = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]
lib.rknn_mem_sync.restype = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(RknnTensorMem), c_int]

NOCACHE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-int8-cumfix-nocache.rknn'
BASELINE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'


def bench(model_path, core_mask, label, n_runs=80, warmup=15):
    with open(model_path, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(buf), 0, None)
    assert ret == 0, f"init failed: {ret}"
    ret = lib.rknn_set_core_mask(ctx, core_mask)
    if ret != 0:
        print(f"  {label}: set_core_mask({core_mask}) FAILED: {ret}")
        lib.rknn_destroy(ctx)
        return None

    io = RknnInputOutputNum()
    lib.rknn_query(ctx, 0, byref(io), ctypes.sizeof(io))

    in_mems, in_attrs = [], []
    for i in range(io.n_input):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, 1, byref(attr), ctypes.sizeof(attr))
        nt = attr.type
        db = 8 if nt == 8 else 4
        if nt == 8: attr.type = 8
        else: attr.type = 0
        attr.pass_through = 0
        attr.fmt = 1 if attr.n_dims == 4 else 0
        nb = NATIVE_BYTES.get(nt, 2)
        nsz = attr.size_with_stride if attr.size_with_stride else attr.size
        mem = lib.rknn_create_mem2(ctx, max(nsz * db // nb, 64), 0)
        in_mems.append(mem); in_attrs.append(attr)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    out_mems, out_attrs = [], []
    for i in range(io.n_output):
        attr = RknnTensorAttr(); attr.index = i
        lib.rknn_query(ctx, 2, byref(attr), ctypes.sizeof(attr))
        nt = attr.type
        db = 8 if nt == 8 else 4
        if nt == 8: attr.type = 8
        else: attr.type = 0
        attr.pass_through = 0
        mem = lib.rknn_create_mem2(ctx, max(attr.n_elems * db, 64), 0)
        out_mems.append(mem); out_attrs.append(attr)
        lib.rknn_set_io_mem(ctx, mem, byref(attr))

    for i in range(io.n_input):
        sz = in_attrs[i].size_with_stride if in_attrs[i].size_with_stride else in_attrs[i].size
        ctypes.memset(in_mems[i].contents.virt_addr, 0, min(sz, 1024*1024))
        lib.rknn_mem_sync(ctx, in_mems[i], 0x1)

    for _ in range(warmup):
        lib.rknn_run(ctx, None)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    for m in in_mems + out_mems:
        lib.rknn_destroy_mem(ctx, m)
    lib.rknn_destroy(ctx)

    med = np.median(times)
    mn = np.min(times)
    print(f"  {label:40s} median={med:7.2f}ms  min={mn:7.2f}ms")
    return {'median': med, 'min': mn}


def main():
    print("=== Multi-core NPU Benchmark ===\n")
    for model, mname in [(NOCACHE, "nocache"), (BASELINE, "baseline")]:
        print(f"Model: {mname}")
        for mask, desc in [(1, "Core0"), (2, "Core1"), (4, "Core2"), (3, "Core0+1"), (7, "Core0+1+2")]:
            try:
                bench(model, mask, f"{mname} mask={mask} ({desc})")
            except Exception as e:
                print(f"  {mname} mask={mask} ({desc}): ERROR - {e}")
        print()


if __name__ == '__main__':
    main()
