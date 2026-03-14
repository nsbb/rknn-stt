"""
Parse RKNN perf_detail: extract per-layer times, sort by time, show Top-N.
Also separate CPU vs NPU layers.
"""
import ctypes, numpy as np, re
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_char_p, c_void_p, POINTER, Structure, byref)

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

class RknnPerfDetail(Structure):
    _fields_ = [('perf_data', c_char_p), ('data_len', c_uint64)]

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

# Load with perf collection
with open(MODEL, 'rb') as f:
    buf = f.read()
md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
ctx = c_uint64(0)
lib.rknn_init(byref(ctx), md, len(md), 0x8, None)  # RKNN_FLAG_COLLECT_PERF_MASK
lib.rknn_set_core_mask(ctx, 1)

io = RknnInputOutputNum()
lib.rknn_query(ctx, 0, byref(io), ctypes.sizeof(io))
n_in, n_out = io.n_input, io.n_output

# Setup IO
in_mems = []
for i in range(n_in):
    attr = RknnTensorAttr(); attr.index = i
    lib.rknn_query(ctx, 1, byref(attr), ctypes.sizeof(attr))
    nt = attr.type; nb = NATIVE_BYTES.get(nt, 2)
    attr.type = 8 if nt == 8 else 0
    db = 8 if nt == 8 else 4
    attr.pass_through = 0
    attr.fmt = 1 if attr.n_dims == 4 else 0  # NHWC/NCHW
    nsz = attr.size_with_stride if attr.size_with_stride else attr.size
    mem = lib.rknn_create_mem2(ctx, max(nsz * db // nb, 64), 0)
    in_mems.append(mem)
    lib.rknn_set_io_mem(ctx, mem, byref(attr))

out_mems = []
for i in range(n_out):
    attr = RknnTensorAttr(); attr.index = i
    lib.rknn_query(ctx, 2, byref(attr), ctypes.sizeof(attr))
    nt = attr.type
    attr.type = 8 if nt == 8 else 0
    db = 8 if nt == 8 else 4
    attr.pass_through = 0
    mem = lib.rknn_create_mem2(ctx, max(attr.n_elems * db, 64), 0)
    out_mems.append(mem)
    lib.rknn_set_io_mem(ctx, mem, byref(attr))

for i in range(n_in):
    ctypes.memset(in_mems[i].contents.virt_addr, 0, in_mems[i].contents.size)
    lib.rknn_mem_sync(ctx, in_mems[i], 0x1)

# Run
lib.rknn_run(ctx, None)
for i in range(n_out):
    lib.rknn_mem_sync(ctx, out_mems[i], 0x2)

# Get perf_detail
pd = RknnPerfDetail()
lib.rknn_query(ctx, 3, byref(pd), ctypes.sizeof(pd))

detail = pd.perf_data.decode('utf-8', errors='replace') if pd.perf_data else ""

# Parse lines
lines = detail.strip().split('\n')

# Find data lines (start with number)
layers = []
for line in lines:
    line = line.strip()
    if not line or line.startswith('-') or line.startswith('ID'):
        continue
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        try:
            layer_id = int(parts[0])
            op_type = parts[1]
            dtype = parts[2]
            target = parts[3]
            # Find Time(us) - it's after the cycles field
            # Parse from the end backwards since shapes can have variable parts
            time_us = None
            for j, p in enumerate(parts):
                if '/' in p and j > 3:  # Cycles field like "0/0/0"
                    # Next field should be Time(us)
                    if j + 1 < len(parts):
                        try:
                            time_us = int(parts[j+1])
                            break
                        except ValueError:
                            pass
            if time_us is not None:
                layers.append({
                    'id': layer_id, 'op': op_type, 'dtype': dtype,
                    'target': target, 'time_us': time_us
                })
        except (ValueError, IndexError):
            pass

# Analysis
total_us = sum(l['time_us'] for l in layers)
cpu_us = sum(l['time_us'] for l in layers if l['target'] == 'CPU')
npu_us = sum(l['time_us'] for l in layers if l['target'] == 'NPU')
other_us = total_us - cpu_us - npu_us

print("=" * 60)
print("PERF DETAIL ANALYSIS")
print("=" * 60)
print(f"Total layers: {len(layers)}")
print(f"Total time: {total_us} us ({total_us/1000:.1f} ms)")
print(f"  CPU: {cpu_us} us ({cpu_us/1000:.1f} ms, {cpu_us/total_us*100:.1f}%)")
print(f"  NPU: {npu_us} us ({npu_us/1000:.1f} ms, {npu_us/total_us*100:.1f}%)")
if other_us:
    print(f"  Other: {other_us} us ({other_us/1000:.1f} ms, {other_us/total_us*100:.1f}%)")

# Top 20 by time
print(f"\n--- Top 20 layers by time ---")
sorted_layers = sorted(layers, key=lambda x: x['time_us'], reverse=True)
print(f"{'ID':>4} {'Op':>20} {'Target':>6} {'Time(us)':>10} {'%':>6}")
for l in sorted_layers[:20]:
    pct = l['time_us'] / total_us * 100
    print(f"{l['id']:4d} {l['op']:>20} {l['target']:>6} {l['time_us']:10d} {pct:6.2f}%")

# By op type
from collections import defaultdict
op_times = defaultdict(lambda: {'count': 0, 'time': 0, 'cpu': 0, 'npu': 0})
for l in layers:
    k = l['op']
    op_times[k]['count'] += 1
    op_times[k]['time'] += l['time_us']
    if l['target'] == 'CPU':
        op_times[k]['cpu'] += l['time_us']
    else:
        op_times[k]['npu'] += l['time_us']

print(f"\n--- By op type (sorted by total time) ---")
print(f"{'Op':>25} {'Count':>6} {'Total(us)':>10} {'CPU(us)':>10} {'NPU(us)':>10} {'%':>6}")
for op, d in sorted(op_times.items(), key=lambda x: x[1]['time'], reverse=True):
    pct = d['time'] / total_us * 100
    print(f"{op:>25} {d['count']:6d} {d['time']:10d} {d['cpu']:10d} {d['npu']:10d} {pct:6.2f}%")

# CPU-only op analysis
print(f"\n--- CPU-only operations ---")
cpu_ops = {op: d for op, d in op_times.items() if d['cpu'] > 0}
print(f"{'Op':>25} {'Count':>6} {'CPU(us)':>10} {'%ofTotal':>8}")
for op, d in sorted(cpu_ops.items(), key=lambda x: x[1]['cpu'], reverse=True):
    pct = d['cpu'] / total_us * 100
    print(f"{op:>25} {d['count']:6d} {d['cpu']:10d} {pct:8.2f}%")

# Input/Output operator times
io_time = sum(l['time_us'] for l in layers if l['op'] in ('InputOperator', 'OutputOperator'))
print(f"\nInput+Output operator time: {io_time} us ({io_time/1000:.1f} ms, {io_time/total_us*100:.1f}%)")

for m in in_mems + out_mems:
    lib.rknn_destroy_mem(ctx, m)
lib.rknn_destroy(ctx)
