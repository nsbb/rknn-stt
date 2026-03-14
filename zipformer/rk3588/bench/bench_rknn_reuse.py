"""
RKNN input 재사용 vs 매번 생성 비교
- 동일 array object 재사용 시 RKNN DMA 캐싱 여부 확인
"""
import numpy as np, time, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from rknnlite.api import RKNNLite

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

ENC_SCHEMA = [
    ('x',[1,39,80],'float32'),
    ('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]

def nchw2nhwc(a): return np.transpose(a, (0,2,3,1))

def make_state():
    s = {}
    for nm, sh, dt in ENC_SCHEMA:
        if dt == 'int64':
            s[nm] = np.zeros(sh, dtype=np.int64)
        else:
            s[nm] = (np.random.randn(*sh) * 0.1).astype(np.float32)
    return s

enc_r = RKNNLite(verbose=False)
enc_r.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

N_WARM = 3
N_BENCH = 20

# Test A: reuse exact same list and array objects each call
# Pre-allocate contiguous NHWC buffers, update in-place
print('=== Test A: Pre-allocated, in-place update ===')
# Pre-allocate the input list once
inplace_bufs = []
buf_map = {}  # nm -> index in inplace_bufs
for i, (nm, sh, dt) in enumerate(ENC_SCHEMA):
    if dt == 'int64':
        buf = np.zeros(sh, dtype=np.int64)
    elif len(sh) == 4:
        # NHWC shape
        nhwc_sh = (sh[0], sh[2], sh[3], sh[1])
        buf = np.zeros(nhwc_sh, dtype=np.float32)
    else:
        buf = np.zeros(sh, dtype=np.float32)
    inplace_bufs.append(buf)
    buf_map[nm] = i

def update_inplace_bufs(state):
    for nm, sh, dt in ENC_SCHEMA:
        idx = buf_map[nm]
        if len(sh) == 4:
            np.copyto(inplace_bufs[idx], nchw2nhwc(state[nm]))
        else:
            np.copyto(inplace_bufs[idx], state[nm])

state = make_state()
update_inplace_bufs(state)

for _ in range(N_WARM):
    enc_r.inference(inputs=inplace_bufs)

times_a = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    enc_r.inference(inputs=inplace_bufs)
    times_a.append((time.perf_counter() - t0) * 1000)

print(f'  median={np.median(times_a):.2f}ms  min={min(times_a):.2f}ms  max={max(times_a):.2f}ms')

# Test B: new list created each call (like current inference code)
print('=== Test B: New list each call (current behavior) ===')

def pack_new(state):
    inp = []
    for nm, sh, _ in ENC_SCHEMA:
        a = state[nm]
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inp.append(a)
    return inp

state2 = make_state()
for _ in range(N_WARM):
    enc_r.inference(inputs=pack_new(state2))

times_b = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    enc_r.inference(inputs=pack_new(state2))
    times_b.append((time.perf_counter() - t0) * 1000)

print(f'  median={np.median(times_b):.2f}ms  min={min(times_b):.2f}ms  max={max(times_b):.2f}ms')

# Test C: new list but with .copy() (contiguous)
print('=== Test C: New list, contiguous (.copy()) ===')

def pack_contig(state):
    inp = []
    for nm, sh, _ in ENC_SCHEMA:
        a = state[nm]
        if len(sh) == 4:
            a = nchw2nhwc(a).copy()  # force contiguous
        inp.append(a)
    return inp

state3 = make_state()
for _ in range(N_WARM):
    enc_r.inference(inputs=pack_contig(state3))

times_c = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    enc_r.inference(inputs=pack_contig(state3))
    times_c.append((time.perf_counter() - t0) * 1000)

print(f'  median={np.median(times_c):.2f}ms  min={min(times_c):.2f}ms  max={max(times_c):.2f}ms')

enc_r.release()

print('\n=== Summary ===')
print(f'A (in-place, reuse same objects):   {np.median(times_a):.2f}ms')
print(f'B (new list, non-contig transpose):  {np.median(times_b):.2f}ms')
print(f'C (new list, contiguous .copy()):    {np.median(times_c):.2f}ms')
print(f'Speedup A vs B: {np.median(times_b)/np.median(times_a):.1f}x')
