"""
하이브리드 모드에서 RKNN vs ONNX 실제 레이턴시 측정
- RKNN + ONNX 동시 로드 상황에서의 각 단계 시간
- 순차 vs 병렬 실행 비교
"""
import numpy as np, time, sys, onnxruntime as ort, threading
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
CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']

def nchw2nhwc(a): return np.transpose(a, (0,2,3,1))

def make_state():
    s = {}
    for nm, sh, dt in ENC_SCHEMA:
        if dt == 'int64':
            s[nm] = np.zeros(sh, dtype=np.int64)
        else:
            s[nm] = (np.random.randn(*sh) * 0.1).astype(np.float32)
    return s

state = make_state()

print('Loading RKNN ...')
enc_r = RKNNLite(verbose=False)
enc_r.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
print('  RKNN ok')

print('Loading ONNX ...')
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
enc_s = ort.InferenceSession(
    f'{BASE}/encoder-epoch-99-avg-1.onnx',
    sess_options=opts,
    providers=['CPUExecutionProvider']
)
print('  ONNX ok\n')

def pack_rknn(state):
    inp = []
    for nm, sh, _ in ENC_SCHEMA:
        a = state[nm]
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inp.append(a)
    return inp

# Warmup
for _ in range(3):
    enc_r.inference(inputs=pack_rknn(state))
    onnx_out = enc_s.run(None, {nm: state[nm] for nm,_,_ in ENC_SCHEMA})
    for i, nm in enumerate(CACHE_NAMES):
        state[nm] = np.array(onnx_out[i+1])

N = 15

# --- Test 1: Sequential (RKNN then ONNX) ---
rknn_times, onnx_times = [], []
state2 = make_state()

for _ in range(N):
    rknn_in = pack_rknn(state2)
    onnx_in = {nm: state2[nm] for nm,_,_ in ENC_SCHEMA}

    t0 = time.perf_counter()
    rknn_out = enc_r.inference(inputs=rknn_in)
    t1 = time.perf_counter()
    onnx_out = enc_s.run(None, onnx_in)
    t2 = time.perf_counter()

    rknn_times.append((t1-t0)*1000)
    onnx_times.append((t2-t1)*1000)
    for i, nm in enumerate(CACHE_NAMES):
        state2[nm] = np.array(onnx_out[i+1])

print('=== Sequential (RKNN → ONNX) ===')
print(f'  RKNN: median={np.median(rknn_times):.1f}ms  min={min(rknn_times):.1f}ms')
print(f'  ONNX: median={np.median(onnx_times):.1f}ms  min={min(onnx_times):.1f}ms')
seq_total = [r+o for r,o in zip(rknn_times, onnx_times)]
print(f'  Total: median={np.median(seq_total):.1f}ms  min={min(seq_total):.1f}ms')

# --- Test 2: Parallel (RKNN and ONNX concurrently) ---
# Run RKNN and ONNX in parallel threads, then combine
state3 = make_state()

parallel_times = []
for _ in range(N):
    rknn_in = pack_rknn(state3)
    onnx_in = {nm: state3[nm] for nm,_,_ in ENC_SCHEMA}

    rknn_result = [None]
    onnx_result = [None]

    def run_rknn():
        rknn_result[0] = enc_r.inference(inputs=rknn_in)

    def run_onnx():
        onnx_result[0] = enc_s.run(None, onnx_in)

    t0 = time.perf_counter()
    t_rknn = threading.Thread(target=run_rknn)
    t_onnx = threading.Thread(target=run_onnx)
    t_rknn.start(); t_onnx.start()
    t_rknn.join(); t_onnx.join()
    t1 = time.perf_counter()

    parallel_times.append((t1-t0)*1000)
    for i, nm in enumerate(CACHE_NAMES):
        state3[nm] = np.array(onnx_result[0][i+1])

print()
print('=== Parallel (RKNN || ONNX) ===')
print(f'  Total: median={np.median(parallel_times):.1f}ms  min={min(parallel_times):.1f}ms')
print(f'  Speedup vs sequential: {np.median(seq_total)/np.median(parallel_times):.2f}x')

# --- Test 3: Pure ONNX (no RKNN) ---
state4 = make_state()
onnx_only_times = []

for _ in range(N):
    onnx_in = {nm: state4[nm] for nm,_,_ in ENC_SCHEMA}
    t0 = time.perf_counter()
    onnx_out = enc_s.run(None, onnx_in)
    t1 = time.perf_counter()
    onnx_only_times.append((t1-t0)*1000)
    for i, nm in enumerate(CACHE_NAMES):
        state4[nm] = np.array(onnx_out[i+1])

print()
print('=== Pure ONNX (4-thread) ===')
print(f'  Total: median={np.median(onnx_only_times):.1f}ms  min={min(onnx_only_times):.1f}ms')

enc_r.release()
print('\n=== Summary ===')
print(f'  Sequential hybrid: {np.median(seq_total):.1f}ms/chunk')
print(f'  Parallel hybrid:   {np.median(parallel_times):.1f}ms/chunk')
print(f'  Pure ONNX:         {np.median(onnx_only_times):.1f}ms/chunk')
