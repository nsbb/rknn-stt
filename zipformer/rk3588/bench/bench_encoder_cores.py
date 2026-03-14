"""
Encoder NPU 코어 수별 레이턴시 벤치마크
- NPU_CORE_0 vs NPU_CORE_0_1 vs NPU_CORE_0_1_2 vs NPU_CORE_AUTO
"""
import numpy as np, time, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')

from rknnlite.api import RKNNLite
BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

ENC_SCHEMA = [
    ('x',              [1, 39, 80],        'float32'),
    ('cached_len_0',   [2, 1],             'int64'),
    ('cached_len_1',   [4, 1],             'int64'),
    ('cached_len_2',   [3, 1],             'int64'),
    ('cached_len_3',   [2, 1],             'int64'),
    ('cached_len_4',   [4, 1],             'int64'),
    ('cached_avg_0',   [2, 1, 384],        'float32'),
    ('cached_avg_1',   [4, 1, 384],        'float32'),
    ('cached_avg_2',   [3, 1, 384],        'float32'),
    ('cached_avg_3',   [2, 1, 384],        'float32'),
    ('cached_avg_4',   [4, 1, 384],        'float32'),
    ('cached_key_0',   [2, 64, 1, 192],    'float32'),
    ('cached_key_1',   [4, 32, 1, 192],    'float32'),
    ('cached_key_2',   [3, 16, 1, 192],    'float32'),
    ('cached_key_3',   [2,  8, 1, 192],    'float32'),
    ('cached_key_4',   [4, 32, 1, 192],    'float32'),
    ('cached_val_0',   [2, 64, 1, 96],     'float32'),
    ('cached_val_1',   [4, 32, 1, 96],     'float32'),
    ('cached_val_2',   [3, 16, 1, 96],     'float32'),
    ('cached_val_3',   [2,  8, 1, 96],     'float32'),
    ('cached_val_4',   [4, 32, 1, 96],     'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],     'float32'),
    ('cached_val2_1',  [4, 32, 1, 96],     'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],     'float32'),
    ('cached_val2_3',  [2,  8, 1, 96],     'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],     'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],    'float32'),
    ('cached_conv1_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],    'float32'),
    ('cached_conv2_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],    'float32'),
]

def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))

def make_dummy_inputs():
    inputs = []
    for nm, sh, dt in ENC_SCHEMA:
        a = np.random.randn(*sh).astype(np.dtype(dt)) * 0.1
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inputs.append(a)
    return inputs

cores = [
    ('NPU_CORE_0',     RKNNLite.NPU_CORE_0),
    ('NPU_CORE_0_1',   RKNNLite.NPU_CORE_0_1),
    ('NPU_CORE_0_1_2', RKNNLite.NPU_CORE_0_1_2),
    ('NPU_CORE_AUTO',  RKNNLite.NPU_CORE_AUTO),
]

enc_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn'
dummy = make_dummy_inputs()
N_WARMUP = 3
N_BENCH  = 20

print(f"Encoder: {enc_path}")
print(f"Warmup={N_WARMUP}, Bench={N_BENCH}\n")

for core_name, core_mask in cores:
    m = RKNNLite(verbose=False)
    ret = m.load_rknn(enc_path)
    assert ret == 0, f"load_rknn failed: {ret}"
    ret = m.init_runtime(core_mask=core_mask)
    assert ret == 0, f"init_runtime failed: {ret}"

    # warmup
    for _ in range(N_WARMUP):
        m.inference(inputs=dummy)

    # bench
    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        m.inference(inputs=dummy)
        times.append((time.perf_counter() - t0) * 1000)

    m.release()
    arr = np.array(times)
    print(f"{core_name:20s}: median={np.median(arr):.1f}ms  min={arr.min():.1f}ms  p90={np.percentile(arr,90):.1f}ms")
