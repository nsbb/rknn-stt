"""
RKNN vs ONNX 캐시 상태 비교 (1 청크)
- 동일한 입력으로 RKNN/ONNX 동시 실행
- encoder_out, 일부 캐시 값 비교
"""
import numpy as np
import onnxruntime as ort
import sys, os

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
sys.path.insert(0, RKNN_DIR)

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
CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']

def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))

np.random.seed(42)

# 공통 입력
state = {}
for name, shape, dtype in ENC_SCHEMA:
    if 'len' in name:
        state[name] = np.zeros(shape, dtype=np.int64)
    else:
        state[name] = (np.random.randn(*shape) * 0.01).astype(np.float32)
state['x'] = (np.random.randn(1, 39, 80) * 0.3).astype(np.float32)

# ── ONNX 추론 ───────────────────────────────────────────────
sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
onnx_in = {name: state[name] for name, _, _ in ENC_SCHEMA}
onnx_out = sess.run(None, onnx_in)
onnx_enc = np.array(onnx_out[0])

# ── RKNN 추론 ──────────────────────────────────────────────
from rknnlite.api import RKNNLite
m = RKNNLite(verbose=False)
m.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
m.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

rknn_inputs = []
for name, shape, _ in ENC_SCHEMA:
    arr = state[name]
    if len(shape) == 4:
        arr = nchw2nhwc(arr)
    rknn_inputs.append(arr)

rknn_out = m.inference(inputs=rknn_inputs)
rknn_enc = np.array(rknn_out[0])
m.release()

# ── 결과 비교 ──────────────────────────────────────────────
print("=== encoder_out 비교 ===")
print(f"ONNX shape: {onnx_enc.shape}")
print(f"RKNN shape: {rknn_enc.shape}")
diff = np.abs(onnx_enc.astype(np.float32) - rknn_enc.astype(np.float32))
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"ONNX[:5]: {onnx_enc.flat[:5]}")
print(f"RKNN[:5]: {rknn_enc.flat[:5]}")

print("\n=== cached_len 비교 ===")
for i in range(5):
    onnx_cl = np.array(onnx_out[i+1])
    rknn_cl = np.array(rknn_out[i+1])
    print(f"  cached_len_{i}: ONNX={onnx_cl.flat[:4].tolist()} RKNN={rknn_cl.flat[:4].tolist()}")

print("\n=== cached_avg_0 비교 ===")
onnx_avg = np.array(onnx_out[6])
rknn_avg = np.array(rknn_out[6])
print(f"  ONNX shape={onnx_avg.shape}, max_diff_from_zero={np.abs(onnx_avg).max():.6f}")
print(f"  RKNN shape={rknn_avg.shape}, max_diff_from_zero={np.abs(rknn_avg).max():.6f}")
diff_avg = np.abs(onnx_avg - rknn_avg)
print(f"  ONNX vs RKNN max_diff: {diff_avg.max():.6f}")

print("\n=== cached_conv1_0 비교 ===")
onnx_cv = np.array(onnx_out[26])
rknn_cv = np.array(rknn_out[26])
print(f"  ONNX shape={onnx_cv.shape}")
print(f"  RKNN shape={rknn_cv.shape}")
diff_cv = np.abs(onnx_cv.astype(np.float32) - rknn_cv.astype(np.float32))
print(f"  Max diff: {diff_cv.max():.6f}")

print("\n=== GREEDY CHECK: argmax(joiner) ===")
dec_s = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
joi_s = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.onnx',  providers=['CPUExecutionProvider'])
y0 = np.array([[0, 0]], dtype=np.int64)
dec = dec_s.run(None, {'y': y0})[0]

for t in range(min(onnx_enc.shape[1], rknn_enc.shape[1])):
    o_enc = onnx_enc[0, t:t+1].astype(np.float32)
    r_enc = rknn_enc[0, t:t+1].astype(np.float32)
    o_joi = joi_s.run(None, {'encoder_out': o_enc, 'decoder_out': dec.reshape(1,512).astype(np.float32)})[0]
    r_joi = joi_s.run(None, {'encoder_out': r_enc, 'decoder_out': dec.reshape(1,512).astype(np.float32)})[0]
    o_y = int(np.argmax(o_joi))
    r_y = int(np.argmax(r_joi))
    print(f"  t={t}: ONNX_argmax={o_y}, RKNN_argmax={r_y}  {'match' if o_y==r_y else 'MISMATCH'}")
