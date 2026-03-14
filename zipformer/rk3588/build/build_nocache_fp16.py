"""
Build nocache RKNN without quantization (FP16).
No quantization = no quant/dequant layers = potentially fewer RKNN layers.
"""
import numpy as np, os, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from rknn.api import RKNN

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
NOCACHE_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
DST = f'{RKNN_DIR}/encoder-fp16-cumfix-nocache.rknn'

ENC_SCHEMA = [
    ('x',              [1, 39, 80],        'float32'),
    ('cached_len_0',   [2, 1],             'int64'),  ('cached_len_1',   [4, 1],             'int64'),
    ('cached_len_2',   [3, 1],             'int64'),  ('cached_len_3',   [2, 1],             'int64'),
    ('cached_len_4',   [4, 1],             'int64'),
    ('cached_avg_0',   [2, 1, 384],        'float32'),('cached_avg_1',   [4, 1, 384],        'float32'),
    ('cached_avg_2',   [3, 1, 384],        'float32'),('cached_avg_3',   [2, 1, 384],        'float32'),
    ('cached_avg_4',   [4, 1, 384],        'float32'),
    ('cached_key_0',   [2, 64, 1, 192],    'float32'),('cached_key_1',   [4, 32, 1, 192],    'float32'),
    ('cached_key_2',   [3, 16, 1, 192],    'float32'),('cached_key_3',   [2,  8, 1, 192],    'float32'),
    ('cached_key_4',   [4, 32, 1, 192],    'float32'),
    ('cached_val_0',   [2, 64, 1, 96],     'float32'),('cached_val_1',   [4, 32, 1, 96],     'float32'),
    ('cached_val_2',   [3, 16, 1, 96],     'float32'),('cached_val_3',   [2,  8, 1, 96],     'float32'),
    ('cached_val_4',   [4, 32, 1, 96],     'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],     'float32'),('cached_val2_1',  [4, 32, 1, 96],     'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],     'float32'),('cached_val2_3',  [2,  8, 1, 96],     'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],     'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],    'float32'),('cached_conv1_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],    'float32'),('cached_conv1_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],    'float32'),('cached_conv2_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],    'float32'),('cached_conv2_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],    'float32'),
]
INPUT_NAMES  = [s[0] for s in ENC_SCHEMA]
INPUT_SHAPES = [s[1] for s in ENC_SCHEMA]

if __name__ == '__main__':
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588', remove_reshape=True, optimization_level=3)
    print(f"Loading: {NOCACHE_ONNX}")
    ret = rknn.load_onnx(model=NOCACHE_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    if ret != 0:
        print(f"Failed: {ret}"); sys.exit(1)
    print("Building FP16 (no quantization)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"Build FAILED: {ret}"); sys.exit(1)
    rknn.export_rknn(DST)
    print(f"Saved: {DST} ({os.path.getsize(DST)/1024/1024:.1f} MB)")
    rknn.release()
