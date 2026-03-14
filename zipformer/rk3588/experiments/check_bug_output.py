"""
float64 cached_len으로 RKNN 출력이 진짜 계산되는지 확인
"""
import numpy as np, time, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from rknnlite.api import RKNNLite
BASE='/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR=f'{BASE}/rk3588'
ENC_SCHEMA=[
    ('x',[1,39,80],'float32'),
    ('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]

def nchw2nhwc(a): return np.transpose(a,(0,2,3,1))

enc_r = RKNNLite(verbose=False)
enc_r.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

def make_buggy(x_val):
    inputs = []
    for nm, sh, dt in ENC_SCHEMA:
        if nm == 'x':
            a = np.ones((1,39,80), dtype=np.float32) * x_val
        else:
            a = np.random.randn(*sh).astype(np.dtype(dt)) * 0.1  # int64 → float64 bug
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inputs.append(a)
    return inputs

# Call with different x values
inp1 = make_buggy(0.5)
inp2 = make_buggy(-0.5)

out1 = enc_r.inference(inputs=inp1)
out2 = enc_r.inference(inputs=inp2)

e1 = np.array(out1[0]).astype(np.float32)
e2 = np.array(out2[0]).astype(np.float32)

print(f'Input x=+0.5: enc_out mean={e1.mean():.4f}')
print(f'Input x=-0.5: enc_out mean={e2.mean():.4f}')
print(f'Max diff:     {np.abs(e1-e2).max():.4f}')
print(f'Outputs vary: {not np.allclose(e1, e2)}')

# Time it
times = []
for _ in range(20):
    t0 = time.perf_counter()
    enc_r.inference(inputs=inp1)
    times.append((time.perf_counter()-t0)*1000)
print(f'Timing (buggy): median={np.median(times):.2f}ms')

enc_r.release()
