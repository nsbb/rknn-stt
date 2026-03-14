"""
CumSum 패치된 RKNN encoder 검증
- 5 청크에 걸쳐 ONNX(원본)와 RKNN(cumfix) 출력 비교
- 청크 1+에서 cached_avg가 non-zero가 되어도 발산하지 않는지 확인
"""
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

SCHEMA = [
    ('x',[1,39,80],'float32'),
    ('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),
    ('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),
    ('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),
    ('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),
    ('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),
    ('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),
    ('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),
    ('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]
CACHE_NAMES = [s[0] for s in SCHEMA if s[0] != 'x']

def nchw2nhwc(a): return np.ascontiguousarray(np.transpose(a, (0,2,3,1)))

print("Loading models...")
sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])

enc = RKNNLite(verbose=False)
enc.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1-cumfix.rknn')
enc.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

so = {nm: np.zeros(sh, dtype=np.int64 if dt=='int64' else np.float32) for nm,sh,dt in SCHEMA}
sr = {nm: np.zeros(sh, dtype=np.int64 if dt=='int64' else np.float32) for nm,sh,dt in SCHEMA}

np.random.seed(42)
print("\n=== 5-chunk 추론 비교 ===")
print(f"{'Chunk':>5}  {'encoder_out diff':>16}  {'max_cached_avg':>14}  {'status':>8}")

all_pass = True
for ci in range(5):
    x = (np.random.randn(1, 39, 80) * 0.3).astype(np.float32)
    so['x'] = x; sr['x'] = x

    # ONNX (ground truth)
    onnx_out = sess.run(None, so)
    enc_o = onnx_out[0]
    for i, nm in enumerate(CACHE_NAMES):
        so[nm] = onnx_out[i+1]

    # RKNN cumfix
    rknn_in = []
    for nm, sh, _ in SCHEMA:
        a = sr[nm]
        if len(sh) == 4:
            a = nchw2nhwc(a)
        rknn_in.append(a)
    rknn_out = enc.inference(inputs=rknn_in)
    enc_r = np.array(rknn_out[0], dtype=np.float32)
    for i, nm in enumerate(CACHE_NAMES):
        a = np.array(rknn_out[i+1])
        if 'cached_len' in nm:
            a = a.astype(np.int64)
        sr[nm] = a

    diff = np.abs(enc_o.astype(np.float32) - enc_r).max()
    max_avg = max(np.abs(sr[f'cached_avg_{k}']).max() for k in range(5))
    ok = diff < 5.0  # FP16 허용 오차 (큰 값 누적 가능)
    all_pass = all_pass and ok
    print(f"  {ci:3d}    {diff:>14.4f}    {max_avg:>12.4f}    {'OK' if ok else 'FAIL'}")

print(f"\n{'결론: PASS' if all_pass else '결론: FAIL'}")

# cached_avg 발산 확인 (원본 RKNN 버그 재현 기준)
avg_diff = np.abs(
    so['cached_avg_0'].astype(np.float32) - sr['cached_avg_0'].astype(np.float32)
).max()
print(f"cached_avg_0 ONNX vs RKNN diff: {avg_diff:.4f}")
print("(원본 버그: chunk 1+에서 이 값이 수백~수천으로 발산했음)")

enc.release()
