"""
CumSum 패치된 encoder → RKNN FP16 변환
encoder-epoch-99-avg-1-cumfix.onnx → encoder-epoch-99-avg-1-cumfix.rknn
"""
import numpy as np
from rknn.api import RKNN

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
OUT  = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

ENCODER_INPUTS = {
    'x':            [1, 39, 80],
    'cached_len_0': [2, 1],
    'cached_len_1': [4, 1],
    'cached_len_2': [3, 1],
    'cached_len_3': [2, 1],
    'cached_len_4': [4, 1],
    'cached_avg_0': [2, 1, 384],
    'cached_avg_1': [4, 1, 384],
    'cached_avg_2': [3, 1, 384],
    'cached_avg_3': [2, 1, 384],
    'cached_avg_4': [4, 1, 384],
    'cached_key_0': [2, 64, 1, 192],
    'cached_key_1': [4, 32, 1, 192],
    'cached_key_2': [3, 16, 1, 192],
    'cached_key_3': [2,  8, 1, 192],
    'cached_key_4': [4, 32, 1, 192],
    'cached_val_0': [2, 64, 1, 96],
    'cached_val_1': [4, 32, 1, 96],
    'cached_val_2': [3, 16, 1, 96],
    'cached_val_3': [2,  8, 1, 96],
    'cached_val_4': [4, 32, 1, 96],
    'cached_val2_0': [2, 64, 1, 96],
    'cached_val2_1': [4, 32, 1, 96],
    'cached_val2_2': [3, 16, 1, 96],
    'cached_val2_3': [2,  8, 1, 96],
    'cached_val2_4': [4, 32, 1, 96],
    'cached_conv1_0': [2, 1, 384, 30],
    'cached_conv1_1': [4, 1, 384, 30],
    'cached_conv1_2': [3, 1, 384, 30],
    'cached_conv1_3': [2, 1, 384, 30],
    'cached_conv1_4': [4, 1, 384, 30],
    'cached_conv2_0': [2, 1, 384, 30],
    'cached_conv2_1': [4, 1, 384, 30],
    'cached_conv2_2': [3, 1, 384, 30],
    'cached_conv2_3': [2, 1, 384, 30],
    'cached_conv2_4': [4, 1, 384, 30],
}

def make_dummy(name, shape):
    if 'cached_len' in name:
        return np.zeros(shape, dtype=np.int64)
    return np.zeros(shape, dtype=np.float32)

input_names  = list(ENCODER_INPUTS.keys())
input_shapes = list(ENCODER_INPUTS.values())

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')

print("==> load_onnx (cumfix) ...")
ret = rknn.load_onnx(
    model=f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx',
    inputs=input_names,
    input_size_list=input_shapes,
)
assert ret == 0, f"load_onnx failed: {ret}"

print("==> build (FP16) ...")
ret = rknn.build(do_quantization=False)
assert ret == 0, f"build failed: {ret}"

rknn_path = f'{OUT}/encoder-epoch-99-avg-1-cumfix.rknn'
ret = rknn.export_rknn(rknn_path)
assert ret == 0, f"export failed: {ret}"
print(f"Exported: {rknn_path}")

print("==> Simulator validation ...")
rknn.init_runtime()
import onnxruntime as ort
sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx', providers=['CPUExecutionProvider'])

dummy_dict = {nm: make_dummy(nm, sh) for nm, sh in ENCODER_INPUTS.items()}
def to_rknn(nm, sh):
    a = make_dummy(nm, sh)
    if len(sh) == 4:
        a = np.ascontiguousarray(np.transpose(a, (0,2,3,1)))
    return a

dummy_list = [to_rknn(nm, sh) for nm, sh in ENCODER_INPUTS.items()]

onnx_out = sess.run(None, dummy_dict)
rknn_out  = rknn.inference(inputs=dummy_list)

diff = np.abs(np.array(onnx_out[0], dtype=np.float32) - np.array(rknn_out[0], dtype=np.float32)).max()
print(f"encoder_out max_diff (ONNX vs sim): {diff:.6f}")

rnd_list = []
for nm, sh in ENCODER_INPUTS.items():
    a = np.random.randn(*sh).astype(np.float32 if 'len' not in nm else np.int64)
    if len(sh) == 4:
        a = np.ascontiguousarray(np.transpose(a, (0,2,3,1)))
    rnd_list.append(a)
rknn_out2 = rknn.inference(inputs=rnd_list)
const = np.allclose(rknn_out[0], rknn_out2[0])
print(f"상수 출력 확인 zeros==random: {const} ({'문제!' if const else 'OK'})")

rknn.release()
print("Done.")
