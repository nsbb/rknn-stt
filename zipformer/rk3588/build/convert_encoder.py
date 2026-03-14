"""
Encoder RKNN 변환 (1차 시도 - 직접 변환)
- encoder: x[1,39,80] + 40개의 캐시 입력 -> encoder_out + 40개의 새 캐시 출력
- 문제 예상: CumSum, Range, ConstantOfShape, Where, ReduceMean (RKNN 미지원 가능)
"""

import numpy as np
import os

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
OUT  = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

# 모든 encoder 입력 이름과 고정 shape 정의
# '?' 차원은 batch=1로 고정
ENCODER_INPUTS = {
    'x':            [1, 39, 80],    # float32
    'cached_len_0': [2, 1],         # int64
    'cached_len_1': [4, 1],         # int64
    'cached_len_2': [3, 1],         # int64
    'cached_len_3': [2, 1],         # int64
    'cached_len_4': [4, 1],         # int64
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

def make_dummy_input(name, shape):
    if 'cached_len' in name:
        return np.zeros(shape, dtype=np.int64)
    return np.zeros(shape, dtype=np.float32)

if __name__ == '__main__':
    from rknn.api import RKNN

    input_names  = list(ENCODER_INPUTS.keys())
    input_shapes = list(ENCODER_INPUTS.values())
    print(f"Encoder inputs: {len(input_names)}")

    rknn = RKNN(verbose=True)
    rknn.config(target_platform='rk3588')

    print("\n==> load_onnx ...")
    ret = rknn.load_onnx(
        model=f'{BASE}/encoder-epoch-99-avg-1.onnx',
        inputs=input_names,
        input_size_list=input_shapes,
    )
    if ret != 0:
        print(f"load_onnx failed: {ret}")
        exit(1)
    print("load_onnx OK")

    print("\n==> build (do_quantization=False) ...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"build failed: {ret}")
        exit(1)
    print("build OK")

    rknn_path = f'{OUT}/encoder-epoch-99-avg-1.rknn'
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn failed: {ret}")
        exit(1)
    print(f"Exported: {rknn_path}")

    # Simulator validation
    print("\n==> Simulator validation ...")
    rknn.init_runtime()
    dummy = [make_dummy_input(nm, sh) for nm, sh in ENCODER_INPUTS.items()]

    import onnxruntime as ort
    sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
    input_dict = {nm: make_dummy_input(nm, sh) for nm, sh in ENCODER_INPUTS.items()}
    onnx_out = sess.run(None, input_dict)
    rknn_out = rknn.inference(inputs=dummy)

    max_diff = np.abs(np.array(onnx_out[0], dtype=np.float32) - np.array(rknn_out[0], dtype=np.float32)).max()
    print(f"\n[Validation] encoder_out ONNX vs RKNN(sim) max_diff: {max_diff:.8f}")
    if max_diff < 0.1:
        print("[OK] Simulation close to ONNX")
    else:
        print("[WARN] Large diff — likely unsupported ops falling back to CPU or failing")

    # 상수 출력 확인
    rnd_dummy = [np.random.randn(*sh).astype(np.float32 if 'len' not in nm else np.int64)
                 for nm, sh in ENCODER_INPUTS.items()]
    rknn_out2 = rknn.inference(inputs=rnd_dummy)
    const_check = np.allclose(rknn_out[0], rknn_out2[0])
    print(f"[상수 출력 확인] zeros==random: {const_check} ({'문제!' if const_check else 'OK (입력에 반응함)'})")

    rknn.release()
    print("\nDone.")
