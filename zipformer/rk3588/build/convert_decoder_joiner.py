"""
decoder/joiner RKNN 변환 스크립트
- decoder: y[1,2] -> decoder_out[1,512]
- joiner: encoder_out[1,512] + decoder_out[1,512] -> logit[1,5000]
- 두 모델 모두 단순 ops (Gemm, Conv 등), 변환 성공 기대
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
OUT  = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588'

def convert_model(onnx_path, rknn_path, input_names, input_shapes):
    from rknn.api import RKNN
    print(f"\n{'='*60}")
    print(f"Converting: {os.path.basename(onnx_path)}")
    print(f"  inputs: {list(zip(input_names, input_shapes))}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=True)
    rknn.config(target_platform='rk3588')

    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=input_names,
        input_size_list=input_shapes,
    )
    if ret != 0:
        print(f"load_onnx failed: {ret}")
        return False

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"build failed: {ret}")
        return False

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn failed: {ret}")
        return False

    print(f"\nExported: {rknn_path}")

    # 빠른 sanity check (simulator)
    rknn.init_runtime()
    dummy_inputs = []
    for shape in input_shapes:
        if shape == [1, 2]:  # decoder y (int64)
            dummy_inputs.append(np.zeros(shape, dtype=np.int64))
        else:
            dummy_inputs.append(np.zeros(shape, dtype=np.float32))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_map = {}
    for nm, shape in zip(input_names, input_shapes):
        if nm == 'y':
            input_map[nm] = np.zeros(shape, dtype=np.int64)
        else:
            input_map[nm] = np.zeros(shape, dtype=np.float32)

    onnx_out = sess.run(None, input_map)
    rknn_out = rknn.inference(inputs=dummy_inputs)

    max_diff = np.abs(np.array(onnx_out[0], dtype=np.float32) - np.array(rknn_out[0], dtype=np.float32)).max()
    print(f"\n[Validation] ONNX vs RKNN(sim) max_diff: {max_diff:.8f}")
    if max_diff < 1e-3:
        print("[OK] Simulation output matches ONNX")
    else:
        print("[WARN] Large diff detected — check model")

    rknn.release()
    return True

if __name__ == '__main__':
    # decoder: y[1,2] -> decoder_out[1,512]
    ok = convert_model(
        onnx_path=f'{BASE}/decoder-epoch-99-avg-1.onnx',
        rknn_path=f'{OUT}/decoder-epoch-99-avg-1.rknn',
        input_names=['y'],
        input_shapes=[[1, 2]],
    )
    print(f"decoder: {'OK' if ok else 'FAILED'}")

    # joiner: encoder_out[1,512] + decoder_out[1,512] -> logit[1,5000]
    ok = convert_model(
        onnx_path=f'{BASE}/joiner-epoch-99-avg-1.onnx',
        rknn_path=f'{OUT}/joiner-epoch-99-avg-1.rknn',
        input_names=['encoder_out', 'decoder_out'],
        input_shapes=[[1, 512], [1, 512]],
    )
    print(f"joiner: {'OK' if ok else 'FAILED'}")
