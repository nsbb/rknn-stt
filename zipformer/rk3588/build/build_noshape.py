"""
시도 12: noshape ONNX (Shape/Gather/Unsqueeze 제거) → RKNN INT8 빌드.
2275 → 1875 노드 (-17.6%). RKNN 레이어 수 감소로 dispatch overhead 줄어드는지 확인.
"""
import sys, os
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
NOSHAPE_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-noshape.onnx'
RKNN_DIR = f'{BASE}/rk3588'

if __name__ == '__main__':
    from rknn.api import RKNN

    print("Generating calibration data...")
    calib_data = make_calib_data(30)
    dataset_txt = save_calib_npy(calib_data, '/tmp/calib_noshape')

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform='rk3588',
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
        remove_reshape=True,
    )

    print(f"Loading ONNX: {NOSHAPE_ONNX}")
    # Don't pass inputs/input_size_list since shapes are now static in the ONNX
    ret = rknn.load_onnx(model=NOSHAPE_ONNX)
    if ret != 0:
        print("Trying with explicit inputs...")
        ret = rknn.load_onnx(model=NOSHAPE_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    assert ret == 0, f"load_onnx failed: {ret}"

    print("Building INT8 (noshape, remove_reshape=True)...")
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    assert ret == 0, f"build failed: {ret}"

    out_path = f'{RKNN_DIR}/encoder-int8-cumfix-noshape-rmreshape.rknn'
    ret = rknn.export_rknn(out_path)
    assert ret == 0, f"export failed: {ret}"
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported: {out_path} ({sz:.1f} MB)")

    rknn.release()
    print("Done!")
