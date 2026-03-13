"""
시도 6: FP16 빌드 — INT8의 exDataConvert/Cast 레이어 제거로 총 레이어 수 감소 가능성.
INT8: ~1619 layers, 31ms dispatch. FP16이면 DataConvert 제거 → 더 적은 레이어?
"""
import sys, os
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES, ENC_SCHEMA

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
RKNN_DIR = f'{BASE}/rk3588'

if __name__ == '__main__':
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform='rk3588',
        optimization_level=3,
        remove_reshape=True,
    )

    print(f"Loading ONNX: {SIM_ONNX}")
    ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    assert ret == 0, f"load_onnx failed: {ret}"

    print("Building FP16 (no quantization, remove_reshape=True)...")
    ret = rknn.build(do_quantization=False)
    assert ret == 0, f"build failed: {ret}"

    out_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1-fp16-cumfix-rmreshape.rknn'
    ret = rknn.export_rknn(out_path)
    assert ret == 0, f"export failed: {ret}"
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported: {out_path} ({sz:.1f} MB)")

    rknn.release()
    print("Done!")
