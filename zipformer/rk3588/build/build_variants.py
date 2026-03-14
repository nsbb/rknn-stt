"""
Build 5 rmreshape model variants with different RKNN configs.
Goal: find config that reduces rknn_run below 30ms (from current 30.7ms).
"""
import numpy as np, os, sys, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import ENC_SCHEMA, INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
CUMFIX_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx'
SIM_ONNX    = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'

CONFIGS = [
    ('rmreshape-opt2',        CUMFIX_ONNX, dict(optimization_level=2, remove_reshape=True)),
    ('rmreshape-singlecore',  CUMFIX_ONNX, dict(remove_reshape=True, single_core_mode=True)),
    ('rmreshape-flash',       CUMFIX_ONNX, dict(remove_reshape=True, enable_flash_attention=True)),
    ('rmreshape-sim',         SIM_ONNX,    dict(remove_reshape=True)),
]

if __name__ == '__main__':
    from rknn.api import RKNN

    print("Calibration data...")
    calib_data = make_calib_data(30)
    dataset_txt = save_calib_npy(calib_data, '/tmp/calib_enc_variants')

    for name, onnx_path, extra_cfg in CONFIGS:
        out_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-{name}.rknn'
        if os.path.exists(out_path):
            print(f"\n=== SKIP {name} (already exists) ===")
            continue

        print(f"\n{'='*60}")
        print(f"Building: {name}")
        print(f"  ONNX: {os.path.basename(onnx_path)}")
        print(f"  Config: {extra_cfg}")
        print(f"{'='*60}")

        rknn = RKNN(verbose=False)
        cfg = dict(
            target_platform='rk3588',
            quantized_dtype='asymmetric_quantized-8',
            optimization_level=3,
        )
        cfg.update(extra_cfg)
        rknn.config(**cfg)

        ret = rknn.load_onnx(model=onnx_path, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
        if ret != 0:
            print(f"  load_onnx FAILED: {ret}")
            rknn.release()
            continue

        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
        if ret != 0:
            print(f"  build FAILED: {ret}")
            rknn.release()
            continue

        ret = rknn.export_rknn(out_path)
        if ret != 0:
            print(f"  export FAILED: {ret}")
        else:
            print(f"  Exported: {out_path}")
            sz = os.path.getsize(out_path) / 1024 / 1024
            print(f"  Size: {sz:.1f} MB")

        rknn.release()

    print("\nAll done!")
