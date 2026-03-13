"""
Build model with ALL Reshape/Transpose layers forced to CPU via op_target.
Uses RKNN internal layer names from accuracy_analysis output.
"""
import sys, os
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
RKNN_DIR = f'{BASE}/rk3588'

# Parse RKNN internal layer names for Reshape/Transpose
MAP_FILE = '/tmp/rknn_acc/map_name_to_file.txt'

def get_reshape_transpose_layers():
    layers = []
    with open(MAP_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[0]
            name_lower = name.lower()
            if 'reshape' in name_lower or 'transpose' in name_lower:
                layers.append(name)
    return layers

if __name__ == '__main__':
    from rknn.api import RKNN

    layers = get_reshape_transpose_layers()
    print(f"Reshape/Transpose layers: {len(layers)}")
    print(f"Sample: {layers[:5]}")

    # Build op_target dict
    op_target = {name: 'cpu' for name in layers}

    print("\nCalibration data...")
    calib_data = make_calib_data(30)
    dataset_txt = save_calib_npy(calib_data, '/tmp/calib_optarget_all')

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform='rk3588',
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
        op_target=op_target,
    )

    print(f"\nLoading ONNX: {SIM_ONNX}")
    ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    assert ret == 0, f"load_onnx failed: {ret}"

    print("\nBuilding INT8 (Reshape/Transpose → CPU)...")
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    assert ret == 0, f"build failed: {ret}"

    out_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-optarget-cpu.rknn'
    ret = rknn.export_rknn(out_path)
    assert ret == 0, f"export failed: {ret}"
    print(f"\nExported: {out_path}")
    print(f"Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")

    rknn.release()
    print("Done!")
