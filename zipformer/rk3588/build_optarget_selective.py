"""
시도 7: 선택적 op_target — 소수의 안전한 레이어만 CPU로 이동.
점진적 테스트: 10개 → 50개 → 100개 → 전체.
크래시 발생 시 가장 많이 이동 가능한 수를 찾는다.
"""
import sys, os
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
RKNN_DIR = f'{BASE}/rk3588'

MAP_FILE = '/tmp/rknn_acc/map_name_to_file.txt'

def get_safe_layers():
    """Get layers with _rs or _tp suffix that are NOT fused with compute ops."""
    fused_markers = ['_mm', '_sw', '_gl', '_sf', '_nm']
    layers = []
    with open(MAP_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == 'layer_name':
                continue
            name = parts[0]
            if name.startswith('---'):
                continue
            is_fused = any(m in name for m in fused_markers)
            if not is_fused and (name.endswith('_rs') or name.endswith('_tp')):
                layers.append(name)
    return layers

def build_with_n_cpu_layers(layers, n, suffix):
    from rknn.api import RKNN
    selected = layers[:n]
    op_target = {name: 'cpu' for name in selected}
    print(f"\n{'='*60}")
    print(f"Building with {n} layers on CPU (suffix: {suffix})")
    print(f"Sample: {selected[:5]}")

    calib_data = make_calib_data(30)
    dataset_txt = save_calib_npy(calib_data, f'/tmp/calib_opt_{suffix}')

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform='rk3588',
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
        remove_reshape=True,
        op_target=op_target,
    )

    ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    if ret != 0:
        print(f"  load_onnx failed: {ret}")
        rknn.release()
        return None

    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print(f"  build failed: {ret}")
        rknn.release()
        return None

    out_path = f'{RKNN_DIR}/encoder-int8-cumfix-rmreshape-opt{suffix}.rknn'
    ret = rknn.export_rknn(out_path)
    if ret != 0:
        print(f"  export failed: {ret}")
        rknn.release()
        return None

    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Exported: {out_path} ({sz:.1f} MB)")
    rknn.release()
    return out_path

if __name__ == '__main__':
    safe = get_safe_layers()
    print(f"Total safe layers: {len(safe)}")

    # Try with increasing counts
    for n in [10, 50]:
        if n > len(safe):
            n = len(safe)
        path = build_with_n_cpu_layers(safe, n, f'{n}cpu')
        if path is None:
            print(f"  Build FAILED for n={n}")
            break

    print("\nAll builds done!")
