"""
시도 11: 다른 양자화 전략 — 다른 quantized_dtype이 다른 RKNN 내부 그래프를 생성할 수 있음.
w8a16: 8-bit weights, 16-bit activations → exDataConvert 감소?
quantized_algorithm=mmse: 더 나은 양자화로 다른 fusion 가능?
"""
import sys, os, time
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
RKNN_DIR = f'{BASE}/rk3588'

VARIANTS = [
    ('w8a16', dict(quantized_dtype='w8a16')),
    ('mmse', dict(quantized_algorithm='mmse')),
    ('kl', dict(quantized_algorithm='kl_divergence')),
]

if __name__ == '__main__':
    from rknn.api import RKNN

    print("Generating calibration data...")
    calib_data = make_calib_data(30)

    for suffix, extra_opts in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Building: {suffix} ({extra_opts})")
        dataset_txt = save_calib_npy(calib_data, f'/tmp/calib_q_{suffix}')

        rknn = RKNN(verbose=False)
        base_config = dict(
            target_platform='rk3588',
            optimization_level=3,
            remove_reshape=True,
        )
        # Set defaults for quantized_dtype/algorithm if not overridden
        if 'quantized_dtype' not in extra_opts:
            base_config['quantized_dtype'] = 'asymmetric_quantized-8'
        base_config.update(extra_opts)
        rknn.config(**base_config)

        ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
        if ret != 0:
            print(f"  load_onnx failed: {ret}")
            rknn.release()
            continue

        t0 = time.time()
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
        dt = time.time() - t0
        if ret != 0:
            print(f"  build failed: {ret} (took {dt:.0f}s)")
            rknn.release()
            continue

        out_path = f'{RKNN_DIR}/encoder-int8-cumfix-rmreshape-{suffix}.rknn'
        rknn.export_rknn(out_path)
        sz = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Exported: {out_path} ({sz:.1f} MB, build {dt:.0f}s)")
        rknn.release()

    print("\nAll builds done!")
