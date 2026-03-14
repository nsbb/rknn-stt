"""
Build rmreshape model with op_target: force Reshape/Transpose to CPU.
Try different node name formats to find valid one for RKNN.
"""
import numpy as np, os, sys, onnx
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import ENC_SCHEMA, INPUT_NAMES, INPUT_SHAPES, make_calib_data, save_calib_npy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
RKNN_DIR = f'{BASE}/rk3588'

def get_reshape_transpose_names(onnx_path):
    m = onnx.load(onnx_path)
    names = []
    for n in m.graph.node:
        if n.op_type in ('Reshape', 'Transpose'):
            names.append(n.name)
    return names

if __name__ == '__main__':
    from rknn.api import RKNN

    names = get_reshape_transpose_names(SIM_ONNX)
    print(f"Reshape/Transpose nodes: {len(names)}")

    # Try different name formats
    formats = [
        ('as-is', lambda n: n),
        ('no-slash', lambda n: n.lstrip('/')),
        ('with-output', lambda n: n + '_output_0'),
    ]

    for fmt_name, fmt_fn in formats:
        print(f"\n{'='*60}")
        print(f"Trying format: {fmt_name}")

        # Test with just 1 node first
        test_name = fmt_fn(names[0])
        print(f"  Test key: '{test_name}'")

        rknn = RKNN(verbose=False)
        rknn.config(
            target_platform='rk3588',
            quantized_dtype='asymmetric_quantized-8',
            optimization_level=3,
            remove_reshape=True,
            op_target={test_name: 'cpu'},
        )

        ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
        if ret != 0:
            print(f"  load_onnx FAILED: {ret}")
            rknn.release()
            continue

        print(f"  load_onnx OK! Building with 1 node on CPU...")
        ret = rknn.build(do_quantization=False)  # Skip quantization for quick test
        if ret != 0:
            print(f"  build FAILED: {ret}")
        else:
            print(f"  build OK!")
            # Now try with ALL reshape/transpose nodes
            rknn.release()

            print(f"\n  Building with ALL {len(names)} nodes on CPU...")
            calib_data = make_calib_data(30)
            dataset_txt = save_calib_npy(calib_data, '/tmp/calib_optarget')

            op_target = {fmt_fn(n): 'cpu' for n in names}
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
                print(f"  load_onnx FAILED: {ret}")
                rknn.release()
                continue

            ret = rknn.build(do_quantization=True, dataset=dataset_txt)
            if ret != 0:
                print(f"  full build FAILED: {ret}")
            else:
                out_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8-cumfix-rmreshape-optarget.rknn'
                rknn.export_rknn(out_path)
                print(f"  Exported: {out_path}")
                print(f"  Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")

        rknn.release()
        break  # Stop after first successful format

    print("\nDone!")
