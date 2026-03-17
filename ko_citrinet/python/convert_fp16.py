"""CitriNet ONNX -> RKNN FP16 conversion."""
import sys
from rknn.api import RKNN

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else '../citrinet_npu_v2_fixlen.onnx'
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else '../model/citrinet_fp16.rknn'

rknn = RKNN(verbose=True)

rknn.config(target_platform='rk3588')

ret = rknn.load_onnx(model=ONNX_PATH)
if ret != 0:
    print(f'Load ONNX failed: {ret}')
    sys.exit(1)

ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f'Build failed: {ret}')
    sys.exit(1)

import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
ret = rknn.export_rknn(OUT_PATH)
if ret != 0:
    print(f'Export failed: {ret}')
    sys.exit(1)

print(f'Done: {OUT_PATH}')
rknn.release()
