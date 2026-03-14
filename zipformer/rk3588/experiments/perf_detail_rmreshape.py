"""
Run perf_detail on rmreshape model to identify per-layer bottlenecks.
Uses rknnlite with perf_detail=True.
"""
import numpy as np, sys, time
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588/encoder-epoch-99-avg-1-int8-cumfix-rmreshape.rknn'

from rknnlite.api import RKNNLite

rknn = RKNNLite(verbose=False)
ret = rknn.load_rknn(MODEL)
assert ret == 0

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0, perf_debug=True)
assert ret == 0

# Build 4D inputs for rmreshape model
# We need to query the model's expected shapes
from convert_encoder_int8_optarget import ENC_SCHEMA

# rmreshape model expects NHWC 4D inputs
# From encoder_capi.py, the shapes are queried at runtime
# For perf_detail, we just need to provide correct-shaped inputs

# Use the C API to get the actual shapes
from encoder_capi import EncoderCAPI
enc = EncoderCAPI(MODEL, core_mask=1)
shapes = enc._in_shapes
dtypes = enc._in_dtypes
cache_names = enc._cache_names
enc.release()

print(f"Input shapes ({len(shapes)}):")
for i, (s, d) in enumerate(zip(shapes, dtypes)):
    name = 'x' if i == 0 else cache_names[i-1]
    print(f"  [{i}] {name}: {s} ({d})")

# Create inputs
inputs = []
for s, d in zip(shapes, dtypes):
    inputs.append(np.zeros(s, dtype=d))

# Run perf detail
print("\nRunning perf_detail...")
perf = rknn.eval_perf(inputs=inputs, is_print=True)

rknn.release()
