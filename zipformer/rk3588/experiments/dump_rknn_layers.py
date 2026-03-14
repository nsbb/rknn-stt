"""
Build with verbose=True to capture RKNN internal layer names.
Then extract Reshape/Transpose layer IDs for op_target.
"""
import sys, os, io, re
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from convert_encoder_int8_optarget import INPUT_NAMES, INPUT_SHAPES

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'

from rknn.api import RKNN

# Redirect stdout/stderr to capture verbose output
import contextlib

log_file = '/tmp/rknn_verbose_build.log'

print(f"Building with verbose=True, capturing to {log_file}")

# Use a simple FP16 build (fast, no calibration needed)
rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')
ret = rknn.load_onnx(model=SIM_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
print(f"load_onnx: {ret}")
if ret != 0:
    sys.exit(1)

# Redirect verbose output
old_stdout = sys.stdout
old_stderr = sys.stderr
with open(log_file, 'w') as f:
    sys.stdout = f
    sys.stderr = f
    ret = rknn.build(do_quantization=False)
    sys.stdout = old_stdout
    sys.stderr = old_stderr

print(f"build: {ret}")
rknn.release()

# Parse log for layer info
print(f"\nParsing {log_file}...")
with open(log_file) as f:
    log = f.read()

print(f"Log size: {len(log)} bytes")

# Look for patterns that might indicate layer names
# Common patterns: "Layer N:", "Op:", "node_name"
patterns = [
    r'[Ll]ayer\s*[\[#]?\d+',
    r'[Rr]eshape',
    r'[Tt]ranspose',
    r'op_target',
    r'node.*name',
]

for pat in patterns:
    matches = re.findall(pat, log)
    if matches:
        print(f"Pattern '{pat}': {len(matches)} matches")
        for m in matches[:3]:
            print(f"  {m}")

# Extract lines with Reshape/Transpose
print("\nLines containing Reshape/Transpose:")
count = 0
for line in log.split('\n'):
    if 'Reshape' in line or 'Transpose' in line:
        count += 1
        if count <= 10:
            print(f"  {line.strip()[:120]}")
if count > 10:
    print(f"  ... ({count} total lines)")

# Show first 50 lines and last 50 lines of log
print("\n=== First 30 lines ===")
lines = log.split('\n')
for l in lines[:30]:
    print(f"  {l[:150]}")

print(f"\n=== Last 30 lines (of {len(lines)}) ===")
for l in lines[-30:]:
    print(f"  {l[:150]}")
