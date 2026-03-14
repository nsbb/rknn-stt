"""
Fix dynamic shapes to static (batch=1) and fold shape ops.
Many Unsqueeze/Shape/Gather/Reshape ops exist only for dynamic batch handling.
With batch=1 fixed, these become constants and can be folded away.
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
from collections import Counter

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SRC = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
DST = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache-static.onnx'

# Fixed shapes with batch=1
ENC_SCHEMA = [
    ('x',              [1, 39, 80],        'float32'),
    ('cached_len_0',   [2, 1],             'int64'),  ('cached_len_1',   [4, 1],             'int64'),
    ('cached_len_2',   [3, 1],             'int64'),  ('cached_len_3',   [2, 1],             'int64'),
    ('cached_len_4',   [4, 1],             'int64'),
    ('cached_avg_0',   [2, 1, 384],        'float32'),('cached_avg_1',   [4, 1, 384],        'float32'),
    ('cached_avg_2',   [3, 1, 384],        'float32'),('cached_avg_3',   [2, 1, 384],        'float32'),
    ('cached_avg_4',   [4, 1, 384],        'float32'),
    ('cached_key_0',   [2, 64, 1, 192],    'float32'),('cached_key_1',   [4, 32, 1, 192],    'float32'),
    ('cached_key_2',   [3, 16, 1, 192],    'float32'),('cached_key_3',   [2,  8, 1, 192],    'float32'),
    ('cached_key_4',   [4, 32, 1, 192],    'float32'),
    ('cached_val_0',   [2, 64, 1, 96],     'float32'),('cached_val_1',   [4, 32, 1, 96],     'float32'),
    ('cached_val_2',   [3, 16, 1, 96],     'float32'),('cached_val_3',   [2,  8, 1, 96],     'float32'),
    ('cached_val_4',   [4, 32, 1, 96],     'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],     'float32'),('cached_val2_1',  [4, 32, 1, 96],     'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],     'float32'),('cached_val2_3',  [2,  8, 1, 96],     'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],     'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],    'float32'),('cached_conv1_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],    'float32'),('cached_conv1_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],    'float32'),('cached_conv2_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],    'float32'),('cached_conv2_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],    'float32'),
]
INPUT_SHAPES = {s[0]: s[1] for s in ENC_SCHEMA}


def fix_input_shapes(model):
    """Replace dynamic dims with fixed values."""
    for inp in model.graph.input:
        if inp.name in INPUT_SHAPES:
            shape = INPUT_SHAPES[inp.name]
            # Clear existing dims and set new ones
            while len(inp.type.tensor_type.shape.dim) > 0:
                inp.type.tensor_type.shape.dim.pop()
            for d in shape:
                dim = inp.type.tensor_type.shape.dim.add()
                dim.dim_value = d
    return model


def main():
    print("Loading nocache model...")
    m = onnx.load(SRC)

    ops_before = Counter(n.op_type for n in m.graph.node)
    print(f"Nodes before: {len(m.graph.node)}")

    # Fix input shapes
    m = fix_input_shapes(m)
    print("Fixed input shapes to static batch=1")

    # Save temp for onnxsim
    tmp = '/tmp/nocache_fixshape.onnx'
    onnx.save(m, tmp)

    # Run onnxsim with fixed shapes
    import onnxsim
    print("Running onnxsim with static shapes...")
    m_sim, check = onnxsim.simplify(
        tmp,
        overwrite_input_shapes={name: shape for name, shape in INPUT_SHAPES.items()},
    )
    if not check:
        print("WARNING: onnxsim check failed")

    ops_after = Counter(n.op_type for n in m_sim.graph.node)
    print(f"Nodes after: {len(m_sim.graph.node)}")

    # Show what was removed
    print("\nOp changes:")
    all_ops = sorted(set(list(ops_before.keys()) + list(ops_after.keys())))
    total_removed = 0
    for op in all_ops:
        b = ops_before.get(op, 0)
        a = ops_after.get(op, 0)
        if b != a:
            diff = a - b
            total_removed += (b - a)
            print(f"  {op:20s} {b:5d} → {a:5d} ({diff:+d})")
    print(f"\nTotal removed: {total_removed}")

    # Verify with ORT
    try:
        sess = ort.InferenceSession(onnx.SerializeToString(m_sim), providers=['CPUExecutionProvider'])
        print(f"\n✓ ORT: {len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs")
        # Quick inference test
        feeds = {}
        for inp in sess.get_inputs():
            shape = inp.shape
            dtype = np.float32 if 'float' in inp.type else np.int64
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
        outs = sess.run(None, feeds)
        print(f"  Output shapes: {[o.shape for o in outs[:3]]}...")
    except Exception as e:
        print(f"\n✗ ORT: {e}")

    onnx.save(m_sim, DST)
    import os
    print(f"\nSaved: {DST} ({os.path.getsize(DST)/1024/1024:.1f} MB)")


if __name__ == '__main__':
    main()
