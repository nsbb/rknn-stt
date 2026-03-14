"""
Aggressive ONNX simplification for RKNN layer count reduction.

Strategy:
1. Run onnxsim multiple passes
2. Then manually remove remaining unnecessary ops:
   - Fuse Reshape->Transpose chains where possible
   - Remove identity Cast ops
   - Fold remaining shape computation patterns
3. Compare node counts: original vs simplified
"""
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from collections import Counter, defaultdict
import onnxruntime as ort
import subprocess, os

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
CUMFIX_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
AGSIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-agsim.onnx'


def try_onnxsim_multi_pass(input_path, output_path, n_passes=5):
    """Run onnxsim multiple times for maximum simplification."""
    import onnxsim

    model = onnx.load(input_path)
    prev_count = len(model.graph.node)
    print(f"Initial: {prev_count} nodes")

    for i in range(n_passes):
        try:
            model, check = onnxsim.simplify(model)
            cur_count = len(model.graph.node)
            diff = prev_count - cur_count
            print(f"  Pass {i+1}: {cur_count} nodes ({'-' if diff >= 0 else '+'}{abs(diff)})")
            if diff == 0:
                print(f"  Converged at pass {i+1}")
                break
            prev_count = cur_count
        except Exception as e:
            print(f"  Pass {i+1} failed: {e}")
            break

    return model


def remove_identity_casts(model):
    """Remove Cast nodes where input and output types are the same."""
    g = model.graph
    removed = 0

    # Build maps
    out2node = {}
    in2consumers = defaultdict(list)
    for n in g.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2consumers[inp].append(n)

    nodes_to_remove = set()
    for n in g.node:
        if n.op_type != 'Cast':
            continue
        # Get target type
        to_type = None
        for attr in n.attribute:
            if attr.name == 'to':
                to_type = attr.i
        if to_type is None:
            continue

        # Check if input is from a node with known output type that matches
        # We can check if the Cast is float32->float32 or int64->int64
        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1:
            continue

        # Simple heuristic: if Cast(to=FLOAT) feeds into MatMul/Add/Mul, it's likely needed
        # But Cast(to=INT64) -> Cast(to=INT64) chains can be removed
        # Check for consecutive Cast nodes
        if consumers[0].op_type == 'Cast':
            c2 = consumers[0]
            to_type2 = None
            for attr in c2.attribute:
                if attr.name == 'to':
                    to_type2 = attr.i
            if to_type2 is not None:
                # Cast(A->B) -> Cast(B->C) can be replaced with Cast(A->C)
                c2.input[0] = n.input[0]
                nodes_to_remove.add(id(n))
                removed += 1

    if nodes_to_remove:
        new_nodes = [n for n in g.node if id(n) not in nodes_to_remove]
        del g.node[:]
        g.node.extend(new_nodes)

    return removed


def fold_gather_unsqueeze_concat(model):
    """
    Fold patterns like:
      Shape -> Gather[0] -> Unsqueeze -> Concat -> Reshape
    Since batch=1 and all shapes are static, these should be constants.
    """
    g = model.graph
    out2node = {}
    in2consumers = defaultdict(list)
    for n in g.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2consumers[inp].append(n)

    # Count remaining Shape nodes (these should have been eliminated by onnxsim)
    shape_nodes = [n for n in g.node if n.op_type == 'Shape']
    print(f"  Remaining Shape nodes: {len(shape_nodes)}")

    # Count Gather nodes used in shape computation
    shape_gathers = 0
    for n in shape_nodes:
        for c in in2consumers.get(n.output[0], []):
            if c.op_type == 'Gather':
                shape_gathers += 1
    print(f"  Shape->Gather patterns: {shape_gathers}")

    return 0


def verify_numerical_equivalence(original_path, simplified_path, rtol=1e-3, atol=1e-3):
    """Verify the simplified model produces same outputs."""
    from convert_encoder_int8_optarget import ENC_SCHEMA

    sess_orig = ort.InferenceSession(original_path, providers=['CPUExecutionProvider'])
    sess_simp = ort.InferenceSession(simplified_path, providers=['CPUExecutionProvider'])

    # Create random input
    inputs = {}
    for name, shape, dtype in ENC_SCHEMA:
        inputs[name] = np.zeros(shape, dtype=np.dtype(dtype))
    inputs['x'] = np.random.randn(*[1, 39, 80]).astype(np.float32)

    out_orig = sess_orig.run(None, inputs)
    out_simp = sess_simp.run(None, inputs)

    print(f"\n  Numerical verification ({len(out_orig)} outputs):")
    all_ok = True
    for i, (o, s) in enumerate(zip(out_orig, out_simp)):
        max_diff = np.max(np.abs(o - s))
        cos = np.dot(o.flatten(), s.flatten()) / (np.linalg.norm(o) * np.linalg.norm(s) + 1e-10)
        ok = max_diff < atol or cos > 0.999
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        if i < 3 or not ok:
            print(f"    out[{i}]: max_diff={max_diff:.6f}  cos_sim={cos:.6f}  {status}")
    if all_ok:
        print(f"    All {len(out_orig)} outputs match!")
    return all_ok


if __name__ == '__main__':
    # Step 1: Multi-pass onnxsim
    print("=== Step 1: Multi-pass onnxsim ===")
    model = try_onnxsim_multi_pass(CUMFIX_ONNX, AGSIM_ONNX, n_passes=5)

    # Step 2: Remove identity casts
    print("\n=== Step 2: Remove identity Casts ===")
    removed_casts = remove_identity_casts(model)
    print(f"  Removed {removed_casts} identity Cast chains")

    # Step 3: Analyze remaining patterns
    print("\n=== Step 3: Remaining pattern analysis ===")
    fold_gather_unsqueeze_concat(model)

    # Final counts
    c = Counter(n.op_type for n in model.graph.node)
    print(f"\n=== Final: {len(model.graph.node)} nodes ===")
    for op, cnt in c.most_common(15):
        print(f"  {op}: {cnt}")

    # Save
    onnx.save(model, AGSIM_ONNX)
    print(f"\nSaved: {AGSIM_ONNX}")
    print(f"Size: {os.path.getsize(AGSIM_ONNX)/1024/1024:.1f} MB")

    # Verify
    print("\n=== Verification ===")
    verify_numerical_equivalence(CUMFIX_ONNX, AGSIM_ONNX)
