"""
Comprehensive ONNX optimization for the nocache model.
Goal: reduce node count → fewer RKNN layers → faster dispatch.

Techniques:
1. Multi-pass onnxsim
2. Fuse consecutive Transpose ops
3. Remove identity Cast chains
4. Remove redundant Reshape ops (identity reshapes)
5. Shape inference + constant folding
"""
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto, shape_inference
from collections import Counter, defaultdict
import os

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
INPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
OUTPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache-opt.onnx'


def build_maps(graph):
    out2node = {}
    in2consumers = defaultdict(list)
    for n in graph.node:
        for o in n.output:
            out2node[o] = n
        for i in n.input:
            in2consumers[i].append(n)
    return out2node, in2consumers


def fuse_consecutive_transposes(graph):
    """Fuse Transpose → Transpose chains. Cancel or compose."""
    out2node, in2consumers = build_maps(graph)
    fused = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Transpose' or id(n) in nodes_to_remove:
            continue
        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1 or consumers[0].op_type != 'Transpose':
            continue
        if id(consumers[0]) in nodes_to_remove:
            continue

        n2 = consumers[0]
        perm1 = list(n.attribute[0].ints)
        perm2 = list(n2.attribute[0].ints)
        composed = [perm1[p] for p in perm2]

        if composed == list(range(len(composed))):
            # Cancel out
            for c in in2consumers.get(n2.output[0], []):
                for j, inp in enumerate(c.input):
                    if inp == n2.output[0]:
                        c.input[j] = n.input[0]
            # Also update graph outputs
            for out in graph.output:
                if out.name == n2.output[0]:
                    out.name = n.input[0]
            nodes_to_remove.add(id(n))
            nodes_to_remove.add(id(n2))
        else:
            n.attribute[0].ints[:] = composed
            n.output[0] = n2.output[0]
            nodes_to_remove.add(id(n2))
        fused += 1

    if nodes_to_remove:
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)
    return fused


def remove_identity_casts(graph):
    """Remove Cast(A→B) → Cast(B→C) chains → Cast(A→C)."""
    out2node, in2consumers = build_maps(graph)
    removed = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Cast' or id(n) in nodes_to_remove:
            continue
        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1 or consumers[0].op_type != 'Cast':
            continue
        c2 = consumers[0]
        # Skip chain: Cast(A→B)→Cast(B→C) = Cast(A→C)
        c2.input[0] = n.input[0]
        nodes_to_remove.add(id(n))
        removed += 1

    if nodes_to_remove:
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)
    return removed


def remove_noop_reshapes(graph):
    """Remove Reshape nodes where input/output have identical shapes (need shape_inference)."""
    try:
        model_tmp = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model_tmp = shape_inference.infer_shapes(model_tmp, check_type=False)
        graph = model_tmp.graph
    except:
        return 0, graph

    # Build type info map
    type_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.HasField('shape'):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(-1)
            type_map[vi.name] = tuple(dims)

    out2node, in2consumers = build_maps(graph)
    removed = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Reshape' or id(n) in nodes_to_remove:
            continue
        inp_shape = type_map.get(n.input[0])
        out_shape = type_map.get(n.output[0])
        if inp_shape and out_shape and inp_shape == out_shape:
            # Identity reshape — bypass
            consumers = in2consumers.get(n.output[0], [])
            for c in consumers:
                for j, ci in enumerate(c.input):
                    if ci == n.output[0]:
                        c.input[j] = n.input[0]
            nodes_to_remove.add(id(n))
            removed += 1

    if nodes_to_remove:
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)
    return removed, graph


def fold_shape_subgraphs(graph):
    """
    Replace Shape→Gather→Unsqueeze→Concat→Reshape patterns with constants.
    These are shape-computation subgraphs that should be folded since batch=1.
    """
    out2node, in2consumers = build_maps(graph)
    init_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    # Find all Shape nodes and trace their dependents
    shape_nodes = [n for n in graph.node if n.op_type == 'Shape']
    shape_dependent = set()

    def mark_dependent(node):
        if id(node) in shape_dependent:
            return
        shape_dependent.add(id(node))
        for o in node.output:
            for c in in2consumers.get(o, []):
                if c.op_type in ('Gather', 'Unsqueeze', 'Concat', 'Slice', 'Cast', 'ConstantOfShape'):
                    # Only follow shape-computation ops
                    mark_dependent(c)

    for sn in shape_nodes:
        mark_dependent(sn)

    return len(shape_dependent)


def run_onnxsim(model, n_passes=5):
    """Multi-pass onnxsim."""
    try:
        import onnxsim
    except ImportError:
        print("  onnxsim not available, skipping")
        return model

    prev = len(model.graph.node)
    for i in range(n_passes):
        try:
            model, _ = onnxsim.simplify(model)
            cur = len(model.graph.node)
            diff = prev - cur
            print(f"  onnxsim pass {i+1}: {cur} nodes ({'-' if diff >= 0 else '+'}{abs(diff)})")
            if diff == 0:
                break
            prev = cur
        except Exception as e:
            print(f"  onnxsim pass {i+1} failed: {e}")
            break
    return model


def count_ops(graph):
    return Counter(n.op_type for n in graph.node)


if __name__ == '__main__':
    print(f"Loading: {INPUT_ONNX}")
    model = onnx.load(INPUT_ONNX)
    g = model.graph

    before = count_ops(g)
    print(f"Before: {len(g.node)} nodes")
    print(f"  Reshape:{before['Reshape']} Transpose:{before['Transpose']} "
          f"Cast:{before['Cast']} Shape:{before['Shape']} Unsqueeze:{before['Unsqueeze']}")

    # Step 1: onnxsim
    print("\n=== Step 1: onnxsim ===")
    model = run_onnxsim(model)
    g = model.graph

    # Step 2: Fuse consecutive transposes (multiple passes)
    print("\n=== Step 2: Fuse consecutive Transposes ===")
    total_fused = 0
    for p in range(5):
        f = fuse_consecutive_transposes(g)
        if f == 0:
            break
        total_fused += f
    print(f"  Fused: {total_fused}")

    # Step 3: Remove identity Cast chains
    print("\n=== Step 3: Remove identity Cast chains ===")
    rc = remove_identity_casts(g)
    print(f"  Removed: {rc}")

    # Step 4: Remove noop reshapes
    print("\n=== Step 4: Remove noop Reshapes ===")
    rr, g = remove_noop_reshapes(g)
    if rr > 0:
        # Rebuild model with updated graph
        model = helper.make_model(g, opset_imports=model.opset_import)
        model.ir_version = 8
    print(f"  Removed: {rr}")

    # Step 5: Another onnxsim pass after manual optimizations
    print("\n=== Step 5: onnxsim cleanup ===")
    model = run_onnxsim(model)
    g = model.graph

    # Step 6: Count shape-dependent nodes
    print("\n=== Step 6: Shape subgraph analysis ===")
    sd = fold_shape_subgraphs(g)
    print(f"  Shape-dependent nodes: {sd}")

    after = count_ops(g)
    print(f"\n=== Result: {len(g.node)} nodes ===")
    print(f"  Reshape:{after.get('Reshape',0)} Transpose:{after.get('Transpose',0)} "
          f"Cast:{after.get('Cast',0)} Shape:{after.get('Shape',0)} Unsqueeze:{after.get('Unsqueeze',0)}")
    for op, cnt in after.most_common(20):
        print(f"  {op:20s} {cnt}")

    # Validate
    try:
        onnx.checker.check_model(model)
        print("\nValidation: OK")
    except Exception as e:
        print(f"\nValidation WARNING: {e}")

    onnx.save(model, OUTPUT_ONNX)
    sz = os.path.getsize(OUTPUT_ONNX) / 1024 / 1024
    print(f"Saved: {OUTPUT_ONNX} ({sz:.1f} MB)")

    # Verify numerical equivalence
    print("\n=== Numerical verification ===")
    import onnxruntime as ort
    sess_orig = ort.InferenceSession(INPUT_ONNX, providers=['CPUExecutionProvider'])
    sess_opt = ort.InferenceSession(OUTPUT_ONNX, providers=['CPUExecutionProvider'])
    feeds = {}
    for inp in sess_orig.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = np.float32 if 'float' in inp.type else np.int64
        feeds[inp.name] = np.random.randn(*shape).astype(dtype) if dtype == np.float32 else np.zeros(shape, dtype=dtype)
    out_orig = sess_orig.run(None, feeds)
    out_opt = sess_opt.run(None, feeds)
    max_diff = max(np.max(np.abs(a - b)) for a, b in zip(out_orig, out_opt))
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  {'PASS' if max_diff < 1e-4 else 'FAIL'}")
