"""
ONNX graph surgery: fuse/remove redundant Reshape+Transpose ops.
Goal: reduce RKNN compiled layer count (1443 → <1000) for faster dispatch.

Strategies:
1. Fuse Reshape→Transpose chains into single Reshape (when net effect is just a reshape)
2. Remove identity Reshapes (output shape == input shape)
3. Fuse consecutive Transpose ops (compose permutations)
4. Remove Unsqueeze→downstream patterns where possible
"""
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from collections import defaultdict
import copy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
INPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx'
OUTPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-fused.onnx'


def build_maps(graph):
    """Build name->node, output->node, input->consumers maps."""
    out2node = {}  # output_name -> producer node
    in2consumers = defaultdict(list)  # output_name -> list of consumer nodes
    for n in graph.node:
        for o in n.output:
            out2node[o] = n
        for i in n.input:
            in2consumers[i].append(n)
    return out2node, in2consumers


def get_constant_value(name, graph, out2node):
    """Get constant tensor value by name."""
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    # Check Constant nodes
    if name in out2node:
        n = out2node[name]
        if n.op_type == 'Constant':
            for attr in n.attribute:
                if attr.name == 'value':
                    return numpy_helper.to_array(attr.t)
    return None


def compose_transpose(perm1, perm2):
    """Compose two transpose permutations: result[i] = perm1[perm2[i]]."""
    return [perm1[p] for p in perm2]


def is_identity_perm(perm):
    return list(perm) == list(range(len(perm)))


def fuse_consecutive_transposes(graph):
    """Fuse Transpose → Transpose chains."""
    out2node, in2consumers = build_maps(graph)
    fused = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Transpose' or id(n) in nodes_to_remove:
            continue
        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1 or consumers[0].op_type != 'Transpose':
            continue

        n2 = consumers[0]
        perm1 = list(n.attribute[0].ints)
        perm2 = list(n2.attribute[0].ints)
        composed = compose_transpose(perm1, perm2)

        if is_identity_perm(composed):
            # Both transposes cancel out - bypass both
            for consumer in in2consumers.get(n2.output[0], []):
                for j, inp in enumerate(consumer.input):
                    if inp == n2.output[0]:
                        consumer.input[j] = n.input[0]
            nodes_to_remove.add(id(n))
            nodes_to_remove.add(id(n2))
            fused += 1
        else:
            # Replace with single transpose
            n.attribute[0].ints[:] = composed
            n.output[0] = n2.output[0]
            nodes_to_remove.add(id(n2))
            fused += 1

    if nodes_to_remove:
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)

    return fused


def remove_identity_reshapes(graph):
    """Remove Reshape nodes that don't change the shape (identity reshapes)."""
    out2node, in2consumers = build_maps(graph)
    removed = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Reshape' or id(n) in nodes_to_remove:
            continue
        if len(n.input) < 2:
            continue

        shape_val = get_constant_value(n.input[1], graph, out2node)
        if shape_val is None:
            continue

        # Check if this reshape has a single consumer and can be bypassed
        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1:
            continue

        # Check if the reshape just adds/removes dimensions of size 1
        # We can't easily know input shape without running shape inference,
        # but we can detect reshape→transpose→reshape patterns
        consumer = consumers[0]
        if consumer.op_type == 'Transpose':
            trans_consumers = in2consumers.get(consumer.output[0], [])
            if len(trans_consumers) == 1 and trans_consumers[0].op_type == 'Reshape':
                # Reshape → Transpose → Reshape chain
                # Could potentially be a single Reshape if the net effect is just reshaping
                pass  # Complex to verify without shape inference

    return removed


def remove_redundant_unsqueeze(graph):
    """Remove Unsqueeze that are immediately consumed by ops that don't need the extra dim."""
    out2node, in2consumers = build_maps(graph)
    removed = 0
    nodes_to_remove = set()

    for n in graph.node:
        if n.op_type != 'Unsqueeze' or id(n) in nodes_to_remove:
            continue

        consumers = in2consumers.get(n.output[0], [])
        if len(consumers) != 1:
            continue

        consumer = consumers[0]
        # Unsqueeze → Gather pattern: Gather can often index directly
        if consumer.op_type == 'Gather':
            # Check if the Unsqueeze just wraps a scalar to 1D for Gather
            axes = None
            for attr in n.attribute:
                if attr.name == 'axes':
                    axes = list(attr.ints)
            if axes is None and len(n.input) > 1:
                axes_val = get_constant_value(n.input[1], graph, out2node)
                if axes_val is not None:
                    axes = axes_val.tolist()

        # Unsqueeze → Concat → Reshape pattern (shape construction)
        # These are just for building shape tensors — RKNN may or may not compile them
        if consumer.op_type == 'Concat':
            # This is typically shape construction, leave it
            pass

    return removed


def simplify_attention_reshapes(graph):
    """
    In attention: Reshape[B,T,H,D] → Transpose[1,2,0,3] → MatMul
    If we can change to Reshape[H,B,T,D] directly, we skip the Transpose.
    But this requires changing the Reshape target shape, which is tied to the model logic.

    Alternative: combine Reshape+Transpose into a single 'FlattenTranspose' equivalent.
    Since RKNN doesn't have FlattenTranspose, we try to merge them using onnx-simplifier's
    constant folding after manual editing.
    """
    out2node, in2consumers = build_maps(graph)
    optimized = 0
    return optimized


def count_ops(graph):
    from collections import Counter
    return Counter(n.op_type for n in graph.node)


if __name__ == '__main__':
    print(f"Loading: {INPUT_ONNX}")
    model = onnx.load(INPUT_ONNX)
    g = model.graph

    before = count_ops(g)
    print(f"Before: {len(g.node)} nodes")
    print(f"  Reshape: {before['Reshape']}, Transpose: {before['Transpose']}, Unsqueeze: {before['Unsqueeze']}")

    # Pass 1: Fuse consecutive transposes
    fused = fuse_consecutive_transposes(g)
    print(f"\nPass 1 - Fuse consecutive Transposes: {fused} fused")

    # Rebuild maps after modification
    # Pass 2: Try again after first pass
    fused2 = fuse_consecutive_transposes(g)
    print(f"Pass 2 - Fuse consecutive Transposes (2nd pass): {fused2} fused")

    after = count_ops(g)
    print(f"\nAfter: {len(g.node)} nodes")
    print(f"  Reshape: {after['Reshape']}, Transpose: {after['Transpose']}, Unsqueeze: {after['Unsqueeze']}")
    print(f"  Removed: {len(list(before.elements())) - len(list(after.elements()))} nodes")

    # Validate
    try:
        onnx.checker.check_model(model)
        print("\nModel validation: OK")
    except Exception as e:
        print(f"\nModel validation WARNING: {e}")

    onnx.save(model, OUTPUT_ONNX)
    print(f"Saved: {OUTPUT_ONNX}")
