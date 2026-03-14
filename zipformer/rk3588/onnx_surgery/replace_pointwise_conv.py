"""
Replace pointwise Conv1D (kernel=1) with MatMul to eliminate Transpose ops.

Pattern before:
  x[T,B,C] -> Transpose[1,2,0] -> Conv1D(C_in,C_out,k=1) -> output[B,C_out,T]

Pattern after:
  x[T,B,C] -> Reshape[T*B,C] -> MatMul(W) -> Reshape[T,B,C_out] -> output

This eliminates the Transpose ops that wrap pointwise convolutions.
For depthwise convs (kernel>1), we keep the original Conv.
"""
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from collections import defaultdict
import os

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SIM_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
OUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-pwconv.onnx'


def build_maps(graph):
    out2node = {}
    in2consumers = defaultdict(list)
    for n in graph.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2consumers[inp].append(n)
    return out2node, in2consumers


def get_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def is_pointwise_conv(node, graph):
    """Check if Conv node has kernel_size=1."""
    if node.op_type != 'Conv':
        return False
    weight = get_initializer(graph, node.input[1])
    if weight is None:
        return False
    # Conv1D weight: [C_out, C_in/groups, kernel_size]
    if len(weight.shape) == 3 and weight.shape[2] == 1:
        return True
    return False


def replace_pointwise_convs(model):
    """Replace pointwise Conv1D with MatMul, removing surrounding Transposes."""
    g = model.graph
    out2node, in2consumers = build_maps(g)

    replaced = 0
    transpose_removed = 0
    nodes_to_remove = set()
    nodes_to_add = []
    inits_to_add = []
    inits_to_remove = set()

    for node in list(g.node):
        if not is_pointwise_conv(node, g) or id(node) in nodes_to_remove:
            continue

        weight = get_initializer(g, node.input[1])
        # Conv1D weight: [C_out, C_in, 1] -> MatMul weight: [C_in, C_out]
        c_out, c_in_per_group, _ = weight.shape

        # Check for groups
        groups = 1
        for attr in node.attribute:
            if attr.name == 'group':
                groups = attr.i
        if groups > 1:
            continue  # Skip grouped/depthwise convolutions

        c_in = c_in_per_group * groups

        # Check if input comes from Transpose[1,2,0] (TBC -> BCT)
        input_transpose = out2node.get(node.input[0])
        has_input_transpose = False
        if input_transpose and input_transpose.op_type == 'Transpose':
            perm = list(input_transpose.attribute[0].ints)
            if perm == [1, 2, 0]:
                has_input_transpose = True

        # Check if output goes to Transpose[2,0,1] (BCT -> TBC)
        consumers = in2consumers.get(node.output[0], [])
        output_transpose = None
        for c in consumers:
            if c.op_type == 'Transpose':
                perm = list(c.attribute[0].ints)
                if perm == [2, 0, 1]:
                    output_transpose = c
                    break

        if not has_input_transpose and output_transpose is None:
            continue

        # Create MatMul weight: [C_in, C_out]
        matmul_weight = weight.reshape(c_out, c_in).T  # [C_in, C_out]
        weight_name = f'{node.name}_matmul_weight'
        weight_init = numpy_helper.from_array(matmul_weight, name=weight_name)
        inits_to_add.append(weight_init)

        # Handle bias
        has_bias = len(node.input) > 2 and node.input[2] != ''
        bias_name = node.input[2] if has_bias else None

        # Determine input and output connections
        # For input: if there's an input Transpose [1,2,0] (TBC→BCT),
        # we can use the pre-Transpose data directly (TBC format) for MatMul.
        # But only remove the Transpose if no other node consumes its output.
        if has_input_transpose:
            all_consumers_of_transpose = in2consumers.get(input_transpose.output[0], [])
            non_removed = [c for c in all_consumers_of_transpose
                          if id(c) not in nodes_to_remove and id(c) != id(node)]
            if not non_removed:
                # Safe to bypass: use pre-transpose input (TBC)
                matmul_input = input_transpose.input[0]
                nodes_to_remove.add(id(input_transpose))
                transpose_removed += 1
            else:
                # Transpose has other consumers — can't remove it
                # Input is BCT format, MatMul needs special handling
                # Skip this conv for now
                continue
        else:
            # No input transpose — input is already in BCT format from preceding Conv/op
            # MatMul expects TBC. Need to handle format.
            # For simplicity, skip if no input transpose AND no output transpose
            if output_transpose is None:
                continue
            # Input is BCT. We need Transpose[2,0,1] to get TBC, MatMul, then output is TBC.
            # But then the output_transpose[2,0,1] would convert TBC→BCT→TBC which is wrong.
            # Actually: Conv in BCT out BCT. Output Transpose[2,0,1] converts BCT→TBC.
            # MatMul replacement: input BCT, we need to transpose to TBC first, then MatMul.
            # This doesn't save any ops. Skip.
            continue

        if output_transpose:
            all_out_consumers = in2consumers.get(node.output[0], [])
            non_removed_out = [c for c in all_out_consumers
                              if id(c) not in nodes_to_remove and id(c) != id(output_transpose)]
            if not non_removed_out:
                matmul_output = output_transpose.output[0]
                nodes_to_remove.add(id(output_transpose))
                transpose_removed += 1
            else:
                matmul_output = node.output[0]
        else:
            # No output transpose — MatMul outputs TBC which is what downstream expects
            matmul_output = node.output[0]

        # Create MatMul node
        # Input: [T, B, C_in] (TBC) -> MatMul([C_in, C_out]) -> [T, B, C_out] (TBC)
        matmul_out_name = f'{node.name}_matmul_out'
        matmul_node = helper.make_node(
            'MatMul',
            inputs=[matmul_input, weight_name],
            outputs=[matmul_out_name if has_bias else matmul_output],
            name=f'{node.name}_as_matmul'
        )
        nodes_to_add.append(matmul_node)

        if has_bias:
            # Add bias: [T, B, C_out] + [C_out] -> [T, B, C_out]
            add_node = helper.make_node(
                'Add',
                inputs=[matmul_out_name, bias_name],
                outputs=[matmul_output],
                name=f'{node.name}_bias_add'
            )
            nodes_to_add.append(add_node)

        nodes_to_remove.add(id(node))
        inits_to_remove.add(node.input[1])  # Remove old Conv weight
        replaced += 1

    # Apply changes: append new nodes, remove old, then topological sort
    new_nodes = [n for n in g.node if id(n) not in nodes_to_remove]
    new_nodes.extend(nodes_to_add)

    # Topological sort
    all_outputs = set()
    for inp in g.input:
        all_outputs.add(inp.name)
    for init in g.initializer:
        all_outputs.add(init.name)

    sorted_nodes = []
    remaining = list(new_nodes)
    max_iter = len(remaining) * 2
    iteration = 0
    while remaining and iteration < max_iter:
        iteration += 1
        progress = False
        next_remaining = []
        for n in remaining:
            if all(i in all_outputs or i == '' for i in n.input):
                sorted_nodes.append(n)
                for o in n.output:
                    all_outputs.add(o)
                progress = True
            else:
                next_remaining.append(n)
        remaining = next_remaining
        if not progress:
            # Deadlock — add remaining as-is
            sorted_nodes.extend(remaining)
            break

    del g.node[:]
    g.node.extend(sorted_nodes)

    # Update initializers
    new_inits = [i for i in g.initializer if i.name not in inits_to_remove]
    new_inits.extend(inits_to_add)
    del g.initializer[:]
    g.initializer.extend(new_inits)

    return replaced, transpose_removed


def count_ops(graph):
    from collections import Counter
    return Counter(n.op_type for n in graph.node)


if __name__ == '__main__':
    print(f"Loading: {SIM_ONNX}")
    model = onnx.load(SIM_ONNX)
    g = model.graph

    before = count_ops(g)
    print(f"Before: {len(g.node)} nodes")
    print(f"  Conv: {before['Conv']}, Transpose: {before['Transpose']}, Reshape: {before['Reshape']}")

    # Count pointwise convs
    pw_count = sum(1 for n in g.node if is_pointwise_conv(n, g))
    print(f"  Pointwise Conv1D (k=1): {pw_count}")

    # Replace
    replaced, trans_removed = replace_pointwise_convs(model)
    print(f"\nReplaced: {replaced} pointwise Conv1D -> MatMul")
    print(f"Removed Transposes: {trans_removed}")

    after = count_ops(g)
    print(f"\nAfter: {len(g.node)} nodes")
    print(f"  Conv: {after.get('Conv',0)}, Transpose: {after.get('Transpose',0)}, Reshape: {after.get('Reshape',0)}")
    print(f"  MatMul: {after.get('MatMul',0)} (was {before.get('MatMul',0)})")
    print(f"  Net change: {len(g.node) - sum(before.values())} nodes")

    # Verify
    print("\n=== Verification ===")
    import onnxruntime as ort
    from convert_encoder_int8_optarget import ENC_SCHEMA

    sess_orig = ort.InferenceSession(SIM_ONNX, providers=['CPUExecutionProvider'])
    onnx.save(model, OUT_ONNX)

    sess_new = ort.InferenceSession(OUT_ONNX, providers=['CPUExecutionProvider'])

    inputs = {}
    for name, shape, dtype in ENC_SCHEMA:
        inputs[name] = np.zeros(shape, dtype=np.dtype(dtype))
    inputs['x'] = np.random.randn(1, 39, 80).astype(np.float32)

    out_orig = sess_orig.run(None, inputs)
    out_new = sess_new.run(None, inputs)

    all_ok = True
    for i, (o, n) in enumerate(zip(out_orig, out_new)):
        max_diff = np.max(np.abs(o - n))
        cos = np.dot(o.flatten(), n.flatten()) / (np.linalg.norm(o) * np.linalg.norm(n) + 1e-10)
        ok = max_diff < 1e-4 or cos > 0.9999
        if not ok:
            all_ok = False
            print(f"  out[{i}]: max_diff={max_diff:.6f}  cos_sim={cos:.6f}  MISMATCH!")
    if all_ok:
        print(f"  All {len(out_orig)} outputs match!")
    else:
        print("  WARNING: Some outputs don't match!")

    print(f"\nSaved: {OUT_ONNX}")
    print(f"Size: {os.path.getsize(OUT_ONNX)/1024/1024:.1f} MB")
