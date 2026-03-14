"""
Fuse Transpose→MatMul patterns.
When Transpose feeds directly into MatMul, we can absorb the transpose
into the MatMul by pre-transposing static weights or restructuring.

For patterns like: Transpose(A, perm=(1,0,2)) → MatMul(A_t, B)
If A comes from a preceding op and perm is a simple transpose of the
last two dims, this is equivalent to: MatMul(A, B) with transA flag.
But ONNX MatMul doesn't have transA/transB flags.

Instead, we handle the case where:
1. Transpose permutes a static weight tensor → pre-transpose the weight
2. Transpose of a dynamic tensor before MatMul → keep (can't fuse)
"""
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import numpy as np
import copy
from collections import Counter

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SRC = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
DST = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache-fused.onnx'


def main():
    print("Loading nocache model...")
    m = onnx.load(SRC)
    g = m.graph

    # Build maps
    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n

    consumers = {}
    for n in g.node:
        for inp in n.input:
            if inp not in consumers:
                consumers[inp] = []
            consumers[inp].append(n)

    init_map = {i.name: i for i in g.initializer}

    # Find fuseable Transpose→MatMul patterns
    # Case 1: Transpose of a constant/initializer → pre-transpose
    # Case 2: Transpose with single MatMul consumer and specific perm
    fused = 0
    removed_nodes = set()

    for n in g.node:
        if n.op_type != 'Transpose' or id(n) in removed_nodes:
            continue

        tp_input = n.input[0]
        tp_output = n.output[0]

        # Get perm
        perm = None
        for attr in n.attribute:
            if attr.name == 'perm':
                perm = tuple(attr.ints)

        if perm is None:
            continue

        # Check if the transpose input is a static initializer
        if tp_input in init_map:
            # Pre-transpose the initializer
            init_tensor = init_map[tp_input]
            arr = numpy_helper.to_array(init_tensor)
            arr_t = np.transpose(arr, perm)

            # Replace the initializer with pre-transposed version
            new_init = numpy_helper.from_array(arr_t, name=tp_input)
            init_map[tp_input] = new_init

            # Redirect all consumers of tp_output to use tp_input directly
            for consumer in consumers.get(tp_output, []):
                for j, ci in enumerate(consumer.input):
                    if ci == tp_output:
                        consumer.input[j] = tp_input

            removed_nodes.add(id(n))
            fused += 1
            continue

        # Check if tp_input comes from a Constant node
        if tp_input in out2node and out2node[tp_input].op_type == 'Constant':
            const_node = out2node[tp_input]
            # Get constant value
            for attr in const_node.attribute:
                if attr.name == 'value':
                    arr = numpy_helper.to_array(attr.t)
                    arr_t = np.transpose(arr, perm)
                    new_const = numpy_helper.from_array(arr_t, name='_pre_transposed')
                    attr.t.CopyFrom(new_const)
                    # Also update dims in constant output
                    break

            # Redirect consumers
            for consumer in consumers.get(tp_output, []):
                for j, ci in enumerate(consumer.input):
                    if ci == tp_output:
                        consumer.input[j] = tp_input

            removed_nodes.add(id(n))
            fused += 1
            continue

    print(f"Fused (pre-transposed): {fused}")

    # Remove fused Transpose nodes
    new_nodes = [n for n in g.node if id(n) not in removed_nodes]
    print(f"Nodes: {len(g.node)} → {len(new_nodes)} (removed {len(g.node) - len(new_nodes)})")

    # Update initializers
    new_inits = list(init_map.values())

    # Build new graph
    new_graph = helper.make_graph(
        new_nodes, g.name + '_fused',
        list(g.input), list(g.output),
        initializer=new_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=m.opset_import)
    new_model.ir_version = m.ir_version

    try:
        onnx.checker.check_model(new_model)
        print("✓ ONNX valid")
    except Exception as e:
        print(f"✗ Validation: {e}")

    onnx.save(new_model, DST)
    import os
    print(f"Saved: {DST} ({os.path.getsize(DST)/1024/1024:.1f} MB)")

    # Verify with ORT
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
        print(f"✓ ORT: {len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs")

        # Compare with original
        sess_orig = ort.InferenceSession(SRC, providers=['CPUExecutionProvider'])
        feeds = {}
        for inp in sess_orig.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dtype = np.float32 if 'float' in inp.type else np.int64
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
        out_orig = sess_orig.run(None, feeds)
        out_fused = sess.run(None, feeds)
        max_diff = max(np.max(np.abs(a - b)) for a, b in zip(out_orig, out_fused))
        print(f"  Max diff vs original: {max_diff:.6e}")
    except Exception as e:
        print(f"✗ ORT: {e}")


if __name__ == '__main__':
    main()
