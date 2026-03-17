"""Fix CitriNet ONNX graph for RKNN compatibility.

Fixes applied:
1. Remove LogSoftmax (unnecessary for CTC greedy decoding)
2. Replace masked SE blocks with simple ReduceMean
   - The fixlen model has fixed input length, so padding masks are always all-True
   - The mask chain (ConstantOfShape/Equal/Less/Not/Cast/Where) + ReduceSum/Div
     is replaced with a single ReduceMean
   - This removes ~184 nodes that cause RKNN to produce wrong output
3. Replace ReduceMean with depthwise Conv (RKNN ReduceMean bug workaround)
   - ReduceMean(axis=-1) on [1,C,T] → depthwise Conv(kernel=[T], w=1/T, group=C)
4. Remove Squeeze/Unsqueeze (RKNN Squeeze bug)
   - RKNN applies Squeeze axis on NHWC layout, producing wrong output
   - Input changed from [1,80,1,300] to [1,80,300], output from [1,2049,1,38] to 3D
"""
import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference
import sys
import numpy as np


def remove_logsoftmax(graph):
    """Remove LogSoftmax nodes, connecting input directly to output."""
    removed = 0
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == 'LogSoftmax':
            logsoftmax_input = node.input[0]
            logsoftmax_output = node.output[0]
            for other_node in graph.node:
                for i, inp in enumerate(other_node.input):
                    if inp == logsoftmax_output:
                        other_node.input[i] = logsoftmax_input
            for out in graph.output:
                if out.name == logsoftmax_output:
                    out.name = logsoftmax_input
            nodes_to_remove.append(node)
            removed += 1
    for node in nodes_to_remove:
        graph.node.remove(node)
    print(f'  Removed {removed} LogSoftmax nodes')
    return removed


def replace_masked_se_with_reducemean(graph):
    """Replace masked SE pooling with simple ReduceMean.

    Pattern in each SE block (23 blocks):
      Conv_output [1, 1024, T]
      → mask chain (ConstantOfShape/Equal/Less/Not/Cast/Where/Expand etc.)
      → ReduceSum(axis=-1) [1, 1024, 1]
      → Div(by count) → effectively ReduceMean
      → Transpose → MatMul → ReLU → MatMul → Transpose → Sigmoid
      → Mul(Conv_output × SE_weight)

    Since fixlen=300, masks are always all-True. Replace with ReduceMean.
    """
    # Build output→node map and input→consumers map
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    se_blocks_fixed = 0
    nodes_to_remove = set()

    # Find ReduceSum pairs grouped by block prefix
    reducesum_nodes = [n for n in graph.node if n.op_type == 'ReduceSum']
    from collections import defaultdict
    block_reducesums = defaultdict(list)
    for n in reducesum_nodes:
        parts = n.name.rsplit('/', 1)
        prefix = parts[0] if len(parts) > 1 else ''
        block_reducesums[prefix].append(n)

    for prefix, rs_nodes in block_reducesums.items():
        if len(rs_nodes) != 2:
            continue

        # Find value ReduceSum (from Where) and count ReduceSum (from Cast)
        value_rs = None
        count_rs = None
        for rs in rs_nodes:
            producer = output_to_node.get(rs.input[0])
            if producer and producer.op_type == 'Where':
                value_rs = rs
            elif producer and producer.op_type == 'Cast':
                count_rs = rs

        if not value_rs or not count_rs:
            continue

        # Find Div node
        div_node = None
        for n in graph.node:
            if n.op_type == 'Div' and value_rs.output[0] in list(n.input):
                div_node = n
                break
        if not div_node:
            continue

        # Find Conv output (data input to Where_1, which is the 3rd input)
        where1_node = output_to_node.get(value_rs.input[0])
        if not where1_node or where1_node.op_type != 'Where':
            continue
        conv_output = where1_node.input[2]

        # Create ReduceMean node (opset 13: axes is attribute, not input)
        reducemean_name = prefix + '/ReduceMean_fixed'
        reducemean_node = helper.make_node(
            'ReduceMean',
            inputs=[conv_output],
            outputs=[div_node.output[0]],
            name=reducemean_name,
            axes=[-1],
            keepdims=1
        )

        # Collect ALL mask chain nodes by tracing dependencies
        # Start from div_node and trace backward, collecting everything
        # that is NOT the conv_output or SE FC chain
        def trace_backward(node, stop_outputs):
            """Recursively collect nodes feeding into this node."""
            collected = set()
            collected.add(id(node))
            for inp in node.input:
                if inp in stop_outputs:
                    continue
                producer = output_to_node.get(inp)
                if producer:
                    collected.update(trace_backward(producer, stop_outputs))
            return collected

        # Stop at conv_output (don't trace into the actual encoder)
        stop_at = {conv_output}
        # Also stop at graph inputs
        for gi in graph.input:
            stop_at.add(gi.name)
        # Also stop at initializers
        for init in graph.initializer:
            stop_at.add(init.name)

        mask_node_ids = trace_backward(div_node, stop_at)
        nodes_to_remove.update(mask_node_ids)

        # Redirect the final Mul to use conv_output instead of Where_1 output
        for n in graph.node:
            if n.op_type == 'Mul':
                for i, inp in enumerate(n.input):
                    if inp == where1_node.output[0]:
                        n.input[i] = conv_output

        # Insert ReduceMean right after the conv_output producer
        conv_producer = output_to_node.get(conv_output)
        if conv_producer:
            idx = list(graph.node).index(conv_producer)
            graph.node.insert(idx + 1, reducemean_node)
        else:
            graph.node.append(reducemean_node)
        se_blocks_fixed += 1

    # Remove collected nodes
    nodes_to_keep = [n for n in graph.node if id(n) not in nodes_to_remove]
    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(nodes_to_keep)

    print(f'  Replaced {se_blocks_fixed} masked SE blocks with ReduceMean')
    print(f'  Removed {len(nodes_to_remove)} mask chain nodes')
    return se_blocks_fixed


def replace_reducemean_with_conv(model, shapes):
    """Replace ReduceMean with depthwise Conv (RKNN bug workaround).

    ReduceMean(axis=-1) on [1,C,1,T] → depthwise Conv(kernel=[1,T], w=1/T, group=C)
    shapes: dict mapping tensor name → shape list, from shape inference.
    """
    graph = model.graph

    replaced = 0
    new_nodes = []
    new_initializers = []

    for node in graph.node:
        if node.op_type != 'ReduceMean':
            new_nodes.append(node)
            continue

        # Get axes from attribute
        axes = []
        for attr in node.attribute:
            if attr.name == 'axes':
                axes = list(attr.ints)

        in_name = node.input[0]
        out_name = node.output[0]
        in_shape = shapes.get(in_name)

        if in_shape is None:
            print(f'  WARNING: no shape for {node.name} input {in_name}, keeping ReduceMean')
            new_nodes.append(node)
            continue

        ndim = len(in_shape)
        # Normalize negative axes
        axes = [a if a >= 0 else a + ndim for a in axes]

        if axes == [ndim - 1]:
            C = in_shape[1]
            T = in_shape[-1]
            w_name = f'{node.name}_conv_w'

            if ndim == 3:
                # 1D: ReduceMean(axis=2) on [1,C,T] → depthwise Conv1D
                w = (1.0 / T) * np.ones((C, 1, T), dtype=np.float32)
                new_initializers.append(numpy_helper.from_array(w, name=w_name))
                conv_node = helper.make_node(
                    'Conv',
                    inputs=[in_name, w_name],
                    outputs=[out_name],
                    name=f'{node.name}_conv',
                    dilations=[1],
                    group=C,
                    kernel_shape=[T],
                    pads=[0, 0],
                    strides=[1],
                )
                print(f'  ReduceMean(axis=-1) → 1D depthwise Conv({T}): {node.name}, C={C}')
            elif ndim == 4:
                # 2D: ReduceMean(axis=3) on [1,C,H,T] → depthwise Conv2D
                w = (1.0 / T) * np.ones((C, 1, 1, T), dtype=np.float32)
                new_initializers.append(numpy_helper.from_array(w, name=w_name))
                conv_node = helper.make_node(
                    'Conv',
                    inputs=[in_name, w_name],
                    outputs=[out_name],
                    name=f'{node.name}_conv',
                    dilations=[1, 1],
                    group=C,
                    kernel_shape=[1, T],
                    pads=[0, 0, 0, 0],
                    strides=[1, 1],
                )
                print(f'  ReduceMean(axis=-1) → 2D depthwise Conv(1,{T}): {node.name}, C={C}')
            else:
                print(f'  WARNING: Unexpected ndim={ndim} in {node.name}, keeping ReduceMean')
                new_nodes.append(node)
                continue

            new_nodes.append(conv_node)
            replaced += 1
        else:
            print(f'  WARNING: Unexpected axes={axes} in {node.name}, keeping ReduceMean')
            new_nodes.append(node)

    # Replace nodes
    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(new_nodes)

    # Add initializers
    for init in new_initializers:
        graph.initializer.append(init)

    print(f'  Replaced {replaced} ReduceMean nodes with depthwise Conv')
    return replaced


def get_intermediate_shapes(model):
    """Get intermediate tensor shapes by running ORT with extra outputs."""
    import copy
    import onnxruntime as ort

    tmp_model = copy.deepcopy(model)
    graph = tmp_model.graph

    # Find SE block conv outputs (inputs to ReduceMean after Fix 2)
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    reducesum_nodes = [n for n in graph.node if n.op_type == 'ReduceSum']
    from collections import defaultdict
    block_reducesums = defaultdict(list)
    for n in reducesum_nodes:
        parts = n.name.rsplit('/', 1)
        prefix = parts[0] if len(parts) > 1 else ''
        block_reducesums[prefix].append(n)

    extra_outputs = []
    for prefix, rs_nodes in sorted(block_reducesums.items()):
        if len(rs_nodes) != 2:
            continue
        value_rs = None
        for rs in rs_nodes:
            producer = output_to_node.get(rs.input[0])
            if producer and producer.op_type == 'Where':
                value_rs = rs
        if not value_rs:
            continue
        where1_node = output_to_node.get(value_rs.input[0])
        if where1_node and where1_node.op_type == 'Where':
            conv_output = where1_node.input[2]
            extra_outputs.append(conv_output)
            graph.output.append(helper.make_tensor_value_info(
                conv_output, TensorProto.FLOAT, None))

    # Run ORT to get shapes
    import tempfile, os
    tmp_path = tempfile.mktemp(suffix='.onnx')
    onnx.save(tmp_model, tmp_path)
    sess = ort.InferenceSession(tmp_path)

    inp_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    inp = np.random.randn(*inp_shape).astype(np.float32)
    inp_name = model.graph.input[0].name
    outputs = sess.run(None, {inp_name: inp})

    shapes = {}
    n_orig = len(model.graph.output)
    for i, name in enumerate(extra_outputs):
        shapes[name] = list(outputs[n_orig + i].shape)

    os.unlink(tmp_path)
    return shapes


def remove_squeeze_unsqueeze(model):
    """Remove Squeeze/Unsqueeze ops that convert between 3D and 4D.

    RKNN bug: Squeeze on NCHW tensors applies axis on NHWC layout internally,
    producing completely wrong output. Fix: use 3D I/O directly.

    Input: [1, 80, 1, 300] → remove Squeeze → change input to [1, 80, 300]
    Output: remove Unsqueeze → change output to [1, 2049, 38] (3D)
    """
    graph = model.graph
    removed = 0

    # Find and remove Squeeze at input
    squeeze_nodes = [n for n in graph.node if n.op_type == 'Squeeze']
    for sq in squeeze_nodes:
        sq_input = sq.input[0]
        sq_output = sq.output[0]
        # Rewire: all consumers of squeeze output → use new 3D input
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == sq_output:
                    node.input[i] = 'audio_signal'
        # Remove Squeeze and its Constant input
        const_name = sq.input[1] if len(sq.input) > 1 else None
        graph.node.remove(sq)
        if const_name:
            const_nodes = [n for n in graph.node if const_name in n.output]
            for cn in const_nodes:
                graph.node.remove(cn)
        # Change input spec to 3D
        while len(graph.input) > 0:
            graph.input.pop()
        graph.input.append(helper.make_tensor_value_info(
            'audio_signal', TensorProto.FLOAT, [1, 80, 300]))
        removed += 1
        print(f'  Removed Squeeze: input changed to [1, 80, 300]')

    # Find and remove Unsqueeze at output
    unsqueeze_nodes = [n for n in graph.node if n.op_type == 'Unsqueeze']
    for usq in unsqueeze_nodes:
        usq_input = usq.input[0]
        usq_output = usq.output[0]
        # Rewire graph output
        for out in graph.output:
            if out.name == usq_output:
                out.name = usq_input
        # Remove Unsqueeze and its Constant input
        const_name = usq.input[1] if len(usq.input) > 1 else None
        graph.node.remove(usq)
        if const_name:
            const_nodes = [n for n in graph.node if const_name in n.output]
            for cn in const_nodes:
                graph.node.remove(cn)
        # Update output spec to 3D
        while len(graph.output) > 0:
            graph.output.pop()
        graph.output.append(helper.make_tensor_value_info(
            usq_input, TensorProto.FLOAT, None))
        removed += 1
        print(f'  Removed Unsqueeze: output is now 3D')

    print(f'  Removed {removed} Squeeze/Unsqueeze nodes')
    return removed


def fix_graph_outputs(graph):
    """Ensure graph outputs reference existing node outputs."""
    all_outputs = set()
    for node in graph.node:
        for o in node.output:
            all_outputs.add(o)
    for inp in graph.input:
        all_outputs.add(inp.name)

    for out in graph.output:
        if out.name not in all_outputs:
            print(f'  WARNING: graph output {out.name} not found in node outputs')


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else '../citrinet_npu_v2_fixlen.onnx'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '../citrinet_npu_v2_fixlen_fixed.onnx'

    print(f'Loading {input_path}...')
    model = onnx.load(input_path)
    graph = model.graph

    print(f'Original nodes: {len(graph.node)}')

    # Get intermediate tensor shapes via ORT (shape_inference fails on large models)
    print('Getting intermediate tensor shapes via ORT...')
    shapes = get_intermediate_shapes(model)
    print(f'  Got shapes for {len(shapes)} tensors')

    print('Fix 1: LogSoftmax removal')
    remove_logsoftmax(graph)

    print('Fix 2: Replace masked SE blocks')
    replace_masked_se_with_reducemean(graph)

    print('Fix 3: Replace ReduceMean with depthwise Conv (RKNN bug)')
    replace_reducemean_with_conv(model, shapes)

    print('Fix 4: Remove Squeeze/Unsqueeze (RKNN 4D↔3D bug)')
    remove_squeeze_unsqueeze(model)
    graph = model.graph

    fix_graph_outputs(graph)

    print(f'Final nodes: {len(graph.node)}')

    # Validate
    try:
        onnx.checker.check_model(model)
        print('ONNX validation passed')
    except Exception as e:
        print(f'ONNX validation warning: {e}')

    onnx.save(model, output_path)
    print(f'Saved: {output_path}')
