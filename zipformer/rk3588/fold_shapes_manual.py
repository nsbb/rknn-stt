"""
시도 12: Shape/Gather/Unsqueeze 체인을 수동으로 상수 폴딩.
onnxsim이 못 잡는 패턴을 직접 처리.

1. ORT로 모델 실행하여 모든 Shape/Gather/Unsqueeze 출력값 캡처
2. 해당 노드들을 Constant 초기화자로 교체
3. 사용되지 않는 노드 제거 (dead code elimination)
"""
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort
import copy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
INPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
OUTPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-noshape.onnx'

STATIC_SHAPES = {
    'x':              [1, 39, 80],
    'cached_len_0':   [2, 1],       'cached_len_1':   [4, 1],
    'cached_len_2':   [3, 1],       'cached_len_3':   [2, 1],
    'cached_len_4':   [4, 1],
    'cached_avg_0':   [2, 1, 384],  'cached_avg_1':   [4, 1, 384],
    'cached_avg_2':   [3, 1, 384],  'cached_avg_3':   [2, 1, 384],
    'cached_avg_4':   [4, 1, 384],
    'cached_key_0':   [2, 64, 1, 192],  'cached_key_1':   [4, 32, 1, 192],
    'cached_key_2':   [3, 16, 1, 192],  'cached_key_3':   [2,  8, 1, 192],
    'cached_key_4':   [4, 32, 1, 192],
    'cached_val_0':   [2, 64, 1, 96],   'cached_val_1':   [4, 32, 1, 96],
    'cached_val_2':   [3, 16, 1, 96],   'cached_val_3':   [2,  8, 1, 96],
    'cached_val_4':   [4, 32, 1, 96],
    'cached_val2_0':  [2, 64, 1, 96],   'cached_val2_1':  [4, 32, 1, 96],
    'cached_val2_2':  [3, 16, 1, 96],   'cached_val2_3':  [2,  8, 1, 96],
    'cached_val2_4':  [4, 32, 1, 96],
    'cached_conv1_0': [2, 1, 384, 30],  'cached_conv1_1': [4, 1, 384, 30],
    'cached_conv1_2': [3, 1, 384, 30],  'cached_conv1_3': [2, 1, 384, 30],
    'cached_conv1_4': [4, 1, 384, 30],
    'cached_conv2_0': [2, 1, 384, 30],  'cached_conv2_1': [4, 1, 384, 30],
    'cached_conv2_2': [3, 1, 384, 30],  'cached_conv2_3': [2, 1, 384, 30],
    'cached_conv2_4': [4, 1, 384, 30],
}

SHAPE_OPS = {'Shape', 'Gather', 'Unsqueeze', 'Squeeze', 'ConstantOfShape'}

def get_shape_subgraph_outputs(model):
    """Find outputs of shape computation subgraph that feed into compute ops."""
    nodes = list(model.graph.node)
    output_to_node = {}
    for n in nodes:
        for o in n.output:
            output_to_node[o] = n

    # BFS: mark all nodes reachable only through shape ops
    shape_outputs = set()  # outputs of shape-only subgraph
    for n in nodes:
        if n.op_type in SHAPE_OPS:
            for o in n.output:
                shape_outputs.add(o)
        # Concat with all-shape inputs
        if n.op_type == 'Concat':
            all_shape = all(
                (inp in output_to_node and output_to_node[inp].op_type in SHAPE_OPS | {'Concat'})
                or any(init.name == inp for init in model.graph.initializer)
                for inp in n.input
            )
            if all_shape:
                for o in n.output:
                    shape_outputs.add(o)

    return shape_outputs

def evaluate_tensors(model_path, tensor_names):
    """Run ORT and capture intermediate tensor values."""
    model = onnx.load(model_path)

    # Add requested tensors as outputs using value_info from shape inference
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)
    vi_map = {vi.name: vi for vi in model.graph.value_info}
    existing_outputs = {o.name for o in model.graph.output}
    added = []
    for name in tensor_names:
        if name not in existing_outputs:
            if name in vi_map:
                model.graph.output.append(copy.deepcopy(vi_map[name]))
            else:
                # Try INT64 as default for shape tensors
                model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.INT64, None))
            added.append(name)

    # Save temp model
    tmp_path = '/tmp/eval_model.onnx'
    onnx.save(model, tmp_path)

    sess = ort.InferenceSession(tmp_path, providers=['CPUExecutionProvider'])

    feeds = {}
    for name, shape in STATIC_SHAPES.items():
        dtype = np.int64 if 'cached_len' in name else np.float32
        feeds[name] = np.zeros(shape, dtype=dtype)

    # Get only the shape outputs
    out_names = [name for name in tensor_names if name in {o.name for o in sess.get_outputs()}]
    results = {}
    if out_names:
        vals = sess.run(out_names, feeds)
        for name, val in zip(out_names, vals):
            results[name] = val

    return results

def replace_with_constants(model, tensor_values):
    """Replace shape computation outputs with constant initializers."""
    nodes = list(model.graph.node)
    init_names = {init.name for init in model.graph.initializer}

    # Add tensor values as initializers
    for name, val in tensor_values.items():
        if name not in init_names:
            tensor = numpy_helper.from_array(val, name=name)
            model.graph.initializer.append(tensor)

    # Remove nodes whose outputs are now initializers
    tensor_value_names = set(tensor_values.keys())
    nodes_to_remove = set()
    for i, n in enumerate(nodes):
        if all(o in tensor_value_names for o in n.output):
            nodes_to_remove.add(i)

    # Iteratively remove dead nodes
    changed = True
    while changed:
        changed = False
        consumed = set()
        for i, n in enumerate(nodes):
            if i in nodes_to_remove:
                continue
            for inp in n.input:
                consumed.add(inp)
        # Also check graph outputs
        for o in model.graph.output:
            consumed.add(o.name)

        for i, n in enumerate(nodes):
            if i in nodes_to_remove:
                continue
            if n.op_type in SHAPE_OPS or n.op_type == 'Concat':
                if all(o not in consumed for o in n.output):
                    nodes_to_remove.add(i)
                    changed = True

    new_nodes = [n for i, n in enumerate(nodes) if i not in nodes_to_remove]
    print(f"Removed {len(nodes) - len(new_nodes)} nodes")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model

if __name__ == '__main__':
    print(f"Loading: {INPUT_ONNX}")
    model = onnx.load(INPUT_ONNX)
    print(f"Original nodes: {len(model.graph.node)}")

    # Step 1: Find shape subgraph outputs
    shape_outputs = get_shape_subgraph_outputs(model)
    print(f"Shape subgraph outputs: {len(shape_outputs)}")

    # Step 2: Evaluate tensors via ORT
    print("Evaluating shape tensors via ORT...")
    tensor_values = evaluate_tensors(INPUT_ONNX, shape_outputs)
    print(f"Evaluated {len(tensor_values)} tensors")

    # Step 3: Replace with constants
    print("Replacing with constants...")
    model = replace_with_constants(model, tensor_values)
    print(f"Nodes after folding: {len(model.graph.node)}")

    # Step 4: Run onnxsim to clean up
    onnx.save(model, OUTPUT_ONNX)
    print(f"Saved: {OUTPUT_ONNX}")

    # Distribution
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    print("\nOp distribution:")
    for op, cnt in ops.most_common():
        print(f'  {cnt:4d}  {op}')

    # Step 5: Fix input shapes to static
    print("\nFixing input shapes to static...")
    for inp in model.graph.input:
        if inp.name in STATIC_SHAPES:
            static_shape = STATIC_SHAPES[inp.name]
            for j, dim in enumerate(inp.type.tensor_type.shape.dim):
                dim.dim_value = static_shape[j]
                dim.ClearField('dim_param')

    onnx.save(model, OUTPUT_ONNX)
    print(f"Saved with static shapes: {OUTPUT_ONNX}")

    # Verify with ORT
    print("\nVerifying with ORT...")
    sess = ort.InferenceSession(OUTPUT_ONNX, providers=['CPUExecutionProvider'])
    feeds = {}
    for name, shape in STATIC_SHAPES.items():
        dtype = np.int64 if 'cached_len' in name else np.float32
        feeds[name] = np.random.randn(*shape).astype(dtype)
    try:
        out = sess.run(None, feeds)
        print(f"  Output shapes: {[o.shape for o in out[:3]]}")
        print("  Verification OK!")
    except Exception as e:
        print(f"  Verification FAILED: {e}")
