"""
시도 12: Shape/Gather/Unsqueeze/Concat 체인을 상수로 폴딩.
정적 입력 shape에서 동적 shape 연산은 불필요 → 상수로 대체하면 ONNX 노드 ~350개 제거 가능.
RKNN 컴파일 시 레이어 수 감소 기대.
"""
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort
import copy

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
INPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-sim.onnx'
OUTPUT_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-folded.onnx'

# Static shapes for all inputs (from ENC_SCHEMA)
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

def fold_constant_shapes(model):
    """Use onnxruntime to evaluate all Shape/Gather/Unsqueeze/Concat chains."""
    # Create dummy inputs with static shapes
    feeds = {}
    for inp in model.graph.input:
        name = inp.name
        if name in STATIC_SHAPES:
            shape = STATIC_SHAPES[name]
            # Determine dtype
            elem_type = inp.type.tensor_type.elem_type
            if elem_type == TensorProto.INT64:
                feeds[name] = np.zeros(shape, dtype=np.int64)
            else:
                feeds[name] = np.zeros(shape, dtype=np.float32)

    # Run shape inference
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)

    # Find all outputs of Shape/Gather/Unsqueeze nodes that feed into Reshape/Concat
    nodes = list(model.graph.node)
    shape_ops = {'Shape', 'Gather', 'Unsqueeze', 'Squeeze', 'ConstantOfShape'}

    # Identify shape-computation subgraph outputs used by compute ops
    output_to_node = {}
    for n in nodes:
        for o in n.output:
            output_to_node[o] = n

    # Use ORT to compute intermediate values
    # Add all intermediate shape tensors as outputs
    shape_tensor_names = set()
    for n in nodes:
        if n.op_type in shape_ops:
            for o in n.output:
                shape_tensor_names.add(o)

    # Also add Concat outputs that are used as Reshape shapes
    for n in nodes:
        if n.op_type == 'Concat':
            # Check if all inputs are from shape ops or constants
            all_shape = True
            for inp in n.input:
                if inp in output_to_node:
                    if output_to_node[inp].op_type not in shape_ops and output_to_node[inp].op_type != 'Concat':
                        all_shape = False
                        break
                # Check if it's an initializer
                elif any(init.name == inp for init in model.graph.initializer):
                    pass  # initializer is fine
                else:
                    all_shape = False
                    break
            if all_shape:
                for o in n.output:
                    shape_tensor_names.add(o)

    print(f"Shape tensor candidates: {len(shape_tensor_names)}")

    # Run ORT to get values
    sess = ort.InferenceSession(INPUT_ONNX, providers=['CPUExecutionProvider'])

    # Can't easily add intermediate outputs. Instead, use a different approach:
    # Iterate through model, find Reshape nodes whose shape input comes from
    # Shape/Gather/Unsqueeze/Concat chain, and replace with constant shape.

    # For each Reshape node, try to statically determine its shape
    reshape_count = 0
    replaced = 0

    for i, n in enumerate(nodes):
        if n.op_type == 'Reshape':
            shape_input = n.input[1]
            # Check if shape_input is already a constant
            is_init = any(init.name == shape_input for init in model.graph.initializer)
            if is_init:
                continue

            # Check if it comes from shape computation
            if shape_input in output_to_node:
                src = output_to_node[shape_input]
                if src.op_type in shape_ops or src.op_type == 'Concat':
                    reshape_count += 1

    print(f"Reshape nodes with computed shapes: {reshape_count}")
    return model

if __name__ == '__main__':
    print(f"Loading: {INPUT_ONNX}")
    model = onnx.load(INPUT_ONNX)
    print(f"Original nodes: {len(model.graph.node)}")

    # Approach: Use onnxsim with static input shapes to fold constants
    import subprocess
    import sys

    # Use onnxsim with input shapes to fold all shape computations
    # onnxsim can fold constants when shapes are known
    print("\nUsing onnxsim with static shapes to fold shape computations...")

    # Write input shapes as overrides
    shape_overrides = []
    for name, shape in STATIC_SHAPES.items():
        shape_str = ','.join(str(s) for s in shape)
        shape_overrides.extend(['--input-shape', f'{name}:{shape_str}'])

    cmd = ['python', '-m', 'onnxsim', INPUT_ONNX, OUTPUT_ONNX] + shape_overrides
    print(f"Running: {' '.join(cmd[:5])} ... ({len(shape_overrides)//2} shape overrides)")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}")

    if result.returncode == 0:
        m2 = onnx.load(OUTPUT_ONNX)
        print(f"\nFolded nodes: {len(m2.graph.node)}")
        from collections import Counter
        ops = Counter(n.op_type for n in m2.graph.node)
        print("Op distribution:")
        for op, cnt in ops.most_common():
            print(f'  {cnt:4d}  {op}')
    else:
        fold_constant_shapes(model)
