"""
nocache ONNX에서 cached_len/cached_avg 전용 노드 제거.
len/avg 업데이트를 CPU에서 처리하고 결과만 입력으로 제공.

분석 결과: 105 노드 (Unsqueeze:45, Gather:30, Cast:15, Mul:15)가
len/avg 입력에서만 도달 가능 → 제거 시 ~50 RKNN 레이어 감소 → ~1ms 절약.
"""
import onnx
from onnx import helper, TensorProto, shape_inference
import numpy as np
import copy
from collections import Counter

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
SRC = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
DST = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache-nolenavg.onnx'


def main():
    print("Loading nocache ONNX model...")
    m = onnx.load(SRC)
    g = m.graph

    # Build maps
    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n

    node_consumers = {}
    for n in g.node:
        for inp in n.input:
            if inp not in node_consumers:
                node_consumers[inp] = []
            node_consumers[inp].append(n)

    graph_input_names = {i.name for i in g.input}
    init_names = {i.name for i in g.initializer}
    lenavg_inputs = {i.name for i in g.input if 'cached_len' in i.name or 'cached_avg' in i.name}
    other_inputs = graph_input_names - lenavg_inputs - init_names

    print(f"len/avg inputs: {sorted(lenavg_inputs)}")
    print(f"Other inputs: {len(other_inputs)}")

    # Forward trace from len/avg inputs
    def forward_trace(start_tensors):
        """Find all nodes reachable from start_tensors via forward edges."""
        visited_tensors = set(start_tensors)
        reachable_nodes = set()
        queue = list(start_tensors)
        while queue:
            t = queue.pop(0)
            if t in node_consumers:
                for n in node_consumers[t]:
                    nid = id(n)
                    if nid not in reachable_nodes:
                        reachable_nodes.add(nid)
                        for o in n.output:
                            if o not in visited_tensors:
                                visited_tensors.add(o)
                                queue.append(o)
        return reachable_nodes

    # Forward trace from non-len/avg inputs (x, cached_key/val/conv)
    other_reachable = forward_trace(other_inputs | init_names)
    lenavg_reachable = forward_trace(lenavg_inputs)

    # Nodes ONLY reachable from len/avg (not from other inputs)
    lenavg_only = lenavg_reachable - other_reachable

    # Count ops
    ops = Counter()
    for n in g.node:
        if id(n) in lenavg_only:
            ops[n.op_type] += 1
    print(f"\nlen/avg-only nodes: {len(lenavg_only)}")
    print("Op breakdown:")
    for op, cnt in ops.most_common():
        print(f"  {op:20s} {cnt}")

    if len(lenavg_only) == 0:
        print("No len/avg-only nodes found. Nothing to remove.")
        return

    # Check what outputs these nodes produce
    lenavg_only_outputs = set()
    for n in g.node:
        if id(n) in lenavg_only:
            for o in n.output:
                lenavg_only_outputs.add(o)

    # Check if any graph outputs depend on lenavg-only nodes
    output_names = {o.name for o in g.output}
    affected_outputs = lenavg_only_outputs & output_names
    print(f"\nAffected graph outputs: {affected_outputs}")

    # Find boundary: outputs of lenavg-only nodes consumed by non-lenavg-only nodes
    boundary_tensors = set()
    for n in g.node:
        if id(n) in lenavg_only:
            for o in n.output:
                if o in node_consumers:
                    for consumer in node_consumers[o]:
                        if id(consumer) not in lenavg_only:
                            boundary_tensors.add(o)

    print(f"Boundary tensors (lenavg→other): {len(boundary_tensors)}")
    for bt in sorted(boundary_tensors):
        # Get shape info
        node = out2node[bt]
        print(f"  {bt} (from {node.op_type})")

    # These boundary tensors need to become new graph inputs
    # First, get shape info via shape inference
    print("\nRunning shape inference...")
    m_inferred = shape_inference.infer_shapes(m)
    vi_map = {vi.name: vi for vi in m_inferred.graph.value_info}
    for i in m_inferred.graph.input:
        vi_map[i.name] = i

    # Create new inputs for boundary tensors
    new_inputs_to_add = []
    for bt in sorted(boundary_tensors):
        if bt in vi_map:
            vi = copy.deepcopy(vi_map[bt])
            new_inputs_to_add.append(vi)
            print(f"  New input: {bt} (shape from inference)")
        else:
            # Fallback
            new_inputs_to_add.append(
                helper.make_tensor_value_info(bt, TensorProto.FLOAT, None))
            print(f"  New input: {bt} (no shape info)")

    # Remove lenavg-only nodes
    new_nodes = [n for n in g.node if id(n) not in lenavg_only]
    print(f"\nNodes: {len(g.node)} → {len(new_nodes)} (removed {len(g.node) - len(new_nodes)})")

    # Update inputs: remove lenavg inputs, add boundary inputs
    # But keep lenavg inputs if they're still consumed by remaining nodes
    remaining_input_needs = set()
    for n in new_nodes:
        for inp in n.input:
            if inp in graph_input_names:
                remaining_input_needs.add(inp)

    new_input_list = [i for i in g.input if i.name in remaining_input_needs]
    removed_inputs = set(i.name for i in g.input) - set(i.name for i in new_input_list)
    print(f"Removed inputs: {sorted(removed_inputs)}")

    # Add boundary tensor inputs
    new_input_list.extend(new_inputs_to_add)
    print(f"Inputs: {len(g.input)} → {len(new_input_list)}")

    # Keep outputs unchanged (remove any that were from lenavg-only)
    new_output_list = [o for o in g.output if o.name not in affected_outputs]
    if len(new_output_list) != len(g.output):
        print(f"Outputs: {len(g.output)} → {len(new_output_list)}")

    # Filter initializers
    remaining_tensors = set()
    for n in new_nodes:
        for inp in n.input:
            remaining_tensors.add(inp)
    new_inits = [i for i in g.initializer if i.name in remaining_tensors]

    new_graph = helper.make_graph(
        new_nodes, g.name + '_nolenavg',
        new_input_list, new_output_list,
        initializer=new_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=m.opset_import)
    new_model.ir_version = m.ir_version

    try:
        onnx.checker.check_model(new_model)
        print("\n✓ ONNX valid")
    except Exception as e:
        print(f"\n✗ Validation: {e}")

    onnx.save(new_model, DST)
    import os
    print(f"Saved: {DST} ({os.path.getsize(DST)/1024/1024:.1f} MB)")

    # ORT verification
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
        print(f"\n✓ ORT: {len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs")
        for o in sess.get_outputs()[:3]:
            print(f"  {o.name}: {o.shape}")
    except Exception as e:
        print(f"\n✗ ORT: {e}")


if __name__ == '__main__':
    main()
