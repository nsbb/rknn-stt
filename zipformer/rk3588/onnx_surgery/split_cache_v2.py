"""
Encoder ONNX에서 캐시 업데이트(Concat/Slice) 제거 v2.
각 new_cached_* Concat의 두 입력 중 'old cache slice' vs 'new computed'를 구분하여
new computed만 출력으로 노출.
"""
import onnx
from onnx import helper, TensorProto, shape_inference
import numpy as np
import copy
from collections import Counter

SRC = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix.onnx'
DST = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix-nocache.onnx'


def trace_back_to_inputs(tensor_name, out2node, graph_inputs, max_depth=50):
    """Trace backward from a tensor to find which graph inputs it depends on."""
    visited = set()
    queue = [(tensor_name, 0)]
    reached_inputs = set()
    while queue:
        t, depth = queue.pop(0)
        if t in visited or depth > max_depth:
            continue
        visited.add(t)
        if t in graph_inputs:
            reached_inputs.add(t)
            continue
        if t in out2node:
            n = out2node[t]
            for inp in n.input:
                queue.append((inp, depth + 1))
    return reached_inputs


def trace_op_path(tensor_name, out2node, max_depth=10):
    """Trace backward op types from a tensor."""
    path = []
    t = tensor_name
    for _ in range(max_depth):
        if t not in out2node:
            break
        n = out2node[t]
        path.append(n.op_type)
        if n.input:
            t = n.input[0]
        else:
            break
    return path


def main():
    print("Loading ONNX model...")
    m = onnx.load(SRC)
    g = m.graph

    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n

    graph_input_names = {i.name for i in g.input}
    init_names = {i.name for i in g.initializer}
    cache_input_names = {i.name for i in g.input if 'cached' in i.name}

    # Find cache output Concat nodes
    cache_outputs = []
    for o in g.output:
        if 'new_cached' in o.name:
            cache_outputs.append(o)

    print(f"Cache outputs: {len(cache_outputs)}")

    # For each cache Concat, identify which input is 'old slice' vs 'new computed'
    concat_to_remove = []
    slice_to_remove = []
    new_output_tensors = []  # (tensor_name, cache_output_name)

    for co in cache_outputs:
        concat_node = out2node[co.name]
        assert concat_node.op_type == 'Concat', f"{co.name}: expected Concat, got {concat_node.op_type}"

        # Check each input
        old_cache_input = None
        new_computed_input = None

        for ci in concat_node.input:
            # Trace back to see which graph inputs this depends on
            deps = trace_back_to_inputs(ci, out2node, graph_input_names)
            op_path = trace_op_path(ci, out2node)

            # If it primarily depends on a matching cached_* input, it's the old cache slice
            cache_deps = deps & cache_input_names
            non_cache_deps = deps - cache_input_names - init_names

            if cache_deps and not non_cache_deps:
                # Only depends on cached inputs → old cache slice
                old_cache_input = ci
            else:
                new_computed_input = ci

        if new_computed_input:
            new_output_tensors.append((new_computed_input, co.name))
            concat_to_remove.append(id(concat_node))
        else:
            print(f"  {co.name}: cannot separate (keeping)")

    print(f"\nIdentified {len(new_output_tensors)} cache outputs to replace")

    # Protect: nodes needed to produce boundary tensors must NOT be removed
    boundary_tensor_names = {src for src, _ in new_output_tensors}
    protected_nodes = set()
    def protect_producers(tensor_name):
        if tensor_name in out2node:
            n = out2node[tensor_name]
            nid = id(n)
            if nid not in protected_nodes:
                protected_nodes.add(nid)
                for inp in n.input:
                    protect_producers(inp)
    for bt in boundary_tensor_names:
        protect_producers(bt)
    print(f"Protected nodes (needed for boundary outputs): {len(protected_nodes)}")

    # Now trace the old_cache_input paths to find Slice nodes to remove
    node_consumers = {}
    for n in g.node:
        for inp in n.input:
            if inp not in node_consumers:
                node_consumers[inp] = []
            node_consumers[inp].append(id(n))

    nodes_to_remove = set(concat_to_remove)

    # BFS: remove nodes whose ONLY consumer is a removed node (but NOT protected)
    changed = True
    while changed:
        changed = False
        for n in g.node:
            nid = id(n)
            if nid in nodes_to_remove or nid in protected_nodes:
                continue
            all_outputs_consumed_by_removed = True
            has_consumers = False
            for o in n.output:
                if o in node_consumers:
                    consumers = node_consumers[o]
                    has_consumers = True
                    for cid in consumers:
                        if cid not in nodes_to_remove:
                            all_outputs_consumed_by_removed = False
                            break
                if not all_outputs_consumed_by_removed:
                    break

            if has_consumers and all_outputs_consumed_by_removed:
                nodes_to_remove.add(nid)
                changed = True

    print(f"Total nodes to remove: {len(nodes_to_remove)}")

    # Op breakdown of removed nodes
    removed_ops = Counter()
    for n in g.node:
        if id(n) in nodes_to_remove:
            removed_ops[n.op_type] += 1
    print("Removed ops:")
    for op, cnt in removed_ops.most_common():
        print(f"  {op:20s} {cnt}")

    # Build new model
    new_nodes = [n for n in g.node if id(n) not in nodes_to_remove]
    print(f"\nNodes: {len(g.node)} → {len(new_nodes)} (removed {len(g.node) - len(new_nodes)})")

    # Build new outputs
    m_inferred = shape_inference.infer_shapes(m)
    vi_map = {vi.name: vi for vi in m_inferred.graph.value_info}
    for i in m_inferred.graph.input:
        vi_map[i.name] = i

    new_output_list = []
    # Keep encoder_out
    for o in g.output:
        if o.name == 'encoder_out':
            new_output_list.append(o)

    # Add raw computed values as outputs
    for src_tensor, cache_name in sorted(new_output_tensors, key=lambda x: x[1]):
        if src_tensor in vi_map:
            vi = copy.deepcopy(vi_map[src_tensor])
            # Keep original tensor name (don't rename)
            new_output_list.append(vi)
        else:
            new_output_list.append(helper.make_tensor_value_info(
                src_tensor, TensorProto.FLOAT, None))

    print(f"Outputs: {len(g.output)} → {len(new_output_list)}")

    # Check which inputs are still needed
    remaining_tensors_needed = set()
    for n in new_nodes:
        for inp in n.input:
            remaining_tensors_needed.add(inp)

    new_input_list = [i for i in g.input if i.name in remaining_tensors_needed]
    removed_inputs = set(i.name for i in g.input) - set(i.name for i in new_input_list)
    print(f"Inputs: {len(g.input)} → {len(new_input_list)}")
    if removed_inputs:
        print(f"  Removed: {sorted(removed_inputs)}")

    # Filter initializers
    new_inits = [i for i in g.initializer if i.name in remaining_tensors_needed]

    new_graph = helper.make_graph(
        new_nodes, g.name + '_nocache',
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
    print(f"Saved: {DST}")

    # ORT verification
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
        print(f"\n✓ ORT loaded: {len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs")
        for o in sess.get_outputs()[:5]:
            print(f"  {o.name}: {o.shape}")
        if len(sess.get_outputs()) > 5:
            print(f"  ... (+{len(sess.get_outputs())-5} more)")
    except Exception as e:
        print(f"\n✗ ORT: {e}")

    # Print mapping for cache reconstruction
    print("\n=== Cache Reconstruction Map ===")
    for src, cache_name in sorted(new_output_tensors, key=lambda x: x[1]):
        # Determine which old cache input this corresponds to
        old_name = cache_name.replace('new_cached', 'cached')
        print(f"  {cache_name} = concat({old_name}[:, 1:], model_output['{src}'])")


if __name__ == '__main__':
    main()
