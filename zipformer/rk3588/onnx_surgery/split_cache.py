"""
Encoder ONNX에서 캐시 업데이트(Concat/Slice) 제거.
new_cached_* 출력 대신 '새로 계산된 값'만 출력으로 노출.
캐시 업데이트(old[:, 1:] + new)는 Python에서 수행.

목적: RKNN 레이어 수 감소 → dispatch overhead 감소
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import copy

SRC = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix.onnx'
DST = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix-nocache.onnx'


def main():
    print("Loading ONNX model...")
    m = onnx.load(SRC)
    g = m.graph

    # Build maps
    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n

    input_names = {i.name for i in g.input}
    init_names = {i.name for i in g.initializer}
    output_names = {o.name for o in g.output}

    # Identify cache outputs (new_cached_*)
    cache_output_names = {o.name for o in g.output if 'new_cached' in o.name}
    print(f"Cache outputs: {len(cache_output_names)}")

    # Trace back from encoder_out to find all needed nodes
    def trace_needed(start_tensors):
        visited = set()
        needed_nodes = set()
        queue = list(start_tensors)
        while queue:
            t = queue.pop(0)
            if t in visited:
                continue
            visited.add(t)
            if t in out2node:
                n = out2node[t]
                needed_nodes.add(id(n))
                for inp in n.input:
                    queue.append(inp)
        return needed_nodes

    enc_needed = trace_needed({'encoder_out'})

    # For each cache output, trace back and find the boundary tensor
    # The boundary is where the cache-only subgraph meets the encoder_out subgraph
    cache_only_ids = set()
    boundary_map = {}  # cache_output_name → boundary tensor name

    for cache_out in sorted(cache_output_names):
        # Trace back from this cache output
        cache_chain = trace_needed({cache_out})
        # Cache-only = nodes in cache_chain but not in enc_needed
        this_cache_only = cache_chain - enc_needed

        # Find boundary: inputs to cache-only nodes that come from enc-needed nodes
        for n in g.node:
            if id(n) in this_cache_only:
                for inp in n.input:
                    if inp in out2node and id(out2node[inp]) in enc_needed:
                        if cache_out not in boundary_map:
                            boundary_map[cache_out] = []
                        boundary_map[cache_out].append(inp)
                    elif inp in input_names:
                        if cache_out not in boundary_map:
                            boundary_map[cache_out] = []
                        boundary_map[cache_out].append(inp)

        cache_only_ids |= this_cache_only

    print(f"\nCache-only nodes: {len(cache_only_ids)}")
    print(f"Boundary mappings (cache_out → boundary tensors):")

    # For each cache output, the last Concat takes [old_cache_sliced, new_computed]
    # We want to expose new_computed as output instead
    new_outputs = []
    output_rename = {}  # old cache output name → new boundary output name

    for cache_out in sorted(cache_output_names):
        # The cache output is produced by a Concat node
        concat_node = out2node[cache_out]
        assert concat_node.op_type == 'Concat', f"Expected Concat, got {concat_node.op_type}"

        # Concat inputs: [old_cache_sliced, new_computed] or [new_computed, old_cache_sliced]
        # We need to find which input is "new computed" (from enc_needed) vs "old sliced" (from cache-only)
        new_value_input = None
        for ci in concat_node.input:
            if ci in out2node and id(out2node[ci]) in enc_needed:
                new_value_input = ci
                break
            elif ci in input_names:
                # Direct from input (e.g., cached_len which might just pass through)
                new_value_input = ci
                break

        if new_value_input is None:
            # Both inputs might be from cache-only chain, check deeper
            for ci in concat_node.input:
                if ci in out2node and id(out2node[ci]) not in cache_only_ids:
                    new_value_input = ci
                    break

        if new_value_input:
            new_name = cache_out.replace('new_cached', 'raw')
            output_rename[cache_out] = (new_value_input, new_name)
            print(f"  {cache_out} → {new_value_input} (as {new_name})")
        else:
            print(f"  {cache_out} → CANNOT FIND BOUNDARY (keeping original)")
            new_outputs.append(cache_out)

    # Build new graph: remove cache-only nodes, replace cache outputs with boundary tensors
    node_id_set = {id(n) for n in g.node}
    new_nodes = [n for n in g.node if id(n) not in cache_only_ids]

    print(f"\nOriginal nodes: {len(g.node)}")
    print(f"New nodes: {len(new_nodes)}")
    print(f"Removed: {len(g.node) - len(new_nodes)}")

    # Build new output list
    from onnx import shape_inference
    # First, get type info for boundary tensors
    m_inferred = shape_inference.infer_shapes(m)
    vi_map = {vi.name: vi for vi in m_inferred.graph.value_info}
    input_vi = {vi.name: vi for vi in m_inferred.graph.input}
    vi_map.update(input_vi)

    new_output_list = []
    # Keep encoder_out
    for o in g.output:
        if o.name == 'encoder_out':
            new_output_list.append(o)
            break

    # Add boundary tensors as new outputs
    for cache_out in sorted(output_rename.keys()):
        src_tensor, new_name = output_rename[cache_out]
        if src_tensor in vi_map:
            vi = copy.deepcopy(vi_map[src_tensor])
            vi.name = src_tensor  # Keep original name
            new_output_list.append(vi)
        else:
            # Fallback: create without type info
            new_output_list.append(helper.make_tensor_value_info(
                src_tensor, TensorProto.FLOAT, None))

    print(f"\nNew outputs: {len(new_output_list)}")
    for o in new_output_list[:5]:
        print(f"  {o.name}")
    if len(new_output_list) > 5:
        print(f"  ... ({len(new_output_list) - 5} more)")

    # Also need to keep inputs that are consumed by remaining nodes
    remaining_inputs_used = set()
    for n in new_nodes:
        for inp in n.input:
            if inp in input_names:
                remaining_inputs_used.add(inp)

    new_input_list = [i for i in g.input if i.name in remaining_inputs_used]
    removed_inputs = len(g.input) - len(new_input_list)
    print(f"\nInputs: {len(g.input)} → {len(new_input_list)} (removed {removed_inputs})")

    # Create new graph
    new_graph = helper.make_graph(
        new_nodes,
        g.name + '_nocache',
        new_input_list,
        new_output_list,
        initializer=g.initializer,
    )

    new_model = helper.make_model(new_graph, opset_imports=m.opset_import)
    new_model.ir_version = m.ir_version

    # Validate
    try:
        onnx.checker.check_model(new_model)
        print("\n✓ ONNX model valid")
    except Exception as e:
        print(f"\n✗ ONNX validation error: {e}")
        # Save anyway for debugging
        print("  Saving anyway for inspection...")

    onnx.save(new_model, DST)
    print(f"\nSaved: {DST}")

    # Verify with ORT
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
        print(f"\n✓ ORT can load the model")
        print(f"  Inputs: {len(sess.get_inputs())}")
        print(f"  Outputs: {len(sess.get_outputs())}")
        for o in sess.get_outputs()[:5]:
            print(f"    {o.name}: {o.shape}")
    except Exception as e:
        print(f"\n✗ ORT error: {e}")


if __name__ == '__main__':
    main()
