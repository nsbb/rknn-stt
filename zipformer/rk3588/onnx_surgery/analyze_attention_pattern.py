"""
Analyze attention Reshape/Transpose patterns in detail.
Goal: understand which ops can be eliminated by weight matrix restructuring.

Common attention pattern:
  x -> Linear -> Reshape[B,T,H,D] -> Transpose[0,2,1,3] -> MatMul(Q@K^T) -> ...

If we restructure the weight matrix to output [H,B,T,D] directly,
we can skip the Reshape+Transpose.
"""
import onnx
import numpy as np
from collections import defaultdict

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix-sim.onnx'


def analyze_attention():
    m = onnx.load(MODEL)
    g = m.graph

    out2node = {}
    in2consumers = defaultdict(list)
    for n in g.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2consumers[inp].append(n)

    # Find MatMul -> Reshape -> Transpose patterns (attention head split)
    print("=== MatMul -> Reshape -> Transpose (head split) ===")
    head_split_count = 0
    for n in g.node:
        if n.op_type != 'MatMul':
            continue
        consumers = in2consumers.get(n.output[0], [])
        for c in consumers:
            if c.op_type == 'Reshape':
                for c2 in in2consumers.get(c.output[0], []):
                    if c2.op_type == 'Transpose':
                        perm = list(c2.attribute[0].ints) if c2.attribute else []
                        head_split_count += 1
                        if head_split_count <= 3:
                            print(f"  {n.name} -> {c.name} -> {c2.name} perm={perm}")

    print(f"  Total: {head_split_count}")

    # Find Transpose -> Reshape -> MatMul patterns (head merge)
    print("\n=== Transpose -> Reshape -> MatMul (head merge) ===")
    head_merge_count = 0
    for n in g.node:
        if n.op_type != 'Transpose':
            continue
        consumers = in2consumers.get(n.output[0], [])
        for c in consumers:
            if c.op_type == 'Reshape':
                for c2 in in2consumers.get(c.output[0], []):
                    if c2.op_type == 'MatMul':
                        perm = list(n.attribute[0].ints) if n.attribute else []
                        head_merge_count += 1
                        if head_merge_count <= 3:
                            print(f"  {n.name} perm={perm} -> {c.name} -> {c2.name}")

    print(f"  Total: {head_merge_count}")

    # Find Conv-related Transpose patterns
    print("\n=== Transpose around Conv ===")
    conv_transpose_count = 0
    for n in g.node:
        if n.op_type != 'Conv':
            continue
        # Check input Transpose
        producer = out2node.get(n.input[0])
        has_input_transpose = producer and producer.op_type == 'Transpose'
        # Check output Transpose
        out_transposes = [c for c in in2consumers.get(n.output[0], []) if c.op_type == 'Transpose']
        if has_input_transpose or out_transposes:
            conv_transpose_count += 1
            if conv_transpose_count <= 3:
                in_perm = list(producer.attribute[0].ints) if has_input_transpose else 'none'
                out_perm = [list(t.attribute[0].ints) for t in out_transposes] if out_transposes else 'none'
                print(f"  Conv {n.name}: in_transpose={in_perm}, out_transpose={out_perm}")

    print(f"  Total Conv with Transpose: {conv_transpose_count}")
    print(f"  Total Conv: {sum(1 for n in g.node if n.op_type == 'Conv')}")

    # Find Slice -> Reshape -> Transpose patterns (cache-related)
    print("\n=== Slice -> Reshape -> Transpose (cache indexing) ===")
    slice_reshape_count = 0
    for n in g.node:
        if n.op_type != 'Slice':
            continue
        consumers = in2consumers.get(n.output[0], [])
        for c in consumers:
            if c.op_type == 'Reshape':
                for c2 in in2consumers.get(c.output[0], []):
                    if c2.op_type == 'Transpose':
                        slice_reshape_count += 1

    print(f"  Total: {slice_reshape_count}")

    # Find Concat -> Reshape -> Transpose patterns
    print("\n=== Concat -> Reshape -> Transpose (concat + reshape) ===")
    concat_reshape_count = 0
    for n in g.node:
        if n.op_type != 'Concat':
            continue
        consumers = in2consumers.get(n.output[0], [])
        for c in consumers:
            if c.op_type == 'Reshape':
                for c2 in in2consumers.get(c.output[0], []):
                    if c2.op_type == 'Transpose':
                        concat_reshape_count += 1

    print(f"  Total: {concat_reshape_count}")

    # Summary
    print("\n=== Summary ===")
    print(f"  Head split (MatMul->Reshape->Transpose): {head_split_count}")
    print(f"  Head merge (Transpose->Reshape->MatMul): {head_merge_count}")
    print(f"  Conv transpose wrapping: {conv_transpose_count}")
    print(f"  Cache Slice->Reshape->Transpose: {slice_reshape_count}")
    print(f"  Concat->Reshape->Transpose: {concat_reshape_count}")
    total_removable = head_split_count * 2 + head_merge_count * 2 + conv_transpose_count * 2 + slice_reshape_count * 2 + concat_reshape_count * 2
    print(f"  Potentially removable ops (approx): {total_removable}")
    print(f"  Current Reshape+Transpose: {sum(1 for n in g.node if n.op_type in ('Reshape', 'Transpose'))}")


if __name__ == '__main__':
    analyze_attention()
