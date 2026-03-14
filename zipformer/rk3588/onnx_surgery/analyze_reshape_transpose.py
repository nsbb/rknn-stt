"""
Analyze Reshape/Transpose patterns in the cumfix ONNX model.
Goal: identify removable ops to reduce RKNN layer count.
"""
import onnx
from collections import Counter, defaultdict

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix.onnx'

def analyze():
    m = onnx.load(MODEL)
    g = m.graph

    # Count op types
    op_counts = Counter(n.op_type for n in g.node)
    print("=== Op type counts ===")
    for op, cnt in op_counts.most_common(30):
        print(f"  {op}: {cnt}")

    print(f"\nTotal nodes: {len(g.node)}")
    print(f"Reshape: {op_counts.get('Reshape', 0)}")
    print(f"Transpose: {op_counts.get('Transpose', 0)}")
    print(f"Unsqueeze: {op_counts.get('Unsqueeze', 0)}")
    print(f"Squeeze: {op_counts.get('Squeeze', 0)}")

    # Build output->node map
    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n

    # Analyze Reshape patterns
    print("\n=== Reshape patterns ===")
    reshape_patterns = Counter()
    for n in g.node:
        if n.op_type == 'Reshape':
            # Check consumer
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            consumer_ops = tuple(sorted(set(nn.op_type for nn in consumers)))
            # Check producer
            producer = out2node.get(n.input[0])
            producer_op = producer.op_type if producer else 'INPUT'
            reshape_patterns[(producer_op, consumer_ops)] += 1

    for (prod, cons), cnt in reshape_patterns.most_common(20):
        print(f"  {prod} -> Reshape -> {cons}: {cnt}")

    # Analyze Transpose patterns
    print("\n=== Transpose patterns ===")
    transpose_patterns = Counter()
    for n in g.node:
        if n.op_type == 'Transpose':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            consumer_ops = tuple(sorted(set(nn.op_type for nn in consumers)))
            producer = out2node.get(n.input[0])
            producer_op = producer.op_type if producer else 'INPUT'
            perm = tuple(n.attribute[0].ints) if n.attribute else ()
            transpose_patterns[(producer_op, perm, consumer_ops)] += 1

    for (prod, perm, cons), cnt in transpose_patterns.most_common(20):
        print(f"  {prod} -> Transpose{list(perm)} -> {cons}: {cnt}")

    # Analyze Reshape-Transpose chains
    print("\n=== Reshape-Transpose chains ===")
    chains = 0
    for n in g.node:
        if n.op_type == 'Reshape':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            for c in consumers:
                if c.op_type == 'Transpose':
                    chains += 1
    print(f"  Reshape -> Transpose chains: {chains}")

    chains2 = 0
    for n in g.node:
        if n.op_type == 'Transpose':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            for c in consumers:
                if c.op_type == 'Reshape':
                    chains2 += 1
    print(f"  Transpose -> Reshape chains: {chains2}")

    # Consecutive Reshape
    consec = 0
    for n in g.node:
        if n.op_type == 'Reshape':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            for c in consumers:
                if c.op_type == 'Reshape':
                    consec += 1
    print(f"  Reshape -> Reshape chains: {consec}")

    # Analyze Unsqueeze/Squeeze patterns
    print("\n=== Unsqueeze/Squeeze patterns ===")
    unsq_sq = 0
    for n in g.node:
        if n.op_type == 'Unsqueeze':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            for c in consumers:
                if c.op_type == 'Squeeze':
                    unsq_sq += 1
    sq_unsq = 0
    for n in g.node:
        if n.op_type == 'Squeeze':
            consumers = [nn for nn in g.node if any(i in n.output for i in nn.input)]
            for c in consumers:
                if c.op_type == 'Unsqueeze':
                    sq_unsq += 1
    print(f"  Unsqueeze -> Squeeze: {unsq_sq}")
    print(f"  Squeeze -> Unsqueeze: {sq_unsq}")

if __name__ == '__main__':
    analyze()
