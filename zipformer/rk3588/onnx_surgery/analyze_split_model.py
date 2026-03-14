"""
Analyze if splitting the encoder into 2 parts could help speed.

The zipformer encoder has 5 layers (num_encoder_layers).
Each layer has attention + convolution modules.

If we split into:
  Part A: encoder_embed + layers 0-2 (compute-heavy, uses all attention caches)
  Part B: layers 3-4 (lighter, fewer caches)

Benefits:
  - Each part has fewer ops → fewer RKNN layers → less dispatch overhead
  - Part B only needs a subset of cache tensors → less I/O

We need to analyze:
1. Which cache tensors belong to which layer
2. Size of intermediate activation between parts
3. Whether the total rknn_run of both parts < single model rknn_run
"""
import onnx
from collections import defaultdict

MODEL = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix-sim.onnx'

m = onnx.load(MODEL)
g = m.graph

# Analyze cache tensor sizes per layer
from convert_encoder_int8_optarget import ENC_SCHEMA

print("=== Cache tensors per layer ===")
layer_sizes = defaultdict(int)
for name, shape, dtype in ENC_SCHEMA:
    if name == 'x':
        continue
    # Extract layer index from name (e.g., cached_key_0 → layer 0)
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        layer_idx = int(parts[1])
        size = 1
        for d in shape:
            size *= d
        size *= 4 if dtype == 'float32' else 8
        layer_sizes[layer_idx] += size

total_cache = sum(layer_sizes.values())
print(f"Total cache size: {total_cache/1024:.1f} KB")
for layer_idx in sorted(layer_sizes):
    print(f"  Layer {layer_idx}: {layer_sizes[layer_idx]/1024:.1f} KB ({layer_sizes[layer_idx]/total_cache*100:.1f}%)")

# Count ops per layer
print("\n=== Ops in model (by layer prefix) ===")
layer_ops = defaultdict(int)
for n in g.node:
    # Try to identify layer from node name
    name = n.name
    for i in range(5):
        if f'_{i}/' in name or name.endswith(f'_{i}') or f'layers.{i}' in name:
            layer_ops[i] += 1
            break
    else:
        if 'encoder_embed' in name or 'embed' in name.lower():
            layer_ops['embed'] += 1
        elif 'cumfix' in name:
            layer_ops['cumfix'] += 1
        else:
            layer_ops['other'] += 1

print(f"Total nodes: {len(g.node)}")
for k in sorted(layer_ops, key=str):
    print(f"  {'Layer ' + str(k) if isinstance(k, int) else k}: {layer_ops[k]}")

# Analyze split points
print("\n=== Split analysis ===")
# If we split layers 0-2 | 3-4:
cache_a = sum(layer_sizes[i] for i in range(3))
cache_b = sum(layer_sizes[i] for i in range(3, 5))
ops_a = sum(layer_ops.get(i, 0) for i in range(3)) + layer_ops.get('embed', 0) + layer_ops.get('cumfix', 0)
ops_b = sum(layer_ops.get(i, 0) for i in range(3, 5))
ops_other = layer_ops.get('other', 0)

print(f"Split A (embed + layers 0-2): {ops_a} ops, {cache_a/1024:.1f} KB cache")
print(f"Split B (layers 3-4): {ops_b} ops, {cache_b/1024:.1f} KB cache")
print(f"Other/shared ops: {ops_other}")
print(f"\nEstimated rknn_run reduction:")
print(f"  Current single model: ~31ms")
print(f"  If dispatch overhead scales with op count:")
ratio_a = ops_a / (ops_a + ops_b + ops_other)
ratio_b = ops_b / (ops_a + ops_b + ops_other)
print(f"  Part A: ~{31 * ratio_a:.1f}ms ({ratio_a*100:.0f}% of ops)")
print(f"  Part B: ~{31 * ratio_b:.1f}ms ({ratio_b*100:.0f}% of ops)")
print(f"  Sequential total: ~{31 * (ratio_a + ratio_b):.1f}ms + transition overhead")
print(f"  Note: dispatch overhead is per-RKNN-layer, not per-ONNX-op")
