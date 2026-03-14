"""
ONNX simplifier로 encoder 그래프 최적화.
목표: 불필요한 Reshape(181개)/Transpose(202개) 제거 → RKNN 15ms 절감.
"""
import onnx
from onnxsim import simplify
from collections import Counter
import sys

SRC = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix.onnx'
DST = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix-sim.onnx'

print("Loading cumfix model...")
m = onnx.load(SRC)
ops = Counter(n.op_type for n in m.graph.node)
print(f"Before: {len(m.graph.node)} nodes")
for o, c in ops.most_common(15):
    print(f"  {o}: {c}")

print("\nSimplifying...")
try:
    m2, ok = simplify(m)
except Exception as e:
    print(f"simplify() failed: {e}")
    print("\nTrying with skip_shape_inference...")
    m2, ok = simplify(m, skip_shape_inference=True)

print(f"Success: {ok}")
ops2 = Counter(n.op_type for n in m2.graph.node)
print(f"After: {len(m2.graph.node)} nodes")
for o, c in ops2.most_common(15):
    print(f"  {o}: {c}")

print("\nChanges:")
all_ops = sorted(set(list(ops.keys()) + list(ops2.keys())))
for o in all_ops:
    b = ops.get(o, 0)
    a = ops2.get(o, 0)
    if b != a:
        print(f"  {o}: {b} -> {a} ({a-b:+d})")

total_before = len(m.graph.node)
total_after = len(m2.graph.node)
print(f"\nTotal: {total_before} -> {total_after} ({total_after-total_before:+d}, {(total_after-total_before)/total_before*100:.1f}%)")

onnx.save(m2, DST)
print(f"Saved to {DST}")

# Also try onnxoptimizer
print("\n--- Also trying onnxoptimizer ---")
try:
    import onnxoptimizer
    passes = onnxoptimizer.get_available_passes()
    print(f"Available passes: {len(passes)}")
    m3 = onnxoptimizer.optimize(m2, passes)
    ops3 = Counter(n.op_type for n in m3.graph.node)
    print(f"After optimizer: {len(m3.graph.node)} nodes")
    if len(m3.graph.node) < len(m2.graph.node):
        DST2 = DST.replace('.onnx', '-opt.onnx')
        onnx.save(m3, DST2)
        print(f"Saved to {DST2}")
    else:
        print("No additional reduction from onnxoptimizer")
except Exception as e:
    print(f"onnxoptimizer failed: {e}")
