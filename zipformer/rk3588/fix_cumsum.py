"""
CumSum → MatMul 패치
RKNN CumSum 버그(non-zero 초기 cached_avg → chunk 1+에서 발산)를 우회.

각 CumSum(x, axis=0) — x shape [T, 1, 384] — 을 다음으로 교체:
  Reshape(x, [T, 384])
  MatMul(L_T, x_2d)        L_T: 하삼각 all-ones [T, T]
  Reshape(matmul_out, [T, 1, 384])

이 변환은 정확히 동등 (exclusive=0, reverse=0 기본값 기준).
"""
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

SRC = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1.onnx'
DST = '/home/rk3588/travail/rk3588/rknn-stt/zipformer/encoder-epoch-99-avg-1-cumfix.onnx'

# CumSum input shapes (from profiling)
CS_SHAPES = [16,16,8,8,8,8,4,4,4,2,2,8,8,8,8]  # T per CumSum_i

model = onnx.load(SRC)
graph = model.graph

# ── 1. CumSum 노드와 전용 Constant axis 노드 색인 ──────────────────────────
cs_nodes = [n for n in graph.node if n.op_type == 'CumSum']
assert len(cs_nodes) == len(CS_SHAPES), f"Expected {len(CS_SHAPES)} CumSum nodes, got {len(cs_nodes)}"

# axis constant 노드 이름 수집 (CumSum 전용인 것만)
axis_const_outputs = {n.input[1] for n in cs_nodes}
in2nodes = {}
for n in graph.node:
    for inp in n.input:
        in2nodes.setdefault(inp, []).append(n)
axis_only_nodes = set()
for n in graph.node:
    if n.op_type == 'Constant' and len(n.output) == 1 and n.output[0] in axis_const_outputs:
        users = in2nodes.get(n.output[0], [])
        if all(u.op_type == 'CumSum' for u in users):
            axis_only_nodes.add(n.name)

# ── 2. 새 이니셜라이저 및 노드 빌드 ──────────────────────────────────────
new_initializers = []
new_nodes_for_cs = {}  # cs_node.name → [replacement_nodes]

unique_Ts = set(CS_SHAPES)
L_init_names = {}
for T in unique_Ts:
    L = np.tril(np.ones((T, T), dtype=np.float32))
    name = f'_cumfix_L{T}'
    new_initializers.append(numpy_helper.from_array(L, name=name))
    L_init_names[T] = name
    # shape constants
    sh2d = np.array([T, 384], dtype=np.int64)
    sh3d = np.array([T, 1, 384], dtype=np.int64)
    new_initializers.append(numpy_helper.from_array(sh2d, name=f'_cumfix_sh2d_T{T}'))
    new_initializers.append(numpy_helper.from_array(sh3d, name=f'_cumfix_sh3d_T{T}'))

cs_name_set = {n.name for n in cs_nodes}

for i, cs in enumerate(cs_nodes):
    T = CS_SHAPES[i]
    in0 = cs.input[0]   # [T, 1, 384]
    out = cs.output[0]  # [T, 1, 384]
    uid = f'cs{i}'

    r1 = helper.make_node('Reshape',
        inputs=[in0, f'_cumfix_sh2d_T{T}'],
        outputs=[f'_cumfix_{uid}_flat'],
        name=f'_cumfix_{uid}_r1')

    mm = helper.make_node('MatMul',
        inputs=[L_init_names[T], f'_cumfix_{uid}_flat'],
        outputs=[f'_cumfix_{uid}_mm'],
        name=f'_cumfix_{uid}_mm')

    r2 = helper.make_node('Reshape',
        inputs=[f'_cumfix_{uid}_mm', f'_cumfix_sh3d_T{T}'],
        outputs=[out],
        name=f'_cumfix_{uid}_r2')

    new_nodes_for_cs[cs.name] = [r1, mm, r2]

# ── 3. 그래프 재조립 ──────────────────────────────────────────────────────
final_nodes = []
for n in graph.node:
    if n.name in cs_name_set:
        final_nodes.extend(new_nodes_for_cs[n.name])
    elif n.name in axis_only_nodes:
        pass  # 제거
    else:
        final_nodes.append(n)

# 기존 노드 교체
del graph.node[:]
graph.node.extend(final_nodes)
graph.initializer.extend(new_initializers)

onnx.save(model, DST)
print(f"Saved → {DST}")
print(f"  Replaced {len(cs_nodes)} CumSum nodes with MatMul chains")
print(f"  Removed  {len(axis_only_nodes)} axis Constant nodes")
print(f"  Added    {len(new_initializers)} initializers (L matrices + shape tensors)")
