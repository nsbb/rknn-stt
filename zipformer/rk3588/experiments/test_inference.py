"""
Zipformer RKNN 추론 테스트 (rknnlite 사용 — 실제 NPU)
- encoder: 상수 출력 확인, ONNX 비교, 레이턴시 측정
- decoder / joiner 포함 전체 파이프라인
"""
import numpy as np
import os, time

BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

try:
    from rknnlite.api import RKNNLite
    NPU_CORE_0 = RKNNLite.NPU_CORE_0
    def load_rknn_model(path):
        m = RKNNLite(verbose=False)
        ret = m.load_rknn(path)
        if ret != 0: raise RuntimeError(f"load_rknn failed: {ret}")
        ret = m.init_runtime(core_mask=NPU_CORE_0)
        if ret != 0: raise RuntimeError(f"init_runtime failed: {ret}")
        return m
    print("Runtime: rknnlite (실제 NPU)")
except ImportError:
    from rknn.api import RKNN as RKNNLite
    NPU_CORE_0 = None
    def load_rknn_model(path):
        m = RKNNLite(verbose=False)
        ret = m.load_rknn(path)
        if ret != 0: raise RuntimeError(f"load_rknn failed: {ret}")
        ret = m.init_runtime()  # simulator
        if ret != 0: raise RuntimeError(f"init_runtime failed: {ret}")
        return m
    print("Runtime: rknn simulator (NPU 없음)")

# ─── Encoder 입력 정의 ─────────────────────────────────────────
ENC_INPUTS = [
    ('x',              [1, 39, 80],       'float32'),
    ('cached_len_0',   [2, 1],            'int64'),
    ('cached_len_1',   [4, 1],            'int64'),
    ('cached_len_2',   [3, 1],            'int64'),
    ('cached_len_3',   [2, 1],            'int64'),
    ('cached_len_4',   [4, 1],            'int64'),
    ('cached_avg_0',   [2, 1, 384],       'float32'),
    ('cached_avg_1',   [4, 1, 384],       'float32'),
    ('cached_avg_2',   [3, 1, 384],       'float32'),
    ('cached_avg_3',   [2, 1, 384],       'float32'),
    ('cached_avg_4',   [4, 1, 384],       'float32'),
    ('cached_key_0',   [2, 64, 1, 192],   'float32'),
    ('cached_key_1',   [4, 32, 1, 192],   'float32'),
    ('cached_key_2',   [3, 16, 1, 192],   'float32'),
    ('cached_key_3',   [2,  8, 1, 192],   'float32'),
    ('cached_key_4',   [4, 32, 1, 192],   'float32'),
    ('cached_val_0',   [2, 64, 1, 96],    'float32'),
    ('cached_val_1',   [4, 32, 1, 96],    'float32'),
    ('cached_val_2',   [3, 16, 1, 96],    'float32'),
    ('cached_val_3',   [2,  8, 1, 96],    'float32'),
    ('cached_val_4',   [4, 32, 1, 96],    'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],    'float32'),
    ('cached_val2_1',  [4, 32, 1, 96],    'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],    'float32'),
    ('cached_val2_3',  [2,  8, 1, 96],    'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],    'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],   'float32'),
    ('cached_conv1_1', [4, 1, 384, 30],   'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],   'float32'),
    ('cached_conv1_3', [2, 1, 384, 30],   'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],   'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],   'float32'),
    ('cached_conv2_1', [4, 1, 384, 30],   'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],   'float32'),
    ('cached_conv2_3', [2, 1, 384, 30],   'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],   'float32'),
]

def nchw_to_nhwc(a): return np.transpose(a, (0, 2, 3, 1))
def nhwc_to_nchw(a): return np.transpose(a, (0, 3, 1, 2))

def make_rknn_inputs(state):
    """RKNN 인퍼런스용 입력 리스트 (4D → NHWC)"""
    out = []
    for name, shape, _ in ENC_INPUTS:
        arr = state[name]
        if len(shape) == 4:
            arr = nchw_to_nhwc(arr)
        out.append(arr)
    return out

def init_state():
    state = {}
    for name, shape, dtype in ENC_INPUTS:
        state[name] = np.zeros(shape, dtype=np.dtype(dtype))
    return state

def update_state_from_rknn_out(state, rknn_out):
    """
    RKNN 출력으로 캐시 상태 업데이트
    출력 순서: encoder_out(0), new_cached_len_0..4(1-5), new_cached_avg_0..4(6-10),
               new_cached_key_0..4(11-15), new_cached_val_0..4(16-20),
               new_cached_val2_0..4(21-25), new_cached_conv1_0..4(26-30),
               new_cached_conv2_0..4(31-35)
    """
    cache_keys = [name for name, _, _ in ENC_INPUTS if name != 'x']
    for idx, name in enumerate(cache_keys):
        out_arr = np.array(rknn_out[idx + 1])  # +1 for encoder_out at idx 0
        shape = [s for n, s, _ in ENC_INPUTS if n == name][0]
        # 4D 출력은 NHWC → NCHW
        if len(shape) == 4:
            out_arr = nhwc_to_nchw(out_arr)
        state[name] = out_arr


# ─── TEST 1: 상수 출력 확인 ────────────────────────────────────
def test_constant_check():
    print("\n=== [TEST 1] Encoder 상수 출력 확인 ===")
    enc = load_rknn_model(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')

    st_zero = init_state()
    in_zero = make_rknn_inputs(st_zero)

    st_rand = init_state()
    st_rand['x'] = (np.random.randn(1, 39, 80) * 0.3).astype(np.float32)
    in_rand = make_rknn_inputs(st_rand)

    t0 = time.time()
    out1 = enc.inference(inputs=in_zero)
    t1 = time.time()
    out2 = enc.inference(inputs=in_rand)
    t2 = time.time()

    e1 = np.array(out1[0])
    e2 = np.array(out2[0])

    is_const = np.allclose(e1, e2, atol=1e-5)
    max_diff  = np.abs(e1 - e2).max()
    lat_ms    = (t1 - t0) * 1000

    print(f"  encoder_out shape:          {e1.shape}")
    print(f"  상수 출력?                  {is_const}  ({'⚠ 문제!' if is_const else '✓ OK — 입력에 반응함'})")
    print(f"  max_diff (zeros vs random): {max_diff:.6f}")
    print(f"  추론 레이턴시 (NPU):        {lat_ms:.1f} ms")
    print(f"  두 번째 추론 레이턴시:      {(t2 - t1)*1000:.1f} ms")

    enc.release()
    return not is_const


# ─── TEST 2: RKNN vs ONNX 출력 비교 ──────────────────────────
def test_rknn_vs_onnx():
    import onnxruntime as ort
    print("\n=== [TEST 2] Encoder RKNN vs ONNX 비교 ===")
    np.random.seed(42)

    state = init_state()
    state['x'] = (np.random.randn(1, 39, 80) * 0.3).astype(np.float32)
    for name, shape, dtype in ENC_INPUTS:
        if 'cached_len' not in name and name != 'x':
            state[name] = (np.random.randn(*shape) * 0.01).astype(np.float32)

    # ONNX 추론
    sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
    onnx_in = {name: state[name] for name, _, _ in ENC_INPUTS}
    t_onnx_0 = time.time()
    onnx_out = sess.run(None, onnx_in)
    t_onnx_1 = time.time()
    onnx_enc = np.array(onnx_out[0])

    # RKNN 추론
    enc = load_rknn_model(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
    rknn_in = make_rknn_inputs(state)
    t_rknn_0 = time.time()
    rknn_out = enc.inference(inputs=rknn_in)
    t_rknn_1 = time.time()
    rknn_enc = np.array(rknn_out[0])
    enc.release()

    print(f"  ONNX encoder_out shape: {onnx_enc.shape}  ({(t_onnx_1-t_onnx_0)*1000:.0f} ms)")
    print(f"  RKNN encoder_out shape: {rknn_enc.shape}  ({(t_rknn_1-t_rknn_0)*1000:.0f} ms)")

    if onnx_enc.shape == rknn_enc.shape:
        diff = np.abs(onnx_enc.astype(np.float32) - rknn_enc.astype(np.float32))
        print(f"  Max diff:  {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        tol = 0.05
        print(f"  {'✓ OK' if diff.max() < tol else '⚠ LARGE DIFF'} (tol={tol})")
    else:
        print(f"  Shape mismatch — RKNN 내부 레이아웃 이슈")
        print(f"  ONNX[:5]: {onnx_enc.flat[:5]}")
        print(f"  RKNN[:5]: {rknn_enc.flat[:5]}")


# ─── TEST 3: Decoder + Joiner 레이턴시 측정 ──────────────────
def test_decoder_joiner_latency():
    print("\n=== [TEST 3] Decoder / Joiner 레이턴시 ===")

    dec = load_rknn_model(f'{RKNN_DIR}/decoder-epoch-99-avg-1.rknn')
    joi = load_rknn_model(f'{RKNN_DIR}/joiner-epoch-99-avg-1.rknn')

    # decoder: y[1,2] int64
    y = np.array([[0, 0]], dtype=np.int64)
    times_dec = []
    for _ in range(10):
        t0 = time.time()
        dec_out = dec.inference(inputs=[y])
        times_dec.append((time.time() - t0) * 1000)
    dec_lat = np.median(times_dec)
    print(f"  Decoder median latency: {dec_lat:.1f} ms  (shape: {np.array(dec_out[0]).shape})")

    # joiner: encoder_out[1,512] + decoder_out[1,512]
    enc_feat = np.zeros([1, 512], dtype=np.float32)
    dec_feat = np.array(dec_out[0]).astype(np.float32)
    if dec_feat.shape != (1, 512):
        dec_feat = dec_feat.reshape(1, 512)

    times_joi = []
    for _ in range(10):
        t0 = time.time()
        joi_out = joi.inference(inputs=[enc_feat, dec_feat])
        times_joi.append((time.time() - t0) * 1000)
    joi_lat = np.median(times_joi)
    logit_shape = np.array(joi_out[0]).shape
    print(f"  Joiner median latency:  {joi_lat:.1f} ms  (shape: {logit_shape})")

    dec.release()
    joi.release()


# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Zipformer RKNN 추론 테스트")
    print("=" * 60)

    ok = test_constant_check()
    test_rknn_vs_onnx()
    test_decoder_joiner_latency()

    print("\n" + "=" * 60)
    print(f"Encoder 상수 출력 문제: {'없음 ✓' if ok else '있음 ⚠'}")
