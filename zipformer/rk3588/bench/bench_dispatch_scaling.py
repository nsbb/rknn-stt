"""
RKNN per-layer dispatch 스케일링 테스트.
모델 크기에 따라 per-layer dispatch가 달라지는지 확인.
decoder (작은 모델)로 비교.
"""
import numpy as np, time, sys

try:
    from rknnlite.api import RKNNLite
    NPU_CORE_0 = RKNNLite.NPU_CORE_0
except ImportError:
    from rknn.api import RKNN as RKNNLite
    NPU_CORE_0 = None

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'


def bench_model(model_path, inputs, n_runs=100, warmup=10):
    """Benchmark a RKNN model's inference time."""
    m = RKNNLite(verbose=False)
    assert m.load_rknn(model_path) == 0
    if NPU_CORE_0:
        assert m.init_runtime(core_mask=NPU_CORE_0) == 0
    else:
        assert m.init_runtime() == 0

    # Warmup
    for _ in range(warmup):
        m.inference(inputs=inputs)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        m.inference(inputs=inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    m.release()
    return times


def main():
    print("=== RKNN Per-Layer Dispatch Scaling Test ===\n")

    # 1. Decoder (small model, ~5MB)
    print("1. Decoder model (5MB)")
    dec_path = f'{RKNN_DIR}/decoder-epoch-99-avg-1.rknn'
    dec_input = [np.array([[0, 0]], dtype=np.int64)]
    dec_times = bench_model(dec_path, dec_input, n_runs=200)
    dec_med = np.median(dec_times)
    print(f"   Median: {dec_med:.3f}ms")

    # 2. Joiner (small model, ~5MB)
    print("2. Joiner model (5MB)")
    joi_path = f'{RKNN_DIR}/joiner-epoch-99-avg-1.rknn'
    joi_enc = np.zeros((1, 512), dtype=np.float32)
    joi_dec = np.zeros((1, 512), dtype=np.float32)
    joi_times = bench_model(joi_path, [joi_enc, joi_dec], n_runs=200)
    joi_med = np.median(joi_times)
    print(f"   Median: {joi_med:.3f}ms")

    # 3. Encoder (large model, ~80MB)
    print("3. Encoder model (80MB)")
    sys.path.insert(0, RKNN_DIR)
    from inference_rknn import ENC_SCHEMA, nchw2nhwc
    enc_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn'

    enc_inputs = []
    for name, shape, dtype in ENC_SCHEMA:
        arr = np.zeros(shape, dtype=np.dtype(dtype))
        if len(shape) == 4:
            arr = nchw2nhwc(arr)
        enc_inputs.append(arr)

    enc_times = bench_model(enc_path, enc_inputs, n_runs=50, warmup=5)
    enc_med = np.median(enc_times)
    print(f"   Median: {enc_med:.3f}ms")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Model':20s} {'Size':>8s} {'Median ms':>10s}")
    print("-" * 42)
    print(f"{'Decoder':20s} {'5 MB':>8s} {dec_med:10.3f}")
    print(f"{'Joiner':20s} {'5 MB':>8s} {joi_med:10.3f}")
    print(f"{'Encoder':20s} {'80 MB':>8s} {enc_med:10.3f}")

    # Estimate RKNN layers from model size
    # Encoder: 1619 layers, 80MB
    # If per-layer dispatch scales with model size, decoder should have proportionally fewer layers
    print(f"\n--- Per-Layer Dispatch Analysis ---")
    # Known: encoder has 1619 RKNN layers
    enc_per_layer = enc_med / 1619 * 1000  # µs
    print(f"  Encoder: {enc_med:.1f}ms / 1619 layers = {enc_per_layer:.1f}µs/layer")
    print(f"  Decoder: {dec_med:.3f}ms (need to check layer count)")
    print(f"  Joiner:  {joi_med:.3f}ms (need to check layer count)")


if __name__ == '__main__':
    main()
