"""
nocache ONNX → RKNN 변환 (캐시 업데이트 제거 모델).
nocache 모델은 원본과 동일한 입력을 받지만, 출력이 다름:
- encoder_out + 35개 raw computed values (new_cached 대신)
"""
import numpy as np, os, sys, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from fbank import KaldiFbank

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
NOCACHE_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix-nocache.onnx'
CUMFIX_ONNX = f'{BASE}/encoder-epoch-99-avg-1-cumfix.onnx'
DST = f'{RKNN_DIR}/encoder-int8-cumfix-nocache.rknn'

ENC_SCHEMA = [
    ('x',              [1, 39, 80],        'float32'),
    ('cached_len_0',   [2, 1],             'int64'),  ('cached_len_1',   [4, 1],             'int64'),
    ('cached_len_2',   [3, 1],             'int64'),  ('cached_len_3',   [2, 1],             'int64'),
    ('cached_len_4',   [4, 1],             'int64'),
    ('cached_avg_0',   [2, 1, 384],        'float32'),('cached_avg_1',   [4, 1, 384],        'float32'),
    ('cached_avg_2',   [3, 1, 384],        'float32'),('cached_avg_3',   [2, 1, 384],        'float32'),
    ('cached_avg_4',   [4, 1, 384],        'float32'),
    ('cached_key_0',   [2, 64, 1, 192],    'float32'),('cached_key_1',   [4, 32, 1, 192],    'float32'),
    ('cached_key_2',   [3, 16, 1, 192],    'float32'),('cached_key_3',   [2,  8, 1, 192],    'float32'),
    ('cached_key_4',   [4, 32, 1, 192],    'float32'),
    ('cached_val_0',   [2, 64, 1, 96],     'float32'),('cached_val_1',   [4, 32, 1, 96],     'float32'),
    ('cached_val_2',   [3, 16, 1, 96],     'float32'),('cached_val_3',   [2,  8, 1, 96],     'float32'),
    ('cached_val_4',   [4, 32, 1, 96],     'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],     'float32'),('cached_val2_1',  [4, 32, 1, 96],     'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],     'float32'),('cached_val2_3',  [2,  8, 1, 96],     'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],     'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],    'float32'),('cached_conv1_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],    'float32'),('cached_conv1_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],    'float32'),('cached_conv2_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],    'float32'),('cached_conv2_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],    'float32'),
]
INPUT_NAMES  = [s[0] for s in ENC_SCHEMA]
INPUT_SHAPES = [s[1] for s in ENC_SCHEMA]
CACHE_NAMES  = [s[0] for s in ENC_SCHEMA if s[0] != 'x']
CHUNK = 39; OFFSET = 32


def make_calib_data(n_samples=30):
    """Generate calibration data using the ORIGINAL cumfix model."""
    import soundfile as sf, onnxruntime as ort
    sess = ort.InferenceSession(CUMFIX_ONNX, providers=['CPUExecutionProvider'])
    calib_chunks = []
    wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
    for wav in wavs:
        audio, _ = sf.read(wav, dtype='float32')
        feats = KaldiFbank().compute_all(audio)
        T = feats.shape[0]
        pad = max(CHUNK, ((T // OFFSET) + 1) * OFFSET) - T
        feats = np.vstack([feats, np.zeros((pad, 80), dtype=np.float32)])
        state = {nm: np.zeros(sh, dtype=np.dtype(dt)) for nm, sh, dt in ENC_SCHEMA}
        num_proc = 0
        while num_proc + CHUNK <= feats.shape[0]:
            state['x'] = feats[num_proc: num_proc + CHUNK][np.newaxis]
            calib_chunks.append([state[nm].copy() for nm, _, _ in ENC_SCHEMA])
            out = sess.run(None, {nm: state[nm] for nm, _, _ in ENC_SCHEMA})
            for i, nm in enumerate(CACHE_NAMES):
                state[nm] = np.array(out[i + 1])
            num_proc += OFFSET
        if len(calib_chunks) >= n_samples:
            break
    calib_data = [[] for _ in range(len(ENC_SCHEMA))]
    for chunk in calib_chunks:
        for idx, arr in enumerate(chunk):
            calib_data[idx].append(arr)
    return [data[:n_samples] for data in calib_data]


def save_calib_npy(calib_data, out_dir='/tmp/calib_nocache'):
    os.makedirs(out_dir, exist_ok=True)
    txt_lines = []
    for s_idx in range(len(calib_data[0])):
        line_parts = []
        for i_idx in range(len(calib_data)):
            fpath = f'{out_dir}/s{s_idx:03d}_i{i_idx:02d}.npy'
            np.save(fpath, calib_data[i_idx][s_idx])
            line_parts.append(fpath)
        txt_lines.append(' '.join(line_parts))
    dataset_txt = f'{out_dir}/dataset.txt'
    with open(dataset_txt, 'w') as f:
        f.write('\n'.join(txt_lines))
    return dataset_txt


if __name__ == '__main__':
    from rknn.api import RKNN

    print("Calibration data...")
    calib_data = make_calib_data(30)
    dataset_txt = save_calib_npy(calib_data)
    print(f"  {len(calib_data[0])} samples saved to {dataset_txt}")

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform='rk3588',
        remove_reshape=True,
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
    )

    print(f"Loading ONNX: {NOCACHE_ONNX}")
    ret = rknn.load_onnx(model=NOCACHE_ONNX, inputs=INPUT_NAMES, input_size_list=INPUT_SHAPES)
    if ret != 0:
        print(f"load_onnx with explicit inputs FAILED: {ret}")
        print("Trying without explicit inputs...")
        ret = rknn.load_onnx(model=NOCACHE_ONNX)
        if ret != 0:
            print(f"load_onnx FAILED: {ret}")
            sys.exit(1)

    print("Building INT8...")
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print(f"Build FAILED: {ret}")
        sys.exit(1)

    ret = rknn.export_rknn(DST)
    print(f"Export: {'OK' if ret == 0 else 'FAILED'}")
    print(f"Saved: {DST} ({os.path.getsize(DST)/1024/1024:.1f} MB)")
    rknn.release()
