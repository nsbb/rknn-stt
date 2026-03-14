"""
Encoder INT8 RKNN 변환
- 캘리브레이션 데이터: test_wavs/*.wav 에서 추출한 Fbank 특징
- INT8 양자화로 NPU 데이터 전송 오버헤드 감소 기대
"""
import numpy as np
import os, sys, glob
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from fbank import KaldiFbank

BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

ENC_SCHEMA = [
    ('x',              [1, 39, 80],        'float32'),
    ('cached_len_0',   [2, 1],             'int64'),
    ('cached_len_1',   [4, 1],             'int64'),
    ('cached_len_2',   [3, 1],             'int64'),
    ('cached_len_3',   [2, 1],             'int64'),
    ('cached_len_4',   [4, 1],             'int64'),
    ('cached_avg_0',   [2, 1, 384],        'float32'),
    ('cached_avg_1',   [4, 1, 384],        'float32'),
    ('cached_avg_2',   [3, 1, 384],        'float32'),
    ('cached_avg_3',   [2, 1, 384],        'float32'),
    ('cached_avg_4',   [4, 1, 384],        'float32'),
    ('cached_key_0',   [2, 64, 1, 192],    'float32'),
    ('cached_key_1',   [4, 32, 1, 192],    'float32'),
    ('cached_key_2',   [3, 16, 1, 192],    'float32'),
    ('cached_key_3',   [2,  8, 1, 192],    'float32'),
    ('cached_key_4',   [4, 32, 1, 192],    'float32'),
    ('cached_val_0',   [2, 64, 1, 96],     'float32'),
    ('cached_val_1',   [4, 32, 1, 96],     'float32'),
    ('cached_val_2',   [3, 16, 1, 96],     'float32'),
    ('cached_val_3',   [2,  8, 1, 96],     'float32'),
    ('cached_val_4',   [4, 32, 1, 96],     'float32'),
    ('cached_val2_0',  [2, 64, 1, 96],     'float32'),
    ('cached_val2_1',  [4, 32, 1, 96],     'float32'),
    ('cached_val2_2',  [3, 16, 1, 96],     'float32'),
    ('cached_val2_3',  [2,  8, 1, 96],     'float32'),
    ('cached_val2_4',  [4, 32, 1, 96],     'float32'),
    ('cached_conv1_0', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv1_2', [3, 1, 384, 30],    'float32'),
    ('cached_conv1_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv1_4', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_0', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_1', [4, 1, 384, 30],    'float32'),
    ('cached_conv2_2', [3, 1, 384, 30],    'float32'),
    ('cached_conv2_3', [2, 1, 384, 30],    'float32'),
    ('cached_conv2_4', [4, 1, 384, 30],    'float32'),
]
INPUT_NAMES  = [s[0] for s in ENC_SCHEMA]
INPUT_SHAPES = [s[1] for s in ENC_SCHEMA]
CACHE_NAMES  = [s[0] for s in ENC_SCHEMA if s[0] != 'x']

CHUNK = 39; OFFSET = 32

def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))

def make_calib_data(n_samples=50):
    """
    캘리브레이션 데이터 생성:
    - test_wavs 에서 청크 추출
    - 각 청크를 36개 입력 텐서로 변환
    Returns: list of 36 lists, each containing n_samples arrays
    """
    import soundfile as sf
    import onnxruntime as ort

    sess = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx',
                                providers=['CPUExecutionProvider'])

    calib_chunks = []  # list of (list of 36 arrays)
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

            # 현재 상태를 캘리브레이션 샘플로 수집 (NCHW 그대로 — 양자화 시 ONNX 포맷 사용)
            inputs_for_calib = []
            for nm, sh, _ in ENC_SCHEMA:
                a = state[nm].copy()
                # NHWC 변환 안 함 — 캘리브레이션은 ONNX 포맷(NCHW) 사용
                inputs_for_calib.append(a)
            calib_chunks.append(inputs_for_calib)

            # ONNX로 캐시 업데이트
            onnx_in = {nm: state[nm] for nm, _, _ in ENC_SCHEMA}
            out = sess.run(None, onnx_in)
            for i, nm in enumerate(CACHE_NAMES):
                state[nm] = np.array(out[i + 1])

            num_proc += OFFSET

        if len(calib_chunks) >= n_samples:
            break

    print(f"  캘리브레이션 샘플: {len(calib_chunks)}")

    # RKNN은 입력별 list of arrays 형식 필요
    # calib_data[input_idx] = [arr_sample_0, arr_sample_1, ...]
    calib_data = [[] for _ in range(len(ENC_SCHEMA))]
    for chunk in calib_chunks:
        for idx, arr in enumerate(chunk):
            calib_data[idx].append(arr)

    # 각 입력별로 최대 n_samples만 사용
    calib_data = [data[:n_samples] for data in calib_data]
    return calib_data


def save_calib_npy(calib_data, out_dir='/tmp/calib_enc'):
    """
    RKNN dataset 형식으로 저장:
    각 샘플을 numpy .npy 파일로, 파일 목록을 txt로
    RKNN은 multi-input 모델의 경우 각 입력별로 파일 목록 필요
    """
    os.makedirs(out_dir, exist_ok=True)
    n_inputs = len(calib_data)
    n_samples = len(calib_data[0])

    # 각 샘플을 하나의 .npy 파일 (모든 입력을 하나로)
    # RKNN 방식: 각 입력별 .npy 파일로 저장
    # dataset.txt 형식: 한 줄에 n개 파일 경로 (tab 구분) 또는
    # 각 입력을 별도 파일로
    txt_lines = []
    for s_idx in range(n_samples):
        line_parts = []
        for i_idx in range(n_inputs):
            fpath = f'{out_dir}/s{s_idx:03d}_i{i_idx:02d}.npy'
            np.save(fpath, calib_data[i_idx][s_idx])
            line_parts.append(fpath)
        txt_lines.append(' '.join(line_parts))

    dataset_txt = f'{out_dir}/dataset.txt'
    with open(dataset_txt, 'w') as f:
        f.write('\n'.join(txt_lines))
    print(f"  Saved {n_samples} samples to {out_dir}/")
    return dataset_txt


if __name__ == '__main__':
    from rknn.api import RKNN

    print("캘리브레이션 데이터 생성 ...")
    calib_data = make_calib_data(n_samples=30)
    print(f"  완료: {len(calib_data)} inputs × {len(calib_data[0])} samples")

    dataset_txt = save_calib_npy(calib_data)
    print(f"  Dataset file: {dataset_txt}")

    rknn = RKNN(verbose=True)
    rknn.config(target_platform='rk3588', quantized_dtype='asymmetric_quantized-8')

    print("\nloading onnx ...")
    ret = rknn.load_onnx(
        model=f'{BASE}/encoder-epoch-99-avg-1.onnx',
        inputs=INPUT_NAMES,
        input_size_list=INPUT_SHAPES,
    )
    if ret != 0:
        print(f"load_onnx failed: {ret}")
        exit(1)

    print("\nbuilding INT8 ...")
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print(f"build failed: {ret}")
        exit(1)

    rknn_path = f'{RKNN_DIR}/encoder-epoch-99-avg-1-int8.rknn'
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export failed: {ret}")
        exit(1)
    print(f"\nExported INT8: {rknn_path}")
    rknn.release()
    print("Done.")
