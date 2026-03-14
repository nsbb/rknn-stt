"""
Zipformer ONNX STT 추론 (RKNN 비교 기준)
동일한 fbank + chunk 방식으로 ONNX 추론
"""
import numpy as np
import os, sys, time, argparse
import onnxruntime as ort

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
sys.path.insert(0, f'{BASE}/rk3588')
from fbank import KaldiFbank

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

CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']
CHUNK = 39
OFFSET = 32
CONTEXT_SIZE = 2
BLANK_ID = 0
UNK_ID   = 2


def init_cache():
    state = {}
    for name, shape, dtype in ENC_SCHEMA:
        state[name] = np.zeros(shape, dtype=np.dtype(dtype))
    return state


def load_vocab(tokens_file):
    vocab = {}
    with open(tokens_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                vocab[int(parts[1])] = parts[0]
            elif len(parts) == 1:
                vocab[0] = parts[0]
    return vocab


def decode_hyp(hyp, vocab):
    text = ''.join(vocab.get(i, '') for i in hyp)
    return text.replace('▁', ' ').strip()


class ZipformerONNX:
    def __init__(self, onnx_dir=BASE, use_int8=True):
        """
        use_int8=True: INT8 ONNX 모델 사용 (권장)
          - encoder INT8: 30.6ms/chunk (FP32: 39.4ms, 1.3x faster)
          - joiner  INT8: 0.07ms/call  (FP32: 0.69ms, 10x faster)
          - CER: FP32와 동일 (19.95%)
        """
        suffix = '.int8.onnx' if use_int8 else '.onnx'
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(f'{onnx_dir}/encoder-epoch-99-avg-1{suffix}', sess_options=opts, providers=['CPUExecutionProvider'])
        self.dec = ort.InferenceSession(f'{onnx_dir}/decoder-epoch-99-avg-1{suffix}', providers=['CPUExecutionProvider'])
        self.joi = ort.InferenceSession(f'{onnx_dir}/joiner-epoch-99-avg-1{suffix}',  providers=['CPUExecutionProvider'])
        self.vocab = load_vocab(f'{onnx_dir}/tokens.txt')

    def transcribe(self, wav_path, verbose=False):
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Fbank 특징
        fb = KaldiFbank()
        feats = fb.compute_all(audio)   # (T, 80)
        T = feats.shape[0]
        if T % OFFSET != 0 or T < CHUNK:
            pad = max(CHUNK, ((T // OFFSET) + 1) * OFFSET) - T
            feats = np.vstack([feats, np.zeros((pad, 80), dtype=np.float32)])

        state = init_cache()
        hyp = [BLANK_ID] * CONTEXT_SIZE
        dec_in = np.array([hyp], dtype=np.int64)
        dec_out = self.dec.run(None, {'y': dec_in})[0]

        num_processed = 0
        enc_times = []

        while num_processed + CHUNK <= feats.shape[0]:
            x_chunk = feats[num_processed: num_processed + CHUNK]
            state['x'] = x_chunk[np.newaxis, :, :]

            onnx_in = {name: state[name] for name, _, _ in ENC_SCHEMA}
            t0 = time.time()
            onnx_out = self.enc.run(None, onnx_in)
            enc_times.append((time.time()-t0)*1000)

            enc_out = np.array(onnx_out[0]).squeeze(0)  # (T_out, 512)
            # 캐시 업데이트
            for idx, name in enumerate(CACHE_NAMES):
                state[name] = np.array(onnx_out[idx + 1])

            for t in range(enc_out.shape[0]):
                cur_enc = enc_out[t: t+1]
                joi_out = self.joi.run(None, {
                    'encoder_out': cur_enc.astype(np.float32),
                    'decoder_out': dec_out.astype(np.float32).reshape(1, 512)
                })[0]
                y = int(np.argmax(joi_out.squeeze()))
                if y != BLANK_ID and y != UNK_ID:
                    hyp.append(y)
                    dec_in = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)
                    dec_out = self.dec.run(None, {'y': dec_in})[0]

            num_processed += OFFSET

        text = decode_hyp(hyp[CONTEXT_SIZE:], self.vocab)
        if verbose:
            print(f"  ONNX enc_median: {np.median(enc_times):.0f}ms, chunks: {len(enc_times)}")
        return {'text': text, 'enc_times': enc_times}


if __name__ == '__main__':
    model = ZipformerONNX()
    trans_path = f'{BASE}/test_wavs/trans.txt'
    gt_map = {}
    with open(trans_path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                gt_map[parts[0]] = parts[1]
            elif ' ' in line:
                fname, gt = line.strip().split(' ', 1)
                gt_map[fname] = gt

    import glob
    wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
    for wav in wavs:
        fname = os.path.basename(wav)
        res = model.transcribe(wav, verbose=True)
        gt  = gt_map.get(fname, '')
        print(f"[{fname}]")
        print(f"  ONNX: {res['text']}")
        print(f"  GT:   {gt}")
