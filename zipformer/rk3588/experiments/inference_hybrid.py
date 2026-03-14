"""
Zipformer 하이브리드 추론:
- Encoder: RKNN(encoder_out 계산) + ONNX(캐시 업데이트)
- Decoder/Joiner: RKNN

근거: RKNN의 CumSum/ReduceMean이 cached_avg를 잘못 계산함.
      encoder_out은 정확하나 새 캐시 값이 발산 → ONNX로 캐시 관리.

레이턴시 분석:
  - 청크당 RKNN encoder: ~62ms
  - 청크당 ONNX cache update: ~50ms (CPU, 필요시 최적화 가능)
  - 총: ~112ms/청크 (RTF≈0.34 for 39 frames)
"""

import numpy as np
import os, sys, time, argparse
import onnxruntime as ort

BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
sys.path.insert(0, RKNN_DIR)
from fbank import KaldiFbank

try:
    from rknnlite.api import RKNNLite
    NPU_CORE_0 = RKNNLite.NPU_CORE_0
    def _load(path):
        m = RKNNLite(verbose=False)
        assert m.load_rknn(path) == 0
        assert m.init_runtime(core_mask=NPU_CORE_0) == 0
        return m
    RUNTIME = 'rknnlite'
except ImportError:
    from rknn.api import RKNN as RKNNLite
    def _load(path):
        m = RKNNLite(verbose=False)
        assert m.load_rknn(path) == 0
        assert m.init_runtime() == 0
        return m
    RUNTIME = 'rknn-sim'

print(f"[Runtime] {RUNTIME}")

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
CHUNK = 39; OFFSET = 32; CONTEXT_SIZE = 2; BLANK_ID = 0; UNK_ID = 2

def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))

def load_vocab():
    vocab = {}
    with open(f'{BASE}/tokens.txt') as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2: vocab[int(p[1])] = p[0]
    return vocab

def decode(hyp, vocab):
    return ''.join(vocab.get(i, '') for i in hyp).replace('▁', ' ').strip()

def init_state():
    return {nm: np.zeros(sh, dtype=np.dtype(dt)) for nm, sh, dt in ENC_SCHEMA}

def pack_rknn_inputs(state):
    inputs = []
    for nm, sh, _ in ENC_SCHEMA:
        a = state[nm]
        if len(sh) == 4: a = nchw2nhwc(a)
        inputs.append(a)
    return inputs


class ZipformerHybrid:
    """
    Hybrid: RKNN encoder_out + ONNX cache update + RKNN decoder/joiner
    """
    def __init__(self):
        print("Loading models ...")
        # RKNN: encoder (encoder_out 추출용), decoder, joiner
        self.enc_r = _load(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
        self.dec_r = _load(f'{RKNN_DIR}/decoder-epoch-99-avg-1.rknn')
        self.joi_r = _load(f'{RKNN_DIR}/joiner-epoch-99-avg-1.rknn')
        # ONNX: encoder (캐시 업데이트용)
        self.enc_s = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx',
                                          providers=['CPUExecutionProvider'])
        self.vocab = load_vocab()
        print("  Loaded.")

    def release(self):
        self.enc_r.release(); self.dec_r.release(); self.joi_r.release()

    def transcribe(self, wav_path, verbose=False):
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=1)
        return self._run(audio, verbose=verbose)

    def _run(self, audio, verbose=False):
        # 특징 추출
        t0 = time.time()
        feats = KaldiFbank().compute_all(audio)  # (T, 80)
        t1 = time.time()
        feat_ms = (t1 - t0) * 1000

        T = feats.shape[0]
        if T % OFFSET != 0 or T < CHUNK:
            pad = max(CHUNK, ((T // OFFSET) + 1) * OFFSET) - T
            feats = np.vstack([feats, np.zeros((pad, 80), dtype=np.float32)])

        state = init_state()
        hyp = [BLANK_ID] * CONTEXT_SIZE
        dec_in = np.array([hyp], dtype=np.int64)
        dec_out = np.array(self.dec_r.inference(inputs=[dec_in])[0])

        enc_r_ms, enc_o_ms, joi_ms = [], [], []
        num_proc = 0

        while num_proc + CHUNK <= feats.shape[0]:
            state['x'] = feats[num_proc: num_proc + CHUNK][np.newaxis]

            # ── RKNN: encoder_out 계산 ──────────────────────────
            rknn_in = pack_rknn_inputs(state)
            t_r0 = time.time()
            rknn_out = self.enc_r.inference(inputs=rknn_in)
            enc_r_ms.append((time.time() - t_r0) * 1000)
            enc_out = np.array(rknn_out[0]).squeeze(0)  # (T_out, 512)

            # ── ONNX: 캐시 업데이트만 (encoder_out 버림) ────────
            onnx_in = {nm: state[nm] for nm, _, _ in ENC_SCHEMA}
            t_o0 = time.time()
            onnx_out = self.enc_s.run(None, onnx_in)
            enc_o_ms.append((time.time() - t_o0) * 1000)
            for i, nm in enumerate(CACHE_NAMES):
                state[nm] = np.array(onnx_out[i + 1])

            # ── Greedy search (T_out 프레임) ────────────────────
            for t in range(enc_out.shape[0]):
                cur_enc = enc_out[t: t + 1].astype(np.float32)
                dec_f   = dec_out.astype(np.float32).reshape(1, 512)
                t_j0 = time.time()
                joi_out = np.array(self.joi_r.inference(inputs=[cur_enc, dec_f])[0])
                joi_ms.append((time.time() - t_j0) * 1000)

                y = int(np.argmax(joi_out.squeeze()))
                if y != BLANK_ID and y != UNK_ID:
                    hyp.append(y)
                    dec_in = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)
                    dec_out = np.array(self.dec_r.inference(inputs=[dec_in])[0])

            num_proc += OFFSET

        text = decode(hyp[CONTEXT_SIZE:], self.vocab)
        audio_ms = len(audio) / 16000 * 1000
        total_enc_ms = sum(enc_r_ms) + sum(enc_o_ms)

        stats = {
            'text': text,
            'feat_ms': feat_ms,
            'enc_rknn_median': float(np.median(enc_r_ms)) if enc_r_ms else 0,
            'enc_onnx_median': float(np.median(enc_o_ms)) if enc_o_ms else 0,
            'joi_median': float(np.median(joi_ms)) if joi_ms else 0,
            'chunks': len(enc_r_ms),
            'audio_ms': audio_ms,
            'rtf': total_enc_ms / audio_ms,
        }
        if verbose:
            print(f"  Chunks: {stats['chunks']}  Feat: {feat_ms:.0f}ms")
            print(f"  Enc-RKNN median: {stats['enc_rknn_median']:.1f}ms")
            print(f"  Enc-ONNX median: {stats['enc_onnx_median']:.1f}ms")
            print(f"  Joiner median:   {stats['joi_median']:.1f}ms")
            print(f"  RTF (enc total): {stats['rtf']:.3f}  ({'✓' if stats['rtf']<1.0 else '⚠'})")
        return stats


if __name__ == '__main__':
    model = ZipformerHybrid()
    vocab = model.vocab
    trans_path = f'{BASE}/test_wavs/trans.txt'
    gt_map = {}
    with open(trans_path) as f:
        for line in f:
            nm, *rest = line.strip().split(' ', 1)
            gt_map[nm] = rest[0] if rest else ''

    import glob
    wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
    print()
    for wav in wavs:
        fname = os.path.basename(wav)
        stats = model.transcribe(wav, verbose=True)
        gt = gt_map.get(fname, '')
        print(f"[{fname}]")
        print(f"  HYBRID: {stats['text']}")
        print(f"  GT:     {gt}")
        print()

    model.release()
