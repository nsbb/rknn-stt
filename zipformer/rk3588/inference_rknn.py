"""
Zipformer RKNN STT 추론 파이프라인

모델: rknn-stt/zipformer/rk3588/*.rknn
특징: KaldiFbank (numpy 구현, kaldifeat 없이 동작)
디코딩: Greedy search (transducer)
캐시: RKNN 4D 입력 NHWC 변환, 출력 NCHW→NHWC 피드백

실행:
  conda run -n RKNN-Toolkit2 python inference_rknn.py [WAV_PATH] [--compare-onnx]
"""

import numpy as np
import os, sys, time, argparse

BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'

sys.path.insert(0, RKNN_DIR)
from fbank import KaldiFbank

# ─── Runtime 선택 ──────────────────────────────────────────────
try:
    from rknnlite.api import RKNNLite
    NPU_CORE_0 = RKNNLite.NPU_CORE_0
    def _load(path, core=NPU_CORE_0):
        m = RKNNLite(verbose=False)
        assert m.load_rknn(path) == 0
        assert m.init_runtime(core_mask=core) == 0
        return m
    RUNTIME = 'rknnlite'
except ImportError:
    from rknn.api import RKNN as RKNNLite
    NPU_CORE_0 = None
    def _load(path, core=None):
        m = RKNNLite(verbose=False)
        assert m.load_rknn(path) == 0
        assert m.init_runtime() == 0
        return m
    RUNTIME = 'rknn-sim'

print(f"[Runtime] {RUNTIME}")

# ─── Encoder 입력 스키마 ─────────────────────────────────────
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

# 입력 이름 리스트 (x 포함)
ENC_INPUT_NAMES = [s[0] for s in ENC_SCHEMA]
# 캐시 이름 (x 제외), 출력 인덱스 1부터 대응
CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']

CHUNK = 39       # encoder input frames per step
OFFSET = 32      # stride: 새로 처리할 프레임 수 (overlap=7)
CONTEXT_SIZE = 2 # decoder context size (BPE 토큰 수)
BLANK_ID = 0
UNK_ID   = 2


def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))
def nhwc2nchw(a): return np.transpose(a, (0, 3, 1, 2))


def init_cache():
    """캐시 상태 초기화 (NCHW 형식으로 저장)"""
    state = {}
    for name, shape, dtype in ENC_SCHEMA:
        state[name] = np.zeros(shape, dtype=np.dtype(dtype))
    return state


def pack_rknn_inputs(state):
    """NCHW 상태를 RKNN 입력 리스트로 변환 (4D → NHWC)"""
    inputs = []
    for name, shape, _ in ENC_SCHEMA:
        arr = state[name]
        if len(shape) == 4:
            arr = nchw2nhwc(arr)   # (N,C,H,W) → (N,H,W,C)
        inputs.append(arr)
    return inputs


def unpack_rknn_outputs(rknn_out, state):
    """
    RKNN 출력으로 캐시 상태 업데이트.
    RKNN 출력: [0]=encoder_out, [1..35]=new_cache 순서
    4D 출력은 NCHW 형식 (no conversion needed for 4D key/val)
    conv 출력은 (N,1,384,30) NCHW
    """
    enc_out = np.array(rknn_out[0])    # (1, T_out, 512)
    for idx, name in enumerate(CACHE_NAMES):
        arr = np.array(rknn_out[idx + 1])
        # cached_len은 float32 출력이지만 int64로 저장
        if 'cached_len' in name:
            arr = arr.astype(np.int64)
        state[name] = arr
    return enc_out, state


def load_vocab(tokens_file):
    vocab = {}
    with open(tokens_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                token, idx = parts[0], int(parts[1])
                vocab[idx] = token
            elif len(parts) == 1:
                vocab[0] = parts[0]
    return vocab


def decode_hyp(hyp, vocab):
    text = ''.join(vocab.get(i, '') for i in hyp)
    return text.replace('▁', ' ').strip()


class ZipformerRKNN:
    """RKNN 기반 Zipformer 스트리밍 추론"""
    def __init__(self, rknn_dir=RKNN_DIR, core=NPU_CORE_0):
        print("Loading RKNN models ...")
        t0 = time.time()
        self.encoder = _load(f'{rknn_dir}/encoder-epoch-99-avg-1.rknn', core)
        self.decoder = _load(f'{rknn_dir}/decoder-epoch-99-avg-1.rknn', core)
        self.joiner  = _load(f'{rknn_dir}/joiner-epoch-99-avg-1.rknn',  core)
        print(f"  Loaded in {(time.time()-t0)*1000:.0f} ms")
        self.vocab = load_vocab(f'{BASE}/tokens.txt')

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.joiner.release()

    def transcribe(self, wav_path, verbose=False):
        """WAV 파일 → 텍스트 (레이턴시 통계 포함)"""
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        return self._run_inference(audio, verbose=verbose)

    def _run_inference(self, audio, verbose=False):
        # 1. 특징 추출
        t_feat0 = time.time()
        fb = KaldiFbank()
        feats = fb.compute_all(audio)   # (T, 80)
        t_feat1 = time.time()

        T = feats.shape[0]
        # 끝 패딩 (마지막 청크가 CHUNK에 모자라는 경우)
        if T % OFFSET != 0 or T < CHUNK:
            pad_needed = max(CHUNK, ((T // OFFSET) + 1) * OFFSET) - T
            feats = np.vstack([feats, np.zeros((pad_needed, 80), dtype=np.float32)])

        # 2. 상태 초기화
        state = init_cache()
        hyp = [BLANK_ID] * CONTEXT_SIZE
        decoder_in = np.array([hyp], dtype=np.int64)
        dec_out = np.array(self.decoder.inference(inputs=[decoder_in])[0])

        # 레이턴시 측정
        enc_times, dec_times, joi_times = [], [], []
        chunk_count = 0
        num_processed = 0

        # 3. 청크 단위 처리
        while num_processed + CHUNK <= feats.shape[0]:
            x_chunk = feats[num_processed: num_processed + CHUNK]   # (39, 80)
            state['x'] = x_chunk[np.newaxis, :, :]                  # (1, 39, 80)

            rknn_inputs = pack_rknn_inputs(state)

            t_enc0 = time.time()
            rknn_out = self.encoder.inference(inputs=rknn_inputs)
            t_enc1 = time.time()
            enc_times.append((t_enc1 - t_enc0) * 1000)

            enc_out, state = unpack_rknn_outputs(rknn_out, state)
            # enc_out: (1, T_out, 512) → (T_out, 512)
            enc_out = enc_out.squeeze(0)

            # 4. Greedy search (T_out 프레임)
            for t in range(enc_out.shape[0]):
                cur_enc = enc_out[t: t + 1]   # (1, 512)

                t_joi0 = time.time()
                joi_in_enc = cur_enc.astype(np.float32)
                joi_in_dec = dec_out.astype(np.float32).reshape(1, 512)
                joi_out = np.array(self.joiner.inference(inputs=[joi_in_enc, joi_in_dec])[0])
                t_joi1 = time.time()
                joi_times.append((t_joi1 - t_joi0) * 1000)

                y = int(np.argmax(joi_out.squeeze()))
                if y != BLANK_ID and y != UNK_ID:
                    hyp.append(y)
                    dec_in = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)
                    t_dec0 = time.time()
                    dec_out = np.array(self.decoder.inference(inputs=[dec_in])[0])
                    t_dec1 = time.time()
                    dec_times.append((t_dec1 - t_dec0) * 1000)

            num_processed += OFFSET
            chunk_count += 1

        text = decode_hyp(hyp[CONTEXT_SIZE:], self.vocab)

        stats = {
            'text': text,
            'feat_ms':     (t_feat1 - t_feat0) * 1000,
            'enc_median':  float(np.median(enc_times)) if enc_times else 0,
            'dec_median':  float(np.median(dec_times)) if dec_times else 0,
            'joi_median':  float(np.median(joi_times)) if joi_times else 0,
            'enc_total':   float(np.sum(enc_times)),
            'chunks':      chunk_count,
            'audio_ms':    len(audio) / 16000 * 1000,
        }
        if verbose:
            rtf = stats['enc_total'] / stats['audio_ms']
            print(f"  Chunks: {chunk_count}, Feat: {stats['feat_ms']:.0f}ms")
            print(f"  Enc median: {stats['enc_median']:.1f}ms, total: {stats['enc_total']:.0f}ms")
            print(f"  Dec median: {stats['dec_median']:.1f}ms")
            print(f"  Joi median: {stats['joi_median']:.1f}ms")
            print(f"  RTF (enc only): {rtf:.3f} ({'✓ realtime' if rtf < 1.0 else '⚠ too slow'})")
        return stats


def run_test_wavs(model, verbose=True):
    """test_wavs/*.wav 전부 추론"""
    import glob
    trans_path = f'{BASE}/test_wavs/trans.txt'
    gt_map = {}
    if os.path.exists(trans_path):
        with open(trans_path) as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    gt_map[parts[0]] = parts[1]

    wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
    results = []
    for wav in wavs:
        fname = os.path.basename(wav)
        stats = model.transcribe(wav, verbose=verbose)
        text  = stats['text']
        gt    = gt_map.get(fname, gt_map.get(fname.split('.')[0], ''))
        print(f"\n[{fname}]")
        print(f"  Result: {text}")
        if gt:
            print(f"  GT:     {gt}")
        results.append({'file': fname, 'text': text, 'gt': gt, **stats})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', nargs='?', default=None)
    parser.add_argument('--compare-onnx', action='store_true')
    parser.add_argument('--bench', action='store_true', help='반복 벤치마크')
    args = parser.parse_args()

    model = ZipformerRKNN()

    if args.wav:
        stats = model.transcribe(args.wav, verbose=True)
        print(f"\n결과: {stats['text']}")

        if args.compare_onnx:
            import onnxruntime as ort
            # ONNX greedy search for comparison
            print("\n--- ONNX 비교 ---")
            enc_s = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
            dec_s = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
            joi_s = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.onnx',  providers=['CPUExecutionProvider'])
            from inference_onnx import ZipformerONNX
            onnx_model = ZipformerONNX(enc_s, dec_s, joi_s, load_vocab(f'{BASE}/tokens.txt'))
            onnx_res = onnx_model.transcribe(args.wav, verbose=True)
            print(f"ONNX: {onnx_res['text']}")
            print(f"RKNN: {stats['text']}")
    else:
        results = run_test_wavs(model, verbose=True)
        print("\n" + "="*60)
        print("요약:")
        for r in results:
            print(f"  {r['file']}: {r['text']}")

    if args.bench:
        wav = args.wav or f'{BASE}/test_wavs/0.wav'
        import soundfile as sf
        audio, _ = sf.read(wav, dtype='float32')
        print(f"\n--- 벤치마크 (10회) ---")
        times = []
        for i in range(10):
            t0 = time.time()
            model._run_inference(audio)
            times.append((time.time()-t0)*1000)
        print(f"  Median: {np.median(times):.1f}ms")
        print(f"  Min:    {np.min(times):.1f}ms")

    model.release()
