"""wav2vec2 inference using split RKNN models.

Architecture: Part1(FP16) -> Part2A(INT8-KL, layers 0-N) -> Part2B(FP16, layers N-23) -> Part3(FP16)

Split points available:
  - split11 (default): layers 0-11 INT8, CER 35.25% (better than FP16 35.96%), 427ms RTF 0.085
  - split15: layers 0-15 INT8, CER 37.06%, 404ms RTF 0.081, 15% faster than FP16
  - split17: layers 0-17 INT8, CER 37.57%, 391ms RTF 0.078, 18% faster than FP16

INT8 quantization uses KL divergence algorithm for best accuracy.
Full INT8 on all encoder layers produces garbage output due to LayerNorm/Softmax/GELU sensitivity.
"""
import numpy as np, time, os, sys, json
import soundfile as sf, scipy.signal

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
from rknnlite.api import RKNNLite

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_CONFIGS = {
    'split11': {
        'part2a': f'{BASE}/model/wav2vec2_part2a_int8_kl.rknn',
        'part2b': f'{BASE}/model/wav2vec2_part2b_fp16.rknn',
    },
    'split15': {
        'part2a': f'{BASE}/model/wav2vec2_enc15a_int8_kl.rknn',
        'part2b': f'{BASE}/model/wav2vec2_enc15b_fp16.rknn',
    },
    'split17': {
        'part2a': f'{BASE}/model/wav2vec2_enc17a_int8_kl.rknn',
        'part2b': f'{BASE}/model/wav2vec2_enc17b_fp16.rknn',
    },
}

MODELS = {
    'part1': f'{BASE}/model/wav2vec2_part1_features_fp16.rknn',
    'part3': f'{BASE}/model/wav2vec2_part3_lmhead_fp16.rknn',
}


def load_audio(path, max_len=80000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
    audio = audio.astype(np.float32)
    # Amplitude normalization: critical for INT8 accuracy
    # target=5.0 optimal for INT8-KL (CER 18.2% → 11.5% vs target=0.95)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 5.0
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    return audio[np.newaxis]


def load_vocab(path):
    with open(path) as f:
        vocab = json.load(f)
    return {int(v): k for k, v in vocab.items()}


def decode(output, id2tok):
    ids = np.argmax(output, axis=-1)[0]
    prev = -1
    chars = []
    for i in ids:
        if i != prev:
            tok = id2tok.get(i, '')
            if tok == '|':
                chars.append(' ')
            elif tok not in ['', '<pad>', '<s>', '</s>', '<unk>'] and i >= 5:
                chars.append(tok)
        prev = i
    return ''.join(chars)


class SplitWav2Vec2:
    def __init__(self, split='split11', core_mask=RKNNLite.NPU_CORE_0_1_2):
        if split not in SPLIT_CONFIGS:
            raise ValueError(f"Unknown split '{split}'. Available: {list(SPLIT_CONFIGS.keys())}")
        split_cfg = SPLIT_CONFIGS[split]
        all_models = {**MODELS, **split_cfg}
        self.rknns = {}
        for name, path in all_models.items():
            rknn = RKNNLite(verbose=False)
            ret = rknn.load_rknn(path)
            if ret != 0:
                raise RuntimeError(f"Failed to load {name}: {path}")
            ret = rknn.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"Failed to init {name}")
            self.rknns[name] = rknn
        self.id2tok = load_vocab(f'{BASE}/json/vocab.json')

    def inference(self, audio):
        o1 = self.rknns['part1'].inference(inputs=[audio])[0]
        o2a = self.rknns['part2a'].inference(inputs=[o1])[0]
        o2b = self.rknns['part2b'].inference(inputs=[o2a])[0]
        o3 = self.rknns['part3'].inference(inputs=[o2b])[0]
        return o3

    def transcribe(self, audio):
        output = self.inference(audio)
        return decode(output, self.id2tok)

    def release(self):
        for rknn in self.rknns.values():
            rknn.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', nargs='?', default=f'{BASE}/input/call_elevator.wav')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--split', default='split11', choices=list(SPLIT_CONFIGS.keys()),
                        help='Split point (default: split11)')
    parser.add_argument('--core_mask', type=int, default=7,
                        help='1=core0, 3=core0+1, 7=all3')
    args = parser.parse_args()

    audio = load_audio(args.wav)
    model = SplitWav2Vec2(split=args.split, core_mask=args.core_mask)

    # Warmup
    model.inference(audio)

    if args.bench:
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.inference(audio)
            times.append((time.perf_counter() - t0) * 1000)
        text = model.transcribe(audio)
        print(f"Text: [{text}]")
        print(f"Median: {np.median(times):.1f}ms  Min: {min(times):.1f}ms  RTF: {np.median(times)/5000:.3f}")
    else:
        t0 = time.perf_counter()
        text = model.transcribe(audio)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[{text}]  ({elapsed:.0f}ms)")

    model.release()
