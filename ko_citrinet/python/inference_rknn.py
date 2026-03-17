"""CitriNet RKNN inference on RK3588 NPU.

Usage:
    conda run -n RKNN-Toolkit2 python inference_rknn.py ../path/to/audio.wav
    conda run -n RKNN-Toolkit2 python inference_rknn.py --bench
"""
import numpy as np
import sys
import os
import time
import argparse
import glob

# NeMo mel spectrogram parameters (from model_config_ko.yaml)
SAMPLE_RATE = 16000
N_FFT = 512
WIN_SIZE = int(0.025 * SAMPLE_RATE)  # 400 samples (25ms)
HOP_SIZE = int(0.01 * SAMPLE_RATE)   # 160 samples (10ms)
N_MELS = 80
DITHER = 1e-5
TARGET_FRAMES = 300  # Fixed input length

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'tokenizer.model')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vocab_ko.txt')


def mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=None):
    """Create mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)

    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_freqs)

    fb = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]
        for j in range(n_freqs):
            if lower <= fftfreqs[j] <= center:
                fb[i, j] = (fftfreqs[j] - lower) / (center - lower)
            elif center < fftfreqs[j] <= upper:
                fb[i, j] = (upper - fftfreqs[j]) / (upper - center)
    return fb


def compute_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Compute 80-dim mel spectrogram matching NeMo CitriNet preprocessing."""
    # Dither
    audio = audio + DITHER * np.random.randn(len(audio))

    # Pre-emphasis (NeMo default: 0.97)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame
    n_frames = 1 + (len(audio) - WIN_SIZE) // HOP_SIZE
    frames = np.zeros((n_frames, WIN_SIZE))
    for i in range(n_frames):
        start = i * HOP_SIZE
        frames[i] = audio[start:start + WIN_SIZE]

    # Hann window
    window = np.hanning(WIN_SIZE)
    frames *= window

    # FFT
    fft = np.fft.rfft(frames, n=N_FFT)
    power = np.abs(fft) ** 2

    # Mel filterbank
    fb = mel_filterbank(sr, N_FFT, N_MELS)
    mel = np.dot(power, fb.T)

    # Log
    mel = np.log(mel + 1e-10)

    # Per-feature normalization (NeMo default)
    mel = (mel - mel.mean(axis=0)) / (mel.std(axis=0) + 1e-5)

    return mel  # [n_frames, 80]


def load_audio(wav_path, target_frames=TARGET_FRAMES):
    """Load audio, compute mel, pad/truncate to target frames."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        # Simple resample
        ratio = SAMPLE_RATE / sr
        n_samples = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    mel = compute_mel_spectrogram(audio)  # [T, 80]

    # Pad or truncate to target_frames
    if mel.shape[0] < target_frames:
        pad = np.zeros((target_frames - mel.shape[0], N_MELS))
        mel = np.concatenate([mel, pad], axis=0)
    else:
        mel = mel[:target_frames]

    # Reshape to model input format: [1, 80, 300]
    mel = mel.T[np.newaxis, :, :]  # [1, 80, 300]
    return mel.astype(np.float32)


def load_vocab(vocab_path=VOCAB_PATH):
    """Load vocabulary from vocab_ko.txt."""
    tokens = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                # Remove ## prefix (continuation marker)
                if token.startswith('##'):
                    token = token[2:]
                tokens.append(token)
    return tokens


def decode_ctc(logits, vocab, blank_id=None):
    """CTC greedy decoding.

    logits: [2049, 38] or [1, 2049, 38] or [1, 2049, 1, 38]
    """
    if logits.ndim == 4:
        logits = logits[0, :, 0, :]  # [2049, 38]
    elif logits.ndim == 3:
        logits = logits[0]  # [2049, 38]

    # argmax over vocab dimension
    ids = np.argmax(logits, axis=0)  # [38]

    if blank_id is None:
        blank_id = logits.shape[0] - 1  # Last token as blank (CTC convention)

    # CTC collapse: remove consecutive duplicates and blanks
    prev = -1
    token_ids = []
    for idx in ids:
        if idx != prev:
            if idx != blank_id and idx != 0:  # 0 might be <unk> or blank
                token_ids.append(int(idx))
            prev = idx
        else:
            prev = idx

    # Try SentencePiece decoding first
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_PATH)
        # SentencePiece token IDs: model output index - 1 (blank offset)
        sp_ids = [tid - 1 for tid in token_ids if tid - 1 >= 0]
        text = sp.decode(sp_ids)
        return text
    except Exception:
        pass

    # Fallback: vocab lookup
    text_parts = []
    for tid in token_ids:
        if 0 < tid <= len(vocab):
            text_parts.append(vocab[tid - 1])
    return ''.join(text_parts).replace('▁', ' ').strip()


class CitriNetRKNN:
    def __init__(self, model_path=None, core_mask=None):
        from rknnlite.api import RKNNLite
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'citrinet_fp16.rknn')
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0_1_2

        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'Failed to load RKNN model: {ret}')

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f'Failed to init runtime: {ret}')

        self.core_mask = core_mask

    def inference(self, mel_input):
        """Run inference. mel_input: [1, 80, 300]"""
        outputs = self.rknn.inference(inputs=[mel_input])
        return outputs[0]  # [1, 2049, 38]

    def release(self):
        self.rknn.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', nargs='?', help='WAV file path')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    parser.add_argument('--model', default=None, help='RKNN model path')
    parser.add_argument('--n_iter', type=int, default=20, help='Benchmark iterations')
    args = parser.parse_args()

    vocab = load_vocab()
    model = CitriNetRKNN(model_path=args.model)

    if args.bench:
        # Benchmark with random input
        print(f'Benchmarking {args.n_iter} iterations...')
        mel = np.random.randn(1, 80, 300).astype(np.float32)
        # Warmup
        for _ in range(3):
            model.inference(mel)

        times = []
        for _ in range(args.n_iter):
            t0 = time.time()
            model.inference(mel)
            times.append((time.time() - t0) * 1000)

        times.sort()
        print(f'Latency: median={np.median(times):.1f}ms, '
              f'min={min(times):.1f}ms, max={max(times):.1f}ms')
        print(f'RTF: {np.median(times)/1000 / 3.0:.4f} (3s audio)')

    elif args.wav:
        mel = load_audio(args.wav)
        t0 = time.time()
        logits = model.inference(mel)
        elapsed = (time.time() - t0) * 1000
        text = decode_ctc(logits, vocab)
        print(f'[{elapsed:.0f}ms] {text}')

    else:
        # Test with available wav files
        test_wavs = glob.glob('/home/rk3588/travail/rk3588/rknn-stt/wav2vec2/input/*.wav')
        for wav_path in sorted(test_wavs):
            mel = load_audio(wav_path)
            t0 = time.time()
            logits = model.inference(mel)
            elapsed = (time.time() - t0) * 1000
            text = decode_ctc(logits, vocab)
            print(f'[{elapsed:.0f}ms] {os.path.basename(wav_path)}: {text}')

    model.release()


if __name__ == '__main__':
    main()
