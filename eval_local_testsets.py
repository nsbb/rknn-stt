"""Evaluate wav2vec2 (split INT8-KL) and citrinet (FP16) on local testsets.

Usage:
    conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model wav2vec2
    conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model citrinet
    conda run -n RKNN-Toolkit2 python eval_local_testsets.py --model all

Output:
    eval_results/<model>/  per-testset CSVs + summary
"""
import numpy as np
import os, sys, csv, time, argparse, unicodedata, re

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = os.path.dirname(os.path.abspath(__file__))
TESTSET_DIR = os.path.join(BASE, 'testset')
OUT_BASE = os.path.join(BASE, 'eval_results')

TESTSETS = [
    '7F_KSK',
    '7F_HJY',
    'modelhouse_2m',
    'modelhouse_2m_noheater',
    'modelhouse_3m',
]


def normalize_text(text):
    """Normalize text for CER comparison."""
    text = text.strip()
    # Remove punctuation
    text = re.sub(r'[?!.,;:~\-\'\"(){}[\]…·]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def char_error_rate(ref, hyp):
    """Compute CER using edit distance."""
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Levenshtein distance
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / len(r) * 100.0


def load_testset(name):
    """Load testset CSV. Returns list of (filepath, gt_text)."""
    csv_path = os.path.join(TESTSET_DIR, f'{name}.csv')
    entries = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row['FileName'], row['gt']))
    return entries


class Wav2Vec2Evaluator:
    MODEL_FILES = {
        'part1': 'wav2vec2_part1_features_fp16.rknn',
        'part2a': 'wav2vec2_part2a_int8_kl.rknn',
        'part2b': 'wav2vec2_part2b_fp16.rknn',
        'part3': 'wav2vec2_part3_lmhead_fp16.rknn',
    }
    TOTAL_SIZE = '462MB (13+160+295+5.2)'

    def __init__(self):
        import soundfile as sf
        import scipy.signal
        import json
        self.sf = sf
        self.scipy_signal = scipy.signal
        from rknnlite.api import RKNNLite
        model_dir = os.path.join(BASE, 'wav2vec2', 'model')
        self.rknns = {}
        for name, fname in self.MODEL_FILES.items():
            path = os.path.join(model_dir, fname)
            rknn = RKNNLite(verbose=False)
            assert rknn.load_rknn(path) == 0, f"Failed: {path}"
            assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
            self.rknns[name] = rknn
        vocab_path = os.path.join(BASE, 'wav2vec2', 'json', 'vocab.json')
        with open(vocab_path) as f:
            vocab = json.load(f)
        self.id2tok = {int(v): k for k, v in vocab.items()}

    def load_audio(self, path, max_len=80000):
        audio, sr = self.sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = self.scipy_signal.resample(audio, int(len(audio) * 16000 / sr))
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 5.0
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        return audio[np.newaxis]

    def transcribe(self, audio_path):
        audio = self.load_audio(audio_path)
        t0 = time.perf_counter()
        o1 = self.rknns['part1'].inference(inputs=[audio])[0]
        o2a = self.rknns['part2a'].inference(inputs=[o1])[0]
        o2b = self.rknns['part2b'].inference(inputs=[o2a])[0]
        o3 = self.rknns['part3'].inference(inputs=[o2b])[0]
        elapsed = (time.perf_counter() - t0) * 1000
        # Decode
        ids = np.argmax(o3, axis=-1)[0]
        prev = -1
        chars = []
        for i in ids:
            if i != prev:
                tok = self.id2tok.get(i, '')
                if tok == '|':
                    chars.append(' ')
                elif tok not in ['', '<pad>', '<s>', '</s>', '<unk>'] and i >= 5:
                    chars.append(tok)
            prev = i
        text = ''.join(chars)
        return text, elapsed

    def release(self):
        for rknn in self.rknns.values():
            rknn.release()

    @staticmethod
    def model_description():
        return [
            "wav2vec2 xls-r-300m (Large, 300M params) Split INT8-KL",
            "원본: facebook/wav2vec2-xls-r-300m + 한국어 fine-tuned CTC head",
            "양자화: Split INT8 (Layer 0-11: INT8-KL, Layer 12-23: FP16)",
            "구성: Part1(CNN FP16) → Part2A(Encoder L0-11, INT8-KL) → Part2B(Encoder L12-23, FP16) → Part3(LM Head, FP16)",
        ]


class CitriNetEvaluator:
    MODEL_FILE = 'citrinet_fp16.rknn'
    MODEL_SIZE = '281MB'

    def __init__(self):
        import soundfile as sf
        self.sf = sf
        from rknnlite.api import RKNNLite
        # Import mel computation from citrinet inference
        sys.path.insert(0, os.path.join(BASE, 'ko_citrinet', 'python'))
        from inference_rknn import compute_mel_spectrogram, load_vocab, decode_ctc
        self.compute_mel = compute_mel_spectrogram
        self.vocab = load_vocab()
        self.decode_ctc = decode_ctc

        model_path = os.path.join(BASE, 'ko_citrinet', 'model', self.MODEL_FILE)
        self.rknn = RKNNLite(verbose=False)
        assert self.rknn.load_rknn(model_path) == 0, f"Failed: {model_path}"
        assert self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

    def load_audio(self, wav_path, target_frames=300):
        audio, sr = self.sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            ratio = 16000 / sr
            n_samples = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_samples),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
        mel = self.compute_mel(audio)  # [T, 80]
        if mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - mel.shape[0], 80))
            mel = np.concatenate([mel, pad], axis=0)
        else:
            mel = mel[:target_frames]
        return mel.T[np.newaxis, :, :].astype(np.float32)  # [1, 80, 300]

    def transcribe(self, audio_path):
        mel = self.load_audio(audio_path)
        t0 = time.perf_counter()
        logits = self.rknn.inference(inputs=[mel])[0]
        elapsed = (time.perf_counter() - t0) * 1000
        text = self.decode_ctc(logits, self.vocab)
        return text, elapsed

    def release(self):
        self.rknn.release()

    @staticmethod
    def model_description():
        return [
            "CitriNet CTC (NeMo, Jasper-based CNN)",
            "원본: NeMo CitriNet 한국어 CTC (SentencePiece BPE 2048 tokens)",
            "양자화: FP16 (양자화 없음)",
            "ONNX 그래프 수정 4가지: LogSoftmax 제거, SE→ReduceMean, ReduceMean→Conv, Squeeze 제거",
        ]


def evaluate_model(evaluator, model_name, model_files_str):
    """Run evaluation on all testsets."""
    out_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    total_cer_sum = 0
    total_count = 0

    for ts_name in TESTSETS:
        entries = load_testset(ts_name)
        print(f"\n=== {ts_name} ({len(entries)} samples) ===")

        results = []
        cer_sum = 0
        time_sum = 0

        for i, (filepath, gt) in enumerate(entries):
            try:
                hyp, elapsed = evaluator.transcribe(filepath)
                cer = char_error_rate(gt, hyp)
                results.append({
                    'file': os.path.basename(filepath),
                    'ref': gt,
                    'hyp': hyp,
                    'cer': round(cer, 1),
                    'time_ms': round(elapsed, 0),
                })
                cer_sum += cer
                time_sum += elapsed
                if (i + 1) % 20 == 0 or (i + 1) == len(entries):
                    avg_cer = cer_sum / (i + 1)
                    print(f"  [{i+1}/{len(entries)}] avg CER={avg_cer:.1f}% avg_time={time_sum/(i+1):.0f}ms")
            except Exception as e:
                print(f"  ERROR {os.path.basename(filepath)}: {e}")
                results.append({
                    'file': os.path.basename(filepath),
                    'ref': gt,
                    'hyp': f'ERROR: {e}',
                    'cer': 100.0,
                    'time_ms': 0,
                })
                cer_sum += 100.0

        avg_cer = cer_sum / len(entries) if entries else 0
        avg_time = time_sum / len(entries) if entries else 0
        total_cer_sum += cer_sum
        total_count += len(entries)

        all_results[ts_name] = {
            'results': results,
            'avg_cer': avg_cer,
            'avg_time': avg_time,
            'count': len(entries),
        }

        # Write per-testset CSV
        csv_path = os.path.join(out_dir, f'{ts_name}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'ref', 'hyp', 'cer', 'time_ms'])
            writer.writeheader()
            writer.writerows(results)

        print(f"  => CER={avg_cer:.1f}%  avg_time={avg_time:.0f}ms  saved: {csv_path}")

    # Write summary
    overall_cer = total_cer_sum / total_count if total_count else 0
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Files: {model_files_str}\n")
        for line in evaluator.model_description():
            f.write(f"  {line}\n")
        f.write(f"\n{'Testset':<30} {'Samples':>7} {'CER':>8} {'Avg ms':>8}\n")
        f.write('-' * 60 + '\n')
        for ts_name in TESTSETS:
            r = all_results[ts_name]
            f.write(f"{ts_name:<30} {r['count']:>7} {r['avg_cer']:>7.1f}% {r['avg_time']:>7.0f}\n")
        f.write('-' * 60 + '\n')
        f.write(f"{'TOTAL':<30} {total_count:>7} {overall_cer:>7.1f}%\n")

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_count} samples, CER={overall_cer:.1f}%")
    print(f"Summary: {summary_path}")

    return all_results, overall_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['wav2vec2', 'citrinet', 'all'])
    args = parser.parse_args()

    models_to_eval = ['wav2vec2', 'citrinet'] if args.model == 'all' else [args.model]

    for model_name in models_to_eval:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {model_name}")
        print(f"{'#'*60}")

        if model_name == 'wav2vec2':
            evaluator = Wav2Vec2Evaluator()
            files_str = ' + '.join(f'{v}' for v in Wav2Vec2Evaluator.MODEL_FILES.values())
            evaluate_model(evaluator, model_name, files_str)
            evaluator.release()
        elif model_name == 'citrinet':
            evaluator = CitriNetEvaluator()
            evaluate_model(evaluator, model_name, CitriNetEvaluator.MODEL_FILE)
            evaluator.release()


if __name__ == '__main__':
    main()
