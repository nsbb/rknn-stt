"""
Kaldi-compatible Fbank (80-bin) 구현 (numpy+scipy)
kaldifeat/torchaudio 없이 동작
참고: sherpa-onnx / kaldifeat 와 동일한 파라미터

Parameters (kaldifeat defaults used in zipformer):
  sample_rate = 16000
  frame_length = 25ms  (400 samples)
  frame_shift  = 10ms  (160 samples)
  fft_size     = 512
  num_mel_bins = 80
  low_freq     = 20 Hz
  high_freq    = 7600 Hz  (= 8000 - 400)
  dither       = 0
  snip_edges   = False (center-pad)
  window       = Povey (= raised cosine, power 0.85)
  preemphasis  = 0.97
  use_energy   = False

변경 이력:
  v2: preemphasis=0.97 추가, frame mean 제거 → CER 26.2% → 19.9%
"""

import numpy as np


def _mel(f):
    """Hz → Mel"""
    return 1127.0 * np.log(1.0 + f / 700.0)


def _mel_inv(m):
    """Mel → Hz"""
    return 700.0 * (np.exp(m / 1127.0) - 1.0)


def _povey_window(n):
    """Kaldi Povey window: raised-cosine ^ 0.85"""
    a = 2 * np.pi / (n - 1)
    w = (0.5 - 0.5 * np.cos(a * np.arange(n))) ** 0.85
    return w.astype(np.float32)


def _mel_filterbank(num_bins, fft_size, sample_rate, low_freq, high_freq):
    """
    Kaldi-style triangular mel filterbank.
    Returns (num_bins, fft_size//2+1) weight matrix.
    """
    nyquist = sample_rate / 2.0
    mel_low  = _mel(low_freq)
    mel_high = _mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, num_bins + 2)
    hz_points  = _mel_inv(mel_points)

    # 주파수 빈 인덱스
    fft_bins = np.floor(hz_points / nyquist * (fft_size // 2)).astype(int)

    filters = np.zeros((num_bins, fft_size // 2 + 1), dtype=np.float32)
    for m in range(1, num_bins + 1):
        f_left   = fft_bins[m - 1]
        f_center = fft_bins[m]
        f_right  = fft_bins[m + 1]
        for k in range(f_left, f_center + 1):
            if f_center > f_left:
                filters[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right + 1):
            if f_right > f_center:
                filters[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filters


class KaldiFbank:
    """
    Kaldi-compatible 80-bin log-Mel filterbank feature extractor.
    쌓인 오디오에서 online-style로 프레임을 뽑는다.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        fft_size: int = 512,
        low_freq: float = 20.0,
        high_freq: float = 7600.0,   # 8000 - 400
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = 0.0,
        preemphasis: float = 0.97,
    ):
        self.sample_rate    = sample_rate
        self.num_mel_bins   = num_mel_bins
        self.frame_length   = int(sample_rate * frame_length_ms / 1000)   # 400
        self.frame_shift    = int(sample_rate * frame_shift_ms  / 1000)   # 160
        self.fft_size       = fft_size
        self.dither         = dither
        self.snip_edges     = snip_edges
        self.energy_floor   = energy_floor
        self.preemphasis    = preemphasis

        self._window  = _povey_window(self.frame_length)
        self._filters = _mel_filterbank(num_mel_bins, fft_size, sample_rate, low_freq, high_freq)

        self._buf = np.array([], dtype=np.float32)

    def accept_waveform(self, waveform: np.ndarray):
        """오디오 추가 (float32, 범위 [-1, 1] 또는 raw PCM)"""
        self._buf = np.concatenate([self._buf, waveform.astype(np.float32)])

    def num_frames_ready(self) -> int:
        n = len(self._buf)
        if self.snip_edges:
            if n < self.frame_length:
                return 0
            return 1 + (n - self.frame_length) // self.frame_shift
        else:
            # center-padded: 첫 프레임 중심이 샘플 0
            return (n + self.frame_shift - 1) // self.frame_shift

    def compute_all(self, waveform: np.ndarray) -> np.ndarray:
        """
        전체 오디오에 대해 한 번에 fbank 특징 계산 (벡터화 FFT).
        Returns: (T, num_mel_bins) float32
        """
        audio = waveform.astype(np.float32)
        if self.preemphasis > 0:
            audio = np.append(audio[0:1], audio[1:] - self.preemphasis * audio[:-1])
        if self.dither > 0:
            audio = audio + self.dither * np.random.randn(len(audio)).astype(np.float32)
        half = self.frame_length // 2
        if not self.snip_edges:
            audio = np.pad(audio, (half, half), mode='reflect')

        n = len(audio)
        if n < self.frame_length:
            return np.zeros((0, self.num_mel_bins), dtype=np.float32)

        # Stride-trick으로 모든 프레임 한 번에 추출
        n_frames = (n - self.frame_length) // self.frame_shift + 1
        strides = (audio.strides[0] * self.frame_shift, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, self.frame_length),
            strides=strides
        ).copy()  # copy() 필요: stride_tricks는 가변 뷰

        # 윈도우 + 벡터화 FFT
        frames *= self._window[np.newaxis, :]
        spectra = np.fft.rfft(frames, n=self.fft_size, axis=1)
        power = np.real(spectra) ** 2 + np.imag(spectra) ** 2  # (T, fft_size//2+1)

        # Mel filterbank
        mel = power @ self._filters.T                           # (T, num_mel_bins)
        mel = np.maximum(mel, 1e-10)
        return np.log(mel).astype(np.float32)


def extract_features(wav_path: str, num_mel_bins: int = 80) -> np.ndarray:
    """WAV 파일에서 fbank 특징 추출 → (T, 80)"""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
    fbank = KaldiFbank()
    return fbank.compute_all(audio)


if __name__ == '__main__':
    # sanity check
    np.random.seed(0)
    audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1초
    fb = KaldiFbank()
    feats = fb.compute_all(audio)
    print(f"Input: {len(audio)} samples = 1s")
    print(f"Output: {feats.shape}  (expected ~100 frames x 80 bins)")
    print(f"Mean: {feats.mean():.3f}, Std: {feats.std():.3f}")
