"""
Fbank 파라미터 변형 비교: 어떤 설정이 모델과 가장 잘 맞는지 테스트
- preemphasis (0.97 vs 없음)
- energy_floor (1.0 vs 1e-10)
- mean subtract (있음 vs 없음)
"""
import numpy as np, sys, glob
import onnxruntime as ort
import soundfile as sf

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
sys.path.insert(0, f'{BASE}/rk3588')
from fbank import _povey_window, _mel_filterbank

ENC_SCHEMA = [
    ('x',[1,39,80],'float32'),
    ('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]
CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']
CHUNK = 39; OFFSET = 32; CONTEXT_SIZE = 2; BLANK_ID = 0; UNK_ID = 2

_window = _povey_window(400)
_filters = _mel_filterbank(80, 512, 16000, 20.0, 7600.0)

def compute_fbank(audio, preemph=0.0, energy_floor=1e-10, subtract_mean=True):
    audio = audio.astype(np.float32)
    if preemph > 0:
        audio = np.append(audio[0:1], audio[1:] - preemph * audio[:-1])
    half = 200  # frame_length // 2 = 400 // 2
    padded = np.pad(audio, (half, half), mode='reflect')
    frames = []
    i = 0
    while i + 400 <= len(padded):
        frame = padded[i:i+400].copy()
        if subtract_mean:
            frame -= frame.mean()
        frame = frame * _window
        spectrum = np.fft.rfft(frame, n=512)
        power = np.real(spectrum)**2 + np.imag(spectrum)**2
        mel = np.dot(_filters, power)
        mel = np.maximum(mel, energy_floor)
        frames.append(np.log(mel))
        i += 160
    return np.stack(frames).astype(np.float32) if frames else np.zeros((0,80),dtype=np.float32)

def load_vocab():
    vocab = {}
    with open(f'{BASE}/tokens.txt') as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2: vocab[int(p[1])] = p[0]
    return vocab

def decode(hyp, vocab):
    return ''.join(vocab.get(i,'') for i in hyp).replace('▁',' ').strip()

def cer(ref, hyp):
    ref = ref.replace(' ','').replace('.','').replace('?','').replace('!','')
    hyp = hyp.replace(' ','').replace('.','').replace('?','').replace('!','')
    if not ref: return 0.0
    r, h = list(ref), list(hyp)
    R, H = len(r), len(h)
    d = np.zeros((R+1,H+1),dtype=int)
    for i in range(R+1): d[i,0]=i
    for j in range(H+1): d[0,j]=j
    for i in range(1,R+1):
        for j in range(1,H+1):
            d[i,j]=min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(0 if r[i-1]==h[j-1] else 1))
    return d[R,H]/R

vocab = load_vocab()
opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
enc_s = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', sess_options=opts, providers=['CPUExecutionProvider'])
dec_s = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
joi_s = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])

gt = {}
with open(f'{BASE}/test_wavs/trans.txt') as f:
    for line in f:
        nm, *rest = line.strip().split(' ', 1)
        gt[nm] = rest[0] if rest else ''

def transcribe_with_feats(feats):
    T = feats.shape[0]
    pad = max(CHUNK, ((T//OFFSET)+1)*OFFSET) - T
    feats = np.vstack([feats, np.zeros((pad,80),dtype=np.float32)])
    state = {nm: np.zeros(sh,dtype=np.int64) if dt=='int64' else np.zeros(sh,dtype=np.float32) for nm,sh,dt in ENC_SCHEMA}
    hyp = [BLANK_ID]*CONTEXT_SIZE
    dec_out = dec_s.run(None, {'y': np.array([hyp],dtype=np.int64)})[0]
    num_proc = 0
    while num_proc+CHUNK <= feats.shape[0]:
        state['x'] = feats[num_proc:num_proc+CHUNK][np.newaxis]
        onnx_in = {nm: state[nm] for nm,_,_ in ENC_SCHEMA}
        out = enc_s.run(None, onnx_in)
        enc_out = np.array(out[0]).squeeze(0)
        for i,nm in enumerate(CACHE_NAMES): state[nm]=np.array(out[i+1])
        for t in range(enc_out.shape[0]):
            joi_out = joi_s.run(None, {'encoder_out': enc_out[t:t+1].astype(np.float32), 'decoder_out': dec_out.astype(np.float32).reshape(1,512)})[0]
            y = int(np.argmax(joi_out.squeeze()))
            if y not in (BLANK_ID, UNK_ID):
                hyp.append(y)
                dec_out = dec_s.run(None, {'y': np.array([hyp[-CONTEXT_SIZE:]],dtype=np.int64)})[0]
        num_proc += OFFSET
    return decode(hyp[CONTEXT_SIZE:], vocab)

configs = [
    ('No-preemph, mean-sub, floor=1e-10 (current)', dict(preemph=0.0, energy_floor=1e-10, subtract_mean=True)),
    ('No-preemph, mean-sub, floor=1.0',             dict(preemph=0.0, energy_floor=1.0,   subtract_mean=True)),
    ('No-preemph, no-mean,  floor=1e-10',           dict(preemph=0.0, energy_floor=1e-10, subtract_mean=False)),
    ('No-preemph, no-mean,  floor=1.0',             dict(preemph=0.0, energy_floor=1.0,   subtract_mean=False)),
    ('Preemph=0.97, no-mean, floor=1e-10',          dict(preemph=0.97, energy_floor=1e-10, subtract_mean=False)),
    ('Preemph=0.97, no-mean, floor=1.0',            dict(preemph=0.97, energy_floor=1.0,   subtract_mean=False)),
    ('Preemph=0.97, mean-sub, floor=1.0',           dict(preemph=0.97, energy_floor=1.0,   subtract_mean=True)),
]

wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
audios = {wav: sf.read(wav, dtype='float32')[0] for wav in wavs}

print(f'{"Config":<50} | avg_cer | per-file CER')
print('-'*90)

for name, cfg in configs:
    cers = []
    for wav in wavs:
        fname = __import__('os').path.basename(wav)
        audio = audios[wav]
        feats = compute_fbank(audio, **cfg)
        text = transcribe_with_feats(feats)
        c = cer(gt.get(fname,''), text)
        cers.append(c)
    avg = np.mean(cers)
    per = '  '.join([f'{c*100:.0f}%' for c in cers])
    print(f'{name:<50} | {avg*100:5.1f}%  | {per}')
