"""
최종 종합 벤치마크
- ONNX FP32 vs INT8 vs RKNN Hybrid
- 실제 오디오로 end-to-end 레이턴시 측정
- 청크당 시간 + 파일당 RTF
"""
import numpy as np, time, sys, glob, os
import soundfile as sf
import onnxruntime as ort
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from fbank import KaldiFbank

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
CHUNK=39; OFFSET=32; BLANK_ID=0; UNK_ID=2; CTX=2

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

def nchw2nhwc(a): return np.transpose(a, (0,2,3,1))

vocab = {}
with open(f'{BASE}/tokens.txt') as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2: vocab[int(p[1])] = p[0]

gt = {}
with open(f'{BASE}/test_wavs/trans.txt') as f:
    for line in f:
        nm, *rest = line.strip().split(' ', 1)
        gt[nm] = rest[0] if rest else ''

def decode(hyp): return ''.join(vocab.get(i,'') for i in hyp).replace('▁',' ').strip()

def cer(ref, hyp):
    ref = ref.replace(' ','').replace('.','').replace('?','').replace('!','')
    hyp = hyp.replace(' ','').replace('.','').replace('?','').replace('!','')
    if not ref: return 0.0
    r, h = list(ref), list(hyp); R, H = len(r), len(h)
    d = np.zeros((R+1,H+1), dtype=int)
    for i in range(R+1): d[i,0]=i
    for j in range(H+1): d[0,j]=j
    for i in range(1,R+1):
        for j in range(1,H+1):
            d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(0 if r[i-1]==h[j-1] else 1))
    return d[R,H]/R

def init_state():
    return {nm: np.zeros(sh,dtype=np.int64) if dt=='int64' else np.zeros(sh,dtype=np.float32) for nm,sh,dt in ENC_SCHEMA}

wavs = sorted(glob.glob(f'{BASE}/test_wavs/*.wav'))
fb = KaldiFbank()

# ── ONNX FP32 ────────────────────────────────────────────────
print('Loading ONNX FP32 ...')
opts4 = ort.SessionOptions(); opts4.intra_op_num_threads = 4
enc_fp32 = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', sess_options=opts4, providers=['CPUExecutionProvider'])
dec_fp32 = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
joi_fp32 = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.onnx',  providers=['CPUExecutionProvider'])

# ── ONNX INT8 ────────────────────────────────────────────────
print('Loading ONNX INT8 ...')
enc_int8 = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.int8.onnx', sess_options=opts4, providers=['CPUExecutionProvider'])
dec_int8 = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.int8.onnx', providers=['CPUExecutionProvider'])
joi_int8 = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.int8.onnx',  providers=['CPUExecutionProvider'])

def run_onnx(enc_s, dec_s, joi_s, wav_path):
    audio, _ = sf.read(wav_path, dtype='float32')
    t_feat = time.perf_counter()
    feats = fb.compute_all(audio)
    feat_ms = (time.perf_counter() - t_feat) * 1000

    T = feats.shape[0]
    pad = max(CHUNK, ((T//OFFSET)+1)*OFFSET) - T
    feats = np.vstack([feats, np.zeros((pad,80),dtype=np.float32)])

    state = init_state()
    hyp = [BLANK_ID] * CTX
    dec_out = dec_s.run(None, {'y': np.array([hyp],dtype=np.int64)})[0]

    enc_ms, joi_ms = [], []
    num_proc = 0
    while num_proc+CHUNK <= feats.shape[0]:
        state['x'] = feats[num_proc:num_proc+CHUNK][np.newaxis]
        t0 = time.perf_counter()
        out = enc_s.run(None, {nm:state[nm] for nm,_,_ in ENC_SCHEMA})
        enc_ms.append((time.perf_counter()-t0)*1000)
        enc_out = np.array(out[0]).squeeze(0)
        for i,nm in enumerate(CACHE_NAMES): state[nm]=np.array(out[i+1])

        for t in range(enc_out.shape[0]):
            t0 = time.perf_counter()
            joi_out = joi_s.run(None, {'encoder_out': enc_out[t:t+1].astype(np.float32), 'decoder_out': dec_out.astype(np.float32).reshape(1,512)})[0]
            joi_ms.append((time.perf_counter()-t0)*1000)
            y = int(np.argmax(joi_out.squeeze()))
            if y not in (BLANK_ID, UNK_ID):
                hyp.append(y)
                dec_out = dec_s.run(None, {'y': np.array([hyp[-CTX:]],dtype=np.int64)})[0]
        num_proc += OFFSET

    audio_ms = len(audio)/16000*1000
    total_enc = sum(enc_ms)
    total_joi = sum(joi_ms)
    return {
        'text': decode(hyp[CTX:]),
        'feat_ms': feat_ms,
        'enc_median': float(np.median(enc_ms)),
        'joi_median': float(np.median(joi_ms)),
        'enc_total': total_enc,
        'joi_total': total_joi,
        'chunks': len(enc_ms),
        'audio_ms': audio_ms,
        'rtf': (total_enc+total_joi)/audio_ms,
    }

for label, enc_s, dec_s, joi_s in [
    ('ONNX FP32', enc_fp32, dec_fp32, joi_fp32),
    ('ONNX INT8', enc_int8, dec_int8, joi_int8),
]:
    print(f'\n{"="*60}')
    print(f'Model: {label}')
    print(f'{"="*60}')
    all_stats = []
    for wav in wavs:
        fname = os.path.basename(wav)
        stats = run_onnx(enc_s, dec_s, joi_s, wav)
        c = cer(gt.get(fname,''), stats['text'])
        all_stats.append({'cer': c, **stats})
        print(f'[{fname}] CER={c*100:.1f}%  enc_med={stats["enc_median"]:.0f}ms  joi_med={stats["joi_median"]:.2f}ms  RTF={stats["rtf"]:.3f}')
        print(f'  HYP: {stats["text"]}')

    avg_cer = np.mean([s['cer'] for s in all_stats])
    avg_enc = np.mean([s['enc_median'] for s in all_stats])
    avg_joi = np.mean([s['joi_median'] for s in all_stats])
    avg_rtf = np.mean([s['rtf'] for s in all_stats])
    avg_feat = np.mean([s['feat_ms'] for s in all_stats])
    print(f'\n  AVG CER:       {avg_cer*100:.2f}%')
    print(f'  Enc/chunk:     {avg_enc:.1f}ms')
    print(f'  Joi/frame:     {avg_joi:.3f}ms')
    print(f'  Fbank/file:    {avg_feat:.1f}ms')
    print(f'  RTF (enc+joi): {avg_rtf:.3f}')

# ── RKNN Hybrid ──────────────────────────────────────────────
print(f'\n{"="*60}')
print('Model: RKNN Hybrid (RKNN encoder_out + ONNX FP32 cache)')
print(f'{"="*60}')

try:
    from rknnlite.api import RKNNLite
    RKNN_DIR = f'{BASE}/rk3588'

    def _load(path):
        m = RKNNLite(verbose=False)
        assert m.load_rknn(path) == 0
        assert m.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
        return m

    enc_r = _load(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
    dec_r = _load(f'{RKNN_DIR}/decoder-epoch-99-avg-1.rknn')
    joi_r = _load(f'{RKNN_DIR}/joiner-epoch-99-avg-1.rknn')
    enc_s_onnx = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', sess_options=opts4, providers=['CPUExecutionProvider'])

    hybrid_stats = []
    for wav in wavs:
        fname = os.path.basename(wav)
        audio, _ = sf.read(wav, dtype='float32')
        feats = fb.compute_all(audio)
        T = feats.shape[0]
        pad = max(CHUNK, ((T//OFFSET)+1)*OFFSET) - T
        feats = np.vstack([feats, np.zeros((pad,80),dtype=np.float32)])

        state = init_state()
        hyp = [BLANK_ID]*CTX
        dec_out = np.array(dec_r.inference(inputs=[np.array([hyp],dtype=np.int64)])[0])

        enc_r_ms, enc_o_ms, joi_ms2 = [], [], []
        num_proc = 0
        while num_proc+CHUNK <= feats.shape[0]:
            state['x'] = feats[num_proc:num_proc+CHUNK][np.newaxis]
            rknn_in = []
            for nm,sh,_ in ENC_SCHEMA:
                a=state[nm]
                if len(sh)==4: a=nchw2nhwc(a)
                rknn_in.append(a)

            t0 = time.perf_counter()
            rknn_out = enc_r.inference(inputs=rknn_in)
            enc_r_ms.append((time.perf_counter()-t0)*1000)
            enc_out = np.array(rknn_out[0]).squeeze(0)

            t0 = time.perf_counter()
            onnx_out = enc_s_onnx.run(None, {nm:state[nm] for nm,_,_ in ENC_SCHEMA})
            enc_o_ms.append((time.perf_counter()-t0)*1000)
            for i,nm in enumerate(CACHE_NAMES): state[nm]=np.array(onnx_out[i+1])

            for t in range(enc_out.shape[0]):
                cur = enc_out[t:t+1].astype(np.float32)
                dec_f = dec_out.astype(np.float32).reshape(1,512)
                t0 = time.perf_counter()
                joi_out = np.array(joi_r.inference(inputs=[cur, dec_f])[0])
                joi_ms2.append((time.perf_counter()-t0)*1000)
                y = int(np.argmax(joi_out.squeeze()))
                if y not in (BLANK_ID, UNK_ID):
                    hyp.append(y)
                    dec_out = np.array(dec_r.inference(inputs=[np.array([hyp[-CTX:]],dtype=np.int64)])[0])
            num_proc += OFFSET

        text = decode(hyp[CTX:])
        c = cer(gt.get(fname,''), text)
        audio_ms = len(audio)/16000*1000
        total = sum(enc_r_ms)+sum(enc_o_ms)+sum(joi_ms2)
        hybrid_stats.append({'cer':c,'rtf':total/audio_ms,'enc_r':np.median(enc_r_ms),'enc_o':np.median(enc_o_ms),'joi':np.median(joi_ms2)})
        print(f'[{fname}] CER={c*100:.1f}%  enc_rknn={np.median(enc_r_ms):.0f}ms  enc_onnx={np.median(enc_o_ms):.0f}ms  joi={np.median(joi_ms2):.2f}ms  RTF={total/audio_ms:.3f}')
        print(f'  HYP: {text}')

    enc_r.release(); dec_r.release(); joi_r.release()
    avg_cer_h = np.mean([s['cer'] for s in hybrid_stats])
    print(f'\n  AVG CER:       {avg_cer_h*100:.2f}%')
    print(f'  Enc-RKNN/chunk: {np.mean([s["enc_r"] for s in hybrid_stats]):.1f}ms')
    print(f'  Enc-ONNX/chunk: {np.mean([s["enc_o"] for s in hybrid_stats]):.1f}ms')
    print(f'  Joi/frame:      {np.mean([s["joi"] for s in hybrid_stats]):.3f}ms')
    print(f'  RTF (enc+joi):  {np.mean([s["rtf"] for s in hybrid_stats]):.3f}')
except ImportError:
    print('  rknnlite not available, skipping RKNN Hybrid')
