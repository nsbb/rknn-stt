"""
3.wav을 ONNX vs RKNN 청크별 비교 디버그
"""
import sys, numpy as np, soundfile as sf, os
import onnxruntime as ort

BASE = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
sys.path.insert(0, RKNN_DIR)
from fbank import KaldiFbank

ENC_SCHEMA = [
    ('x',[1,39,80],'float32'),('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]
CACHE_NAMES = [s[0] for s in ENC_SCHEMA if s[0] != 'x']
CHUNK=39; OFFSET=32; CONTEXT_SIZE=2; BLANK_ID=0; UNK_ID=2

def nchw2nhwc(a): return np.transpose(a, (0,2,3,1))

def load_vocab():
    vocab={}
    with open(f'{BASE}/tokens.txt') as f:
        for line in f:
            p = line.strip().split()
            if len(p)==2: vocab[int(p[1])] = p[0]
    return vocab

def decode(hyp, vocab):
    return ''.join(vocab.get(i,'') for i in hyp).replace('▁',' ').strip()

# Audio
audio, sr = sf.read(f'{BASE}/test_wavs/3.wav', dtype='float32')
fb = KaldiFbank()
feats = fb.compute_all(audio)
T = feats.shape[0]
pad = max(CHUNK, ((T//OFFSET)+1)*OFFSET) - T
feats_padded = np.vstack([feats, np.zeros((pad,80),dtype=np.float32)])

# Sessions
enc_s = ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
dec_s = ort.InferenceSession(f'{BASE}/decoder-epoch-99-avg-1.onnx', providers=['CPUExecutionProvider'])
joi_s = ort.InferenceSession(f'{BASE}/joiner-epoch-99-avg-1.onnx',  providers=['CPUExecutionProvider'])

from rknnlite.api import RKNNLite
enc_r = RKNNLite(verbose=False); enc_r.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn'); enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
dec_r = RKNNLite(verbose=False); dec_r.load_rknn(f'{RKNN_DIR}/decoder-epoch-99-avg-1.rknn'); dec_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
joi_r = RKNNLite(verbose=False); joi_r.load_rknn(f'{RKNN_DIR}/joiner-epoch-99-avg-1.rknn');  joi_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

vocab = load_vocab()

# State
def init_state():
    s={}
    for name,shape,dtype in ENC_SCHEMA:
        s[name]=np.zeros(shape,dtype=np.dtype(dtype))
    return s

state_o = init_state()
state_r = init_state()
hyp_o = [0]*CONTEXT_SIZE
hyp_r = [0]*CONTEXT_SIZE
dec_out_o = dec_s.run(None,{'y':np.array([hyp_o],dtype=np.int64)})[0]
dec_out_r = np.array(dec_r.inference(inputs=[np.array([hyp_r],dtype=np.int64)])[0])

num_proc = 0
chunk_idx = 0
print(f"Audio: 3.wav  frames={T}  chunks={(feats_padded.shape[0]-CHUNK)//OFFSET+1}")
print(f"GT: 주민등록증을 보여 주시겠어요?")

while num_proc + CHUNK <= feats_padded.shape[0]:
    x = feats_padded[num_proc:num_proc+CHUNK]
    state_o['x'] = x[np.newaxis]; state_r['x'] = x[np.newaxis]

    # ONNX encoder
    onnx_in = {nm: state_o[nm] for nm,_,_ in ENC_SCHEMA}
    onnx_out = enc_s.run(None, onnx_in)
    enc_o = np.array(onnx_out[0]).squeeze(0)
    for i, nm in enumerate(CACHE_NAMES): state_o[nm] = np.array(onnx_out[i+1])

    # RKNN encoder
    rknn_in = []
    for nm, shape, _ in ENC_SCHEMA:
        a = state_r[nm]
        if len(shape)==4: a = nchw2nhwc(a)
        rknn_in.append(a)
    rknn_out = enc_r.inference(inputs=rknn_in)
    enc_r_ = np.array(rknn_out[0]).squeeze(0)
    for i, nm in enumerate(CACHE_NAMES):
        a = np.array(rknn_out[i+1])
        if 'cached_len' in nm: a = a.astype(np.int64)
        state_r[nm] = a

    diff = np.abs(enc_o.astype(np.float32) - enc_r_.astype(np.float32)).max()
    print(f"\n[Chunk {chunk_idx}] frames {num_proc}-{num_proc+CHUNK-1}  enc_max_diff={diff:.4f}")

    # Per-frame greedy
    tokens_o, tokens_r = [], []
    for t in range(enc_o.shape[0]):
        cur_o = enc_o[t:t+1].astype(np.float32)
        cur_r = enc_r_[t:t+1].astype(np.float32)
        dec_o_f = dec_out_o.astype(np.float32).reshape(1,512)
        dec_r_f = dec_out_r.astype(np.float32).reshape(1,512)

        joi_o = joi_s.run(None, {'encoder_out': cur_o, 'decoder_out': dec_o_f})[0]
        joi_r_ = np.array(joi_r.inference(inputs=[cur_r, dec_r_f])[0])

        y_o = int(np.argmax(joi_o.squeeze()))
        y_r = int(np.argmax(joi_r_.squeeze()))
        tokens_o.append(y_o)
        tokens_r.append(y_r)

        if y_o not in (BLANK_ID, UNK_ID):
            hyp_o.append(y_o)
            dec_out_o = dec_s.run(None,{'y':np.array([hyp_o[-CONTEXT_SIZE:]],dtype=np.int64)})[0]
        if y_r not in (BLANK_ID, UNK_ID):
            hyp_r.append(y_r)
            dec_out_r = np.array(dec_r.inference(inputs=[np.array([hyp_r[-CONTEXT_SIZE:]],dtype=np.int64)])[0])

    tok_str = lambda ts: ' '.join([vocab.get(t,'<unk>') if t not in (0,2) else '_' for t in ts])
    print(f"  ONNX tokens: {tok_str(tokens_o)}")
    print(f"  RKNN tokens: {tok_str(tokens_r)}")

    num_proc += OFFSET; chunk_idx += 1

print(f"\nFinal ONNX: {decode(hyp_o[CONTEXT_SIZE:], vocab)}")
print(f"Final RKNN: {decode(hyp_r[CONTEXT_SIZE:], vocab)}")
enc_r.release(); dec_r.release(); joi_r.release()
