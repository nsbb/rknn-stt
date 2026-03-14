"""캐시별 diff 추적 — chunk 0,1,2"""
import sys, numpy as np, soundfile as sf
import onnxruntime as ort
sys.path.insert(0, '/home/rk3588/travail/rk3588/rknn-stt/zipformer/rk3588')
from fbank import KaldiFbank
from rknnlite.api import RKNNLite

BASE='/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR=f'{BASE}/rk3588'
ENC_SCHEMA=[
    ('x',[1,39,80],'float32'),('cached_len_0',[2,1],'int64'),('cached_len_1',[4,1],'int64'),('cached_len_2',[3,1],'int64'),('cached_len_3',[2,1],'int64'),('cached_len_4',[4,1],'int64'),
    ('cached_avg_0',[2,1,384],'float32'),('cached_avg_1',[4,1,384],'float32'),('cached_avg_2',[3,1,384],'float32'),('cached_avg_3',[2,1,384],'float32'),('cached_avg_4',[4,1,384],'float32'),
    ('cached_key_0',[2,64,1,192],'float32'),('cached_key_1',[4,32,1,192],'float32'),('cached_key_2',[3,16,1,192],'float32'),('cached_key_3',[2,8,1,192],'float32'),('cached_key_4',[4,32,1,192],'float32'),
    ('cached_val_0',[2,64,1,96],'float32'),('cached_val_1',[4,32,1,96],'float32'),('cached_val_2',[3,16,1,96],'float32'),('cached_val_3',[2,8,1,96],'float32'),('cached_val_4',[4,32,1,96],'float32'),
    ('cached_val2_0',[2,64,1,96],'float32'),('cached_val2_1',[4,32,1,96],'float32'),('cached_val2_2',[3,16,1,96],'float32'),('cached_val2_3',[2,8,1,96],'float32'),('cached_val2_4',[4,32,1,96],'float32'),
    ('cached_conv1_0',[2,1,384,30],'float32'),('cached_conv1_1',[4,1,384,30],'float32'),('cached_conv1_2',[3,1,384,30],'float32'),('cached_conv1_3',[2,1,384,30],'float32'),('cached_conv1_4',[4,1,384,30],'float32'),
    ('cached_conv2_0',[2,1,384,30],'float32'),('cached_conv2_1',[4,1,384,30],'float32'),('cached_conv2_2',[3,1,384,30],'float32'),('cached_conv2_3',[2,1,384,30],'float32'),('cached_conv2_4',[4,1,384,30],'float32'),
]
CACHE_NAMES=[s[0] for s in ENC_SCHEMA if s[0] != 'x']

def nchw2nhwc(a): return np.transpose(a,(0,2,3,1))

audio,_=sf.read(f'{BASE}/test_wavs/3.wav',dtype='float32')
fb=KaldiFbank(); feats=fb.compute_all(audio)
T=feats.shape[0]; pad=max(39,((T//32)+1)*32)-T
feats=np.vstack([feats,np.zeros((pad,80),dtype=np.float32)])

sess=ort.InferenceSession(f'{BASE}/encoder-epoch-99-avg-1.onnx',providers=['CPUExecutionProvider'])
enc_r=RKNNLite(verbose=False)
enc_r.load_rknn(f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn')
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

so={nm:np.zeros(shape,dtype=np.dtype(dtype)) for nm,shape,dtype in ENC_SCHEMA}
sr={nm:np.zeros(shape,dtype=np.dtype(dtype)) for nm,shape,dtype in ENC_SCHEMA}

for ci in range(4):
    x=feats[ci*32:ci*32+39][np.newaxis]
    so['x']=x; sr['x']=x

    # ONNX
    onnx_in={nm:so[nm] for nm,_,_ in ENC_SCHEMA}
    out_o=sess.run(None,onnx_in)
    enc_o=np.array(out_o[0])
    for i,nm in enumerate(CACHE_NAMES):
        so[nm]=np.array(out_o[i+1])

    # RKNN
    rknn_in=[]
    for nm,shape,_ in ENC_SCHEMA:
        a=sr[nm]
        if len(shape)==4: a=nchw2nhwc(a)
        rknn_in.append(a)
    out_r=enc_r.inference(inputs=rknn_in)
    enc_rr=np.array(out_r[0])
    for i,nm in enumerate(CACHE_NAMES):
        a=np.array(out_r[i+1])
        if 'cached_len' in nm: a=a.astype(np.int64)
        sr[nm]=a

    enc_diff=np.abs(enc_o.astype(np.float32)-enc_rr.astype(np.float32)).max()
    print(f'\n=== Chunk {ci}: encoder_out max_diff={enc_diff:.4f} ===')

    # 캐시별 ONNX vs RKNN diff (RKNN state → ONNX shape로 비교)
    diffs={}
    for nm in CACHE_NAMES:
        o=so[nm].astype(np.float32)
        r=sr[nm].astype(np.float32)
        if o.shape==r.shape:
            diffs[nm]=(np.abs(o-r).max(), np.abs(o).max())
        else:
            diffs[nm]=(float('nan'), np.abs(o).max())

    # 가장 큰 diff 상위 10개
    sorted_d=sorted(diffs.items(),key=lambda x:x[1][0] if not np.isnan(x[1][0]) else 0,reverse=True)
    for nm,(d,mx) in sorted_d[:10]:
        print(f'  {nm:30s}: diff={d:.4f}  max_val={mx:.2f}')

enc_r.release()
