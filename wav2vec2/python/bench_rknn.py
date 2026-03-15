"""Benchmark wav2vec2 RKNN models using C API (set_io_mem)."""
import numpy as np, time, ctypes, os, sys, json
import soundfile as sf, scipy.signal

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                     c_float, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
NB = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}

class IONum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]

class Attr(Structure):
    _fields_ = [
        ('index', c_uint32), ('n_dims', c_uint32),
        ('dims', c_uint32 * RKNN_MAX_DIMS),
        ('name', ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_elems', c_uint32), ('size', c_uint32),
        ('fmt', c_int), ('type', c_int), ('qnt_type', c_int),
        ('fl', c_int8), ('zp', c_int32), ('scale', c_float),
        ('w_stride', c_uint32), ('size_with_stride', c_uint32),
        ('pass_through', c_uint8), ('h_stride', c_uint32),
    ]

class Mem(Structure):
    _fields_ = [
        ('virt_addr', c_void_p), ('phys_addr', c_uint64),
        ('fd', c_int32), ('offset', c_int32), ('size', c_uint32),
        ('flags', c_uint32), ('priv_data', c_void_p),
    ]

class MemSize(Structure):
    _fields_ = [
        ('total_weight_size', c_uint32), ('total_internal_size', c_uint32),
        ('total_dma_alloc_size', c_uint64), ('total_sram_size', c_uint32),
        ('free_sram_size', c_uint32), ('reserved', c_uint32 * 10),
    ]

lib.rknn_init.restype = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
lib.rknn_destroy.restype = c_int
lib.rknn_destroy.argtypes = [c_uint64]
lib.rknn_query.restype = c_int
lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
lib.rknn_set_core_mask.restype = c_int
lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]
lib.rknn_run.restype = c_int
lib.rknn_run.argtypes = [c_uint64, c_void_p]
lib.rknn_create_mem2.restype = POINTER(Mem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(Mem)]
lib.rknn_set_io_mem.restype = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(Mem), POINTER(Attr)]
lib.rknn_mem_sync.restype = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(Mem), c_int]

BASE = '/home/rk3588/travail/rk3588/rknn-stt/wav2vec2'


def load_audio(path, max_len=80000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
    audio = audio.astype(np.float32)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    return audio[np.newaxis]


def bench_model(path, label, audio_data=None, n_runs=20, warmup=5):
    if not os.path.exists(path):
        print(f'{label:45s} NOT FOUND')
        return None

    with open(path, 'rb') as f:
        buf = f.read()
    md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)

    ctx = c_uint64(0)
    ret = lib.rknn_init(byref(ctx), md, len(buf), 0, None)
    if ret != 0:
        print(f'{label:45s} INIT FAILED: {ret}')
        return None

    lib.rknn_set_core_mask(ctx, 1)  # core0 only

    io = IONum()
    lib.rknn_query(ctx, 0, byref(io), ctypes.sizeof(io))
    ms = MemSize()
    lib.rknn_query(ctx, 6, byref(ms), ctypes.sizeof(ms))

    # Setup input memory
    ims, ias = [], []
    for i in range(io.n_input):
        a = Attr()
        a.index = i
        lib.rknn_query(ctx, 1, byref(a), ctypes.sizeof(a))
        nt = a.type
        if nt == 8:
            a.type = 8
            db = 8
        else:
            a.type = 0
            db = 4
        a.pass_through = 0
        a.fmt = 0
        nb = NB.get(nt, 2)
        nsz = a.size_with_stride if a.size_with_stride else a.size
        sz = nsz * db // nb
        m = lib.rknn_create_mem2(ctx, max(sz, 64), 0)
        ims.append(m)
        ias.append(a)
        lib.rknn_set_io_mem(ctx, m, byref(a))

    # Setup output memory
    oms, oas = [], []
    for i in range(io.n_output):
        a = Attr()
        a.index = i
        lib.rknn_query(ctx, 2, byref(a), ctypes.sizeof(a))
        if a.type == 8:
            a.type = 8
            db = 8
        else:
            a.type = 0
            db = 4
        a.pass_through = 0
        m = lib.rknn_create_mem2(ctx, max(a.n_elems * db, 64), 0)
        oms.append(m)
        oas.append(a)
        lib.rknn_set_io_mem(ctx, m, byref(a))

    # Copy input data
    if audio_data is not None:
        data_bytes = audio_data.astype(np.float32).tobytes()
        ctypes.memmove(ims[0].contents.virt_addr, data_bytes, len(data_bytes))
        lib.rknn_mem_sync(ctx, ims[0], 0x1)

    # Warmup
    for _ in range(warmup):
        lib.rknn_run(ctx, None)

    # Benchmark
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lib.rknn_run(ctx, None)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000)

    # Read output for verification
    out_data = None
    if oas:
        a = oas[0]
        lib.rknn_mem_sync(ctx, oms[0], 0x2)
        out_buf = (ctypes.c_float * a.n_elems).from_address(oms[0].contents.virt_addr)
        dims = [a.dims[d] for d in range(a.n_dims)]
        out_data = np.ctypeslib.as_array(out_buf).reshape(dims).copy()

    w_mb = ms.total_weight_size / 1024 / 1024
    for m in ims + oms:
        lib.rknn_destroy_mem(ctx, m)
    lib.rknn_destroy(ctx)

    print(f'{label:45s} med={np.median(ts):7.1f}ms  min={np.min(ts):7.1f}ms  w={w_mb:.0f}MB')
    return out_data


def decode_output(output, vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    id2tok = {int(v): k for k, v in vocab.items()}

    ids = np.argmax(output, axis=-1)[0]
    prev = -1
    chars = []
    for i in ids:
        if i != prev:
            tok = id2tok.get(i, '')
            if tok == '|':
                chars.append(' ')
            elif tok not in ['', '\b', '\n', '\r', '\x10',
                             '<pad>', '<s>', '</s>', '<unk>'] and i >= 5:
                chars.append(tok)
        prev = i
    return ''.join(chars)


if __name__ == '__main__':
    audio = load_audio(f'{BASE}/input/call_elevator.wav')
    print(f"Audio shape: {audio.shape}\n")
    print("=== wav2vec2 RKNN Benchmark (C API set_io_mem, core0) ===\n")

    models = [
        (f'{BASE}/model/wav2vec-xls-r-300m_5s_fp16.rknn', 'FP16'),
        (f'{BASE}/model/wav2vec-xls-r-300m_5s_int8.rknn', 'INT8 (original)'),
        (f'{BASE}/model/wav2vec-xls-r-300m_5s_int8_v2.rknn', 'INT8 v2 (recalib)'),
    ]

    vocab_path = f'{BASE}/json/vocab.json'
    for path, label in models:
        out = bench_model(path, label, audio)
        if out is not None:
            text = decode_output(out, vocab_path)
            print(f'{"":45s} → [{text}]\n')
