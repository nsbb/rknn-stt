"""
RKNN Zero-copy 벤치마크
ctypes로 C API 직접 호출 vs rknnlite.api inference() 비교

핵심 가설: rknnlite.api inference()의 66ms 오버헤드가
          텐서당 ioctl 또는 입력 처리 비용이라면
          C API + 사전 할당 DMA 버퍼로 극적 감소 가능
"""
import ctypes, ctypes.util, numpy as np, time, sys, os
from ctypes import (c_int, c_int8, c_int32, c_int64, c_uint8, c_uint32,
                    c_uint64, c_float, c_void_p, c_size_t, POINTER, Structure, byref)

# ─── librknnrt.so 로드 ─────────────────────────────────────────────
LIBRKNN_PATH = '/usr/lib/librknnrt.so'
lib = ctypes.CDLL(LIBRKNN_PATH)
print(f"[ctypes] Loaded {LIBRKNN_PATH}")

# ─── 상수 ─────────────────────────────────────────────────────────
RKNN_MAX_DIMS        = 16
RKNN_MAX_NAME_LEN    = 256
RKNN_SUCC            = 0

# rknn_query_cmd
RKNN_QUERY_IN_OUT_NUM    = 0
RKNN_QUERY_INPUT_ATTR    = 1
RKNN_QUERY_OUTPUT_ATTR   = 2
RKNN_QUERY_SDK_VERSION   = 5

# rknn_tensor_format
RKNN_TENSOR_NCHW  = 0
RKNN_TENSOR_NHWC  = 1

# rknn_tensor_type
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8    = 2
RKNN_TENSOR_INT64   = 8

# rknn_core_mask
RKNN_NPU_CORE_0 = 1

# rknn_mem_alloc_flags
RKNN_FLAG_MEMORY_NON_CACHEABLE = 1 << 1   # No cache sync needed

# rknn_mem_sync_mode
RKNN_MEMORY_SYNC_TO_DEVICE   = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2


# ─── 구조체 정의 ───────────────────────────────────────────────────
class RknnInputOutputNum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]


class RknnTensorAttr(Structure):
    _fields_ = [
        ('index',           c_uint32),
        ('n_dims',          c_uint32),
        ('dims',            c_uint32 * RKNN_MAX_DIMS),
        ('name',            ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_elems',         c_uint32),
        ('size',            c_uint32),
        ('fmt',             c_int),
        ('type',            c_int),
        ('qnt_type',        c_int),
        ('fl',              c_int8),
        # 3 bytes padding before int32 (ctypes handles automatically)
        ('zp',              c_int32),
        ('scale',           c_float),
        ('w_stride',        c_uint32),
        ('size_with_stride', c_uint32),
        ('pass_through',    c_uint8),
        # 3 bytes padding before uint32 (ctypes handles automatically)
        ('h_stride',        c_uint32),
    ]


class RknnTensorMem(Structure):
    _fields_ = [
        ('virt_addr',  c_void_p),
        ('phys_addr',  c_uint64),
        ('fd',         c_int32),
        ('offset',     c_int32),
        ('size',       c_uint32),
        ('flags',      c_uint32),
        ('priv_data',  c_void_p),
    ]


# ─── 함수 시그니처 설정 ─────────────────────────────────────────────
lib.rknn_init.restype  = c_int
lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]

lib.rknn_destroy.restype  = c_int
lib.rknn_destroy.argtypes = [c_uint64]

lib.rknn_query.restype  = c_int
lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]

lib.rknn_set_core_mask.restype  = c_int
lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]

lib.rknn_run.restype  = c_int
lib.rknn_run.argtypes = [c_uint64, c_void_p]

lib.rknn_create_mem2.restype  = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]

lib.rknn_destroy_mem.restype  = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]

lib.rknn_set_io_mem.restype  = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]

lib.rknn_mem_sync.restype  = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(RknnTensorMem), c_int]


def check(ret, msg=''):
    if ret != RKNN_SUCC:
        raise RuntimeError(f"RKNN error {ret}: {msg}")


# ─── ZeroCopyEncoder ────────────────────────────────────────────────
class ZeroCopyEncoder:
    """
    RKNN C API 직접 호출 기반 Encoder.
    Non-cacheable DMA 버퍼를 사전 할당하고 rknn_run만 호출.
    """
    def __init__(self, rknn_path, core_mask=RKNN_NPU_CORE_0):
        # 1. 모델 로드
        with open(rknn_path, 'rb') as f:
            model_buf = f.read()
        model_data = ctypes.create_string_buffer(model_buf)

        self.ctx = c_uint64(0)
        ret = lib.rknn_init(byref(self.ctx), model_data, len(model_buf), 0, None)
        check(ret, 'rknn_init')
        print(f"  [ZeroCopy] ctx={self.ctx.value:#x}")

        # 2. NPU 코어 설정
        ret = lib.rknn_set_core_mask(self.ctx, core_mask)
        check(ret, 'rknn_set_core_mask')

        # 3. 텐서 수 조회
        io_num = RknnInputOutputNum()
        check(lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                             byref(io_num), ctypes.sizeof(io_num)), 'query io_num')
        self.n_inputs  = io_num.n_input
        self.n_outputs = io_num.n_output
        print(f"  [ZeroCopy] n_inputs={self.n_inputs}, n_outputs={self.n_outputs}")

        # 4. 입력 텐서 속성 조회
        self.input_attrs = (RknnTensorAttr * self.n_inputs)()
        for i in range(self.n_inputs):
            self.input_attrs[i].index = i
            check(lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR,
                                 byref(self.input_attrs[i]),
                                 ctypes.sizeof(RknnTensorAttr)), f'query input {i}')
            # pass_through=1: 변환 없이 raw bytes 직접 전달
            self.input_attrs[i].pass_through = 1

        # 5. 출력 텐서 속성 조회
        self.output_attrs = (RknnTensorAttr * self.n_outputs)()
        for i in range(self.n_outputs):
            self.output_attrs[i].index = i
            check(lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                                 byref(self.output_attrs[i]),
                                 ctypes.sizeof(RknnTensorAttr)), f'query output {i}')
            # float32로 출력받기
            self.output_attrs[i].want_float = 1  # 없는 필드 — pass_through 대신 type 설정
            self.output_attrs[i].type = RKNN_TENSOR_FLOAT32

        # native 타입별 바이트 수
        _NATIVE_BYTES = {
            RKNN_TENSOR_FLOAT32: 4,
            RKNN_TENSOR_FLOAT16: 2,
            RKNN_TENSOR_INT8:    1,
            RKNN_TENSOR_INT64:   8,
        }

        # 6. Non-cacheable DMA 버퍼 할당
        # pass_through=0, type=FLOAT32/INT64로 설정
        # 버퍼 크기: native_size_with_stride * (desired_bytes / native_bytes)
        # → stride 패딩 고려
        self.input_mems  = []
        self.output_mems = []

        for i in range(self.n_inputs):
            attr = self.input_attrs[i]
            native_type = attr.type
            native_bytes = _NATIVE_BYTES.get(native_type, 2)
            if native_type == RKNN_TENSOR_INT64:
                desired_bytes = 8
                attr.type = RKNN_TENSOR_INT64
            else:
                desired_bytes = 4
                attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            n_dims = attr.n_dims
            attr.fmt = RKNN_TENSOR_NHWC if n_dims == 4 else RKNN_TENSOR_NCHW
            # stride 패딩을 반영한 크기: size_with_stride는 native 단위
            native_szs = attr.size_with_stride if attr.size_with_stride else attr.size
            sz = native_szs * desired_bytes // native_bytes
            mem = lib.rknn_create_mem2(self.ctx, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
            if not mem:
                raise RuntimeError(f"rknn_create_mem2 failed for input {i}")
            self.input_mems.append(mem)
            ret = lib.rknn_set_io_mem(self.ctx, mem, byref(attr))
            if ret != RKNN_SUCC:
                nm = attr.name.decode()
                raise RuntimeError(f"rknn_set_io_mem input {i} ({nm}) failed: {ret}, sz={sz}")

        for i in range(self.n_outputs):
            attr = self.output_attrs[i]
            native_type = attr.type
            native_bytes = _NATIVE_BYTES.get(native_type, 2)
            if native_type == RKNN_TENSOR_INT64:
                desired_bytes = 8
                attr.type = RKNN_TENSOR_INT64
            else:
                desired_bytes = 4
                attr.type = RKNN_TENSOR_FLOAT32
            attr.pass_through = 0
            native_szs = attr.size_with_stride if attr.size_with_stride else attr.size
            sz = native_szs * desired_bytes // native_bytes
            mem = lib.rknn_create_mem2(self.ctx, sz, RKNN_FLAG_MEMORY_NON_CACHEABLE)
            if not mem:
                raise RuntimeError(f"rknn_create_mem2 failed for output {i}")
            self.output_mems.append(mem)
            ret = lib.rknn_set_io_mem(self.ctx, mem, byref(attr))
            if ret != RKNN_SUCC:
                nm = attr.name.decode()
                raise RuntimeError(f"rknn_set_io_mem output {i} ({nm}) failed: {ret}")

        print(f"  [ZeroCopy] DMA buffers allocated ({self.n_inputs}in, {self.n_outputs}out)")

    def _copy_to_input(self, idx, arr):
        """numpy array를 DMA 입력 버퍼로 복사"""
        mem  = self.input_mems[idx]
        size = arr.nbytes
        ctypes.memmove(mem.contents.virt_addr, arr.ctypes.data, size)

    def _copy_from_output(self, idx, n_elems):
        """DMA 출력 버퍼에서 numpy array로 복사"""
        mem  = self.output_mems[idx]
        attr = self.output_attrs[idx]
        if attr.type == RKNN_TENSOR_INT64:
            arr = np.empty(n_elems, dtype=np.int64)
            ctypes.memmove(arr.ctypes.data, mem.contents.virt_addr, n_elems * 8)
        else:
            arr = np.empty(n_elems, dtype=np.float32)
            ctypes.memmove(arr.ctypes.data, mem.contents.virt_addr, n_elems * 4)
        return arr

    def run(self, inputs_list):
        """
        inputs_list: NHWC numpy 배열 리스트 (inference_rknn.py의 pack_rknn_inputs 출력과 동일)
        returns: list of numpy arrays (rknnlite.api inference()와 동일 인터페이스)
        """
        # 입력 복사
        for i, arr in enumerate(inputs_list):
            # pass_through=1 이므로 contiguous flat bytes로 전달
            flat = np.ascontiguousarray(arr)
            self._copy_to_input(i, flat)

        # 추론
        ret = lib.rknn_run(self.ctx, None)
        check(ret, 'rknn_run')

        # 출력 읽기
        outputs = []
        for i in range(self.n_outputs):
            n = self.output_attrs[i].n_elems
            arr = self._copy_from_output(i, n)
            # 출력 shape 복원
            dims = list(self.output_attrs[i].dims[:self.output_attrs[i].n_dims])
            arr = arr.reshape(dims)
            outputs.append(arr)
        return outputs

    def release(self):
        for mem in self.input_mems:
            lib.rknn_destroy_mem(self.ctx, mem)
        for mem in self.output_mems:
            lib.rknn_destroy_mem(self.ctx, mem)
        lib.rknn_destroy(self.ctx)
        print("  [ZeroCopy] Released")


# ─── 메인 벤치마크 ────────────────────────────────────────────────────
BASE     = '/home/rk3588/travail/rk3588/rknn-stt/zipformer'
RKNN_DIR = f'{BASE}/rk3588'
MODEL    = f'{RKNN_DIR}/encoder-epoch-99-avg-1.rknn'

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


def nchw2nhwc(a): return np.transpose(a, (0, 2, 3, 1))


def make_inputs():
    """랜덤 입력 생성 (NHWC 변환 포함)"""
    inp = []
    for nm, sh, dt in ENC_SCHEMA:
        if dt == 'int64':
            a = np.zeros(sh, dtype=np.int64)
        else:
            a = (np.random.randn(*sh) * 0.1).astype(np.float32)
        if len(sh) == 4:
            a = nchw2nhwc(a)
        inp.append(np.ascontiguousarray(a))
    return inp


N_WARM  = 3
N_BENCH = 20

# ─── Test A: Zero-copy (ctypes C API) ───────────────────────────────
print("\n=== Test A: Zero-copy (ctypes C API) ===")
try:
    zc = ZeroCopyEncoder(MODEL)

    inputs = make_inputs()  # 새 입력 매 반복 생성 (현실적)

    # Warmup
    for _ in range(N_WARM):
        inputs = make_inputs()
        zc.run(inputs)

    times_zc = []
    for _ in range(N_BENCH):
        inputs = make_inputs()
        t0 = time.perf_counter()
        zc.run(inputs)
        times_zc.append((time.perf_counter() - t0) * 1000)

    print(f"  median={np.median(times_zc):.2f}ms  min={min(times_zc):.2f}ms  max={max(times_zc):.2f}ms")

    # ─── run 부분만 측정 (입력 복사 제외) ──────────────────────────
    print("\n=== Test A2: rknn_run만 (입력 복사 제외) ===")
    inputs = make_inputs()
    for i, arr in enumerate(inputs):
        flat = np.ascontiguousarray(arr)
        zc._copy_to_input(i, flat)

    for _ in range(N_WARM):
        lib.rknn_run(zc.ctx, None)

    times_run = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        lib.rknn_run(zc.ctx, None)
        times_run.append((time.perf_counter() - t0) * 1000)

    print(f"  median={np.median(times_run):.2f}ms  min={min(times_run):.2f}ms")

    # ─── 입력 복사만 측정 ─────────────────────────────────────────────
    print("\n=== Test A3: 입력 복사만 (rknn_run 제외) ===")
    times_copy = []
    for _ in range(N_BENCH):
        inputs = make_inputs()
        t0 = time.perf_counter()
        for i, arr in enumerate(inputs):
            flat = np.ascontiguousarray(arr)
            zc._copy_to_input(i, flat)
        times_copy.append((time.perf_counter() - t0) * 1000)

    print(f"  median={np.median(times_copy):.2f}ms  min={min(times_copy):.2f}ms")

    zc.release()

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
    times_zc = None

# ─── Test B: rknnlite.api inference() (기준) ────────────────────────
print("\n=== Test B: rknnlite.api inference() (기준) ===")
from rknnlite.api import RKNNLite

enc_r = RKNNLite(verbose=False)
enc_r.load_rknn(MODEL)
enc_r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

for _ in range(N_WARM):
    enc_r.inference(inputs=make_inputs())

times_lite = []
for _ in range(N_BENCH):
    inp = make_inputs()
    t0 = time.perf_counter()
    enc_r.inference(inputs=inp)
    times_lite.append((time.perf_counter() - t0) * 1000)

print(f"  median={np.median(times_lite):.2f}ms  min={min(times_lite):.2f}ms  max={max(times_lite):.2f}ms")
enc_r.release()

# ─── 요약 ──────────────────────────────────────────────────────────
print("\n=== 요약 ===")
if times_zc:
    print(f"A  (zero-copy full):   {np.median(times_zc):.2f}ms")
    print(f"A2 (rknn_run only):    {np.median(times_run):.2f}ms")
    print(f"A3 (copy only):        {np.median(times_copy):.2f}ms")
    print(f"B  (rknnlite.api):     {np.median(times_lite):.2f}ms")
    speedup = np.median(times_lite) / np.median(times_zc)
    print(f"Speedup A vs B: {speedup:.1f}x")
else:
    print(f"B  (rknnlite.api): {np.median(times_lite):.2f}ms")
    print("Zero-copy 실패")
