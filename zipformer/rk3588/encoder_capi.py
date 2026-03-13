"""
C API encoder wrapper for rmreshape model.
Uses set_io_mem for zero-copy, handles NCHW↔NHWC conversion.
Target: 30ms/chunk (vs rknnlite 49ms).
"""
import ctypes, numpy as np
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_float, c_char_p, c_void_p, POINTER, Structure, byref)

lib = ctypes.CDLL('/usr/lib/librknnrt.so')

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
NATIVE_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 8: 8}

class RknnInputOutputNum(Structure):
    _fields_ = [('n_input', c_uint32), ('n_output', c_uint32)]

class RknnTensorAttr(Structure):
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

class RknnTensorMem(Structure):
    _fields_ = [
        ('virt_addr', c_void_p), ('phys_addr', c_uint64),
        ('fd', c_int32), ('offset', c_int32),
        ('size', c_uint32), ('flags', c_uint32), ('priv_data', c_void_p),
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
lib.rknn_create_mem2.restype = POINTER(RknnTensorMem)
lib.rknn_create_mem2.argtypes = [c_uint64, c_uint64, c_uint64]
lib.rknn_destroy_mem.restype = c_int
lib.rknn_destroy_mem.argtypes = [c_uint64, POINTER(RknnTensorMem)]
lib.rknn_set_io_mem.restype = c_int
lib.rknn_set_io_mem.argtypes = [c_uint64, POINTER(RknnTensorMem), POINTER(RknnTensorAttr)]
lib.rknn_mem_sync.restype = c_int
lib.rknn_mem_sync.argtypes = [c_uint64, POINTER(RknnTensorMem), c_int]

SYNC_TO_DEVICE = 0x1
SYNC_FROM_DEVICE = 0x2


def out_nchw_to_in_nhwc(out_arr, in_nhwc_shape):
    """Convert output (NCHW fmt=0) to input (NHWC fmt=1) format."""
    N, H, W, C = in_nhwc_shape
    return np.ascontiguousarray(np.transpose(out_arr.reshape(N, C, H, W), (0, 2, 3, 1)))


class EncoderCAPI:
    """C API encoder with set_io_mem for rmreshape model."""

    def __init__(self, model_path, core_mask=1):
        with open(model_path, 'rb') as f:
            buf = f.read()
        self._md = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
        self._ctx = c_uint64(0)
        ret = lib.rknn_init(byref(self._ctx), self._md, len(buf), 0, None)
        assert ret == 0, f"rknn_init failed: {ret}"
        lib.rknn_set_core_mask(self._ctx, core_mask)

        io = RknnInputOutputNum()
        lib.rknn_query(self._ctx, 0, byref(io), ctypes.sizeof(io))
        self.n_in = io.n_input
        self.n_out = io.n_output

        # Setup input IO
        self._in_attrs = []
        self._in_mems = []
        self._in_shapes = []  # NHWC shapes for data writing
        self._in_dtypes = []
        for i in range(self.n_in):
            attr = RknnTensorAttr()
            attr.index = i
            lib.rknn_query(self._ctx, 1, byref(attr), ctypes.sizeof(attr))
            nt = attr.type
            nb = NATIVE_BYTES.get(nt, 2)
            if nt == 8:
                attr.type = 8; db = 8
            else:
                attr.type = 0; db = 4
            attr.pass_through = 0
            attr.fmt = 1 if attr.n_dims == 4 else 0  # NHWC for 4D
            dims = tuple(attr.dims[j] for j in range(attr.n_dims))
            self._in_shapes.append(dims)
            self._in_dtypes.append(np.int64 if nt == 8 else np.float32)
            nsz = attr.size_with_stride if attr.size_with_stride else attr.size
            mem = lib.rknn_create_mem2(self._ctx, max(nsz * db // nb, 64), 0)
            self._in_mems.append(mem)
            self._in_attrs.append(attr)
            lib.rknn_set_io_mem(self._ctx, mem, byref(attr))

        # Setup output IO
        self._out_attrs = []
        self._out_mems = []
        self._out_shapes = []  # NCHW shapes
        self._out_dtypes = []
        for i in range(self.n_out):
            attr = RknnTensorAttr()
            attr.index = i
            lib.rknn_query(self._ctx, 2, byref(attr), ctypes.sizeof(attr))
            nt = attr.type
            if nt == 8:
                attr.type = 8; db = 8
            else:
                attr.type = 0; db = 4
            attr.pass_through = 0
            dims = tuple(attr.dims[j] for j in range(attr.n_dims))
            self._out_shapes.append(dims)
            self._out_dtypes.append(np.int64 if nt == 8 else np.float32)
            mem = lib.rknn_create_mem2(self._ctx, max(attr.n_elems * db, 64), 0)
            self._out_mems.append(mem)
            self._out_attrs.append(attr)
            lib.rknn_set_io_mem(self._ctx, mem, byref(attr))

        # Cache names (skip x at index 0)
        self._cache_names = []
        for i in range(1, self.n_in):
            self._cache_names.append(self._in_attrs[i].name.decode().strip('\x00'))

    def _write_input(self, idx, data):
        """Write numpy array to input buffer."""
        mem = self._in_mems[idx]
        src = np.ascontiguousarray(data)
        nbytes = src.nbytes
        ctypes.memmove(mem.contents.virt_addr, src.ctypes.data, nbytes)
        lib.rknn_mem_sync(self._ctx, mem, SYNC_TO_DEVICE)

    def _read_output(self, idx):
        """Read output buffer as numpy array (with sync)."""
        lib.rknn_mem_sync(self._ctx, self._out_mems[idx], SYNC_FROM_DEVICE)
        return self._read_output_nosync(idx)

    def _read_output_nosync(self, idx):
        """Read output buffer as numpy array (no sync — caller must sync first)."""
        mem = self._out_mems[idx]
        shape = self._out_shapes[idx]
        dtype = self._out_dtypes[idx]
        n_elems = 1
        for d in shape:
            n_elems *= d
        buf = (ctypes.c_byte * (n_elems * np.dtype(dtype).itemsize)).from_address(mem.contents.virt_addr)
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    def run(self, x_nhwc, cache_nhwc):
        """
        Run encoder.
        x_nhwc: (1, 39, 80, 1) float32
        cache_nhwc: dict of cache tensors in NHWC format
        Returns: (encoder_out, new_cache_nhwc)
        """
        # Write x
        self._write_input(0, x_nhwc)
        # Write cache
        for i, nm in enumerate(self._cache_names):
            self._write_input(i + 1, cache_nhwc[nm])

        # Run
        lib.rknn_run(self._ctx, None)

        # Batch sync all outputs from device first (saves ~0.4ms vs per-output sync)
        for i in range(self.n_out):
            lib.rknn_mem_sync(self._ctx, self._out_mems[i], SYNC_FROM_DEVICE)

        # Read encoder_out (NCHW → reshape to [1, 8, 512])
        enc_nchw = self._read_output_nosync(0)
        enc_out = out_nchw_to_in_nhwc(enc_nchw, (1, 8, 512, 1)).reshape(1, 8, 512)

        # Read and convert cache outputs (no sync needed — already done above)
        new_cache = {}
        for i, nm in enumerate(self._cache_names):
            out_arr = self._read_output_nosync(i + 1)
            in_shape = self._in_shapes[i + 1]
            converted = out_nchw_to_in_nhwc(out_arr, in_shape)
            if 'cached_len' in nm:
                converted = converted.astype(np.int64)
            new_cache[nm] = converted

        return enc_out, new_cache

    def init_cache(self):
        """Create zero-initialized cache dict."""
        cache = {}
        for i, nm in enumerate(self._cache_names):
            shape = self._in_shapes[i + 1]
            dtype = self._in_dtypes[i + 1]
            cache[nm] = np.zeros(shape, dtype=dtype)
        return cache

    def release(self):
        for m in self._in_mems + self._out_mems:
            lib.rknn_destroy_mem(self._ctx, m)
        lib.rknn_destroy(self._ctx)
