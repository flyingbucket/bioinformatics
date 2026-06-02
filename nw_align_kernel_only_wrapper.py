import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_O = -11
GAP_E = -1
MIN_INF = -999999

BLOSUM_MATRIX = np.array(BLOSUM62, dtype=np.int32)
CHAR_TO_IDX = np.full(256, -1, dtype=np.int32)
for idx, char in enumerate(BLOSUM62.alphabet):  # pyright: ignore
    CHAR_TO_IDX[ord(char)] = idx

lib_path = os.path.abspath("./lib/libnw_align.so")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found: {lib_path}. Compile first.")

_lib = ctypes.CDLL(lib_path)

_lib.nw_align_c_kernel.argtypes = [
    ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),  # arr1
    ctypes.c_int,  # l1
    ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),  # arr2
    ctypes.c_int,  # l2
    ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),  # matrix (BLOSUM)
    ctypes.c_int,  # matrix_cols
    ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # char_to_idx
    ctypes.c_int32,  # gap_o
    ctypes.c_int32,  # gap_e
    ctypes.c_int32,  # min_inf
    ndpointer(
        dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"
    ),  # M (由 Python 传入用于填表)
    ndpointer(
        dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"
    ),  # X (由 Python 传入用于填表)
    ndpointer(
        dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"
    ),  # Y (由 Python 传入用于填表)
]

_lib.nw_align_c_kernel.restype = ctypes.c_int32


def nw_align_c(
    seq1_str: str,
    seq2_str: str,
    blosum_matrix: np.ndarray = BLOSUM_MATRIX,
    char_to_idx: np.ndarray = CHAR_TO_IDX,
    gap_o: int = GAP_O,
    gap_e: int = GAP_E,
    min_inf: int = MIN_INF,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:

    arr1 = np.frombuffer(seq1_str.encode("ascii"), dtype=np.uint8)
    arr2 = np.frombuffer(seq2_str.encode("ascii"), dtype=np.uint8)

    l1, l2 = len(arr1), len(arr2)

    M = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)
    X = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)
    Y = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)

    matrix_cols = blosum_matrix.shape[1]

    best_score = _lib.nw_align_c_kernel(
        arr1,
        l1,
        arr2,
        l2,
        blosum_matrix,
        matrix_cols,
        char_to_idx,
        gap_o,
        gap_e,
        min_inf,
        M,
        X,
        Y,
    )

    return int(best_score), M, X, Y
