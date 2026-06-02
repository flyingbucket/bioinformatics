import os
import ctypes
import numpy as np

from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_O = -11
GAP_E = -1
MIN_INF = -999999

BLOSUM_MATRIX = np.array(BLOSUM62, dtype=np.int32)
MATRIX_COLS = BLOSUM_MATRIX.shape[1]
CHAR_TO_IDX = np.full(256, -1, dtype=np.int32)
for idx, char in enumerate(BLOSUM62.alphabet):  # pyright: ignore
    CHAR_TO_IDX[ord(char)] = idx

MAX_PROTEIN_LEN = 320
ID_LEN = 6


class Result(ctypes.Structure):
    _fields_ = [
        ("id1", ctypes.c_char * 7),
        ("id2", ctypes.c_char * 7),
        ("score", ctypes.c_int32),
        ("al1", ctypes.c_uint8 * MAX_PROTEIN_LEN),
        ("al2", ctypes.c_uint8 * MAX_PROTEIN_LEN),
        ("identity_len1", ctypes.c_double),
        ("identity_len2", ctypes.c_double),
    ]


lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "lib", "libnw_align.so")
)
if not os.path.exists(lib_path):
    lib_path = os.path.abspath("./lib/libnw_align.so")

lib = ctypes.CDLL(lib_path)

lib.nw_align_c_full.argtypes = [
    ctypes.c_int,  # num_records
    ctypes.c_void_p,  # full_seq (uint8_t*)
    ctypes.c_void_p,  # seq_len (int32_t*)
    ctypes.c_void_p,  # seq_offset (int32_t*)
    ctypes.c_char_p,  # full_id (char*)
    ctypes.c_void_p,  # matrix (int32_t*)
    ctypes.c_int,  # matrix_cols
    ctypes.c_void_p,  # char_to_idx (int32_t*)
    ctypes.c_int32,  # gap_o
    ctypes.c_int32,  # gap_e
    ctypes.c_int32,  # min_inf
]
lib.nw_align_c_full.restype = ctypes.POINTER(Result)


def nw_align_c_all_records(processed_records):
    num_records = len(processed_records)
    if num_records < 2:
        return []

    seq_list = [
        np.frombuffer(r["seq"].encode("ascii"), dtype=np.uint8)
        for r in processed_records
    ]
    seq_len = np.array([len(s) for s in seq_list], dtype=np.int32)

    seq_offset = np.zeros(num_records, dtype=np.int32)
    for i in range(1, num_records):
        seq_offset[i] = seq_offset[i - 1] + seq_len[i - 1]

    full_seq = np.concatenate(seq_list)

    id_bytes_list = []
    for r in processed_records:
        b = r["id"].encode("ascii")[:ID_LEN]
        b = b.ljust(ID_LEN, b"\x00")
        id_bytes_list.append(b)
    full_id = b"".join(id_bytes_list)

    res_ptr = lib.nw_align_c_full(
        num_records,
        full_seq.ctypes.data,
        seq_len.ctypes.data,
        seq_offset.ctypes.data,
        full_id,
        BLOSUM_MATRIX.ctypes.data,
        MATRIX_COLS,
        CHAR_TO_IDX.ctypes.data,
        GAP_O,
        GAP_E,
        MIN_INF,
    )

    if not res_ptr:
        raise MemoryError("C malloc failed inside nw_align_c_full")

    total_pairs = num_records * (num_records - 1) // 2
    results = []

    for k in range(total_pairs):
        res = res_ptr[k]

        al1_str = ctypes.string_at(res.al1).decode("ascii")
        al2_str = ctypes.string_at(res.al2).decode("ascii")

        results.append(
            {
                "Uniprot ID 1": res.id1.decode("ascii"),
                "Uniprot ID 2": res.id2.decode("ascii"),
                "DP Score": res.score,
                "Aligned Sequence 1": al1_str,
                "Aligned Sequence 2": al2_str,
                "Sequence Identity (Length 1)": res.identity_len1,
                "Sequence Identity (Length 2)": res.identity_len2,
            }
        )

    try:
        libc = ctypes.CDLL(None)
    except Exception:
        libc = ctypes.cdll.msvcrt if os.name == "nt" else ctypes.CDLL("libc.so.6")
    libc.free(res_ptr)

    return results
