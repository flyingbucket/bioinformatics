import numpy as np
from Bio.Align import substitution_matrices
from numba import njit

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_O = -11
GAP_E = -1
MIN_INF = -999999

BLOSUM_MATRIX = np.array(BLOSUM62, dtype=np.int32)
CHAR_TO_IDX = np.full(256, -1, dtype=np.int32)
for idx, char in enumerate(BLOSUM62.alphabet):  # pyright: ignore[]
    CHAR_TO_IDX[ord(char)] = idx


def nw_align_py(aa1: str, aa2: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    l1 = len(aa1)
    l2 = len(aa2)
    M = np.full((l1 + 1, l2 + 1), MIN_INF, dtype=np.int32)
    X = np.full((l1 + 1, l2 + 1), MIN_INF, dtype=np.int32)
    Y = np.full((l1 + 1, l2 + 1), MIN_INF, dtype=np.int32)
    M[0, 0] = 0
    for i in range(1, l1 + 1):
        X[i, 0] = GAP_O + (i - 1) * GAP_E
    for j in range(1, l2 + 1):
        Y[0, j] = GAP_O + (j - 1) * GAP_E
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            c1, c2 = aa1[i - 1], aa2[j - 1]
            match_score = int(BLOSUM62[c1, c2])  # pyright: ignore[reportArgumentType, reportCallIssue]
            M[i, j] = match_score + max(
                M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]
            )
            X[i, j] = max(GAP_O + M[i - 1, j], GAP_E + X[i - 1, j])
            Y[i, j] = max(GAP_O + M[i, j - 1], GAP_E + Y[i, j - 1])
    best_score = max(M[l1, l2], X[l1, l2], Y[l1, l2])
    return int(best_score), M, X, Y


@njit
def nw_align_kernel(
    arr1: np.ndarray,
    arr2: np.ndarray,
    matrix: np.ndarray,
    char_to_idx: np.ndarray,
    gap_o: int,
    gap_e: int,
    min_inf: int,
):
    l1 = len(arr1)
    l2 = len(arr2)

    M = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)
    X = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)
    Y = np.full((l1 + 1, l2 + 1), min_inf, dtype=np.int32)

    M[0, 0] = 0
    for i in range(1, l1 + 1):
        X[i, 0] = gap_o + (i - 1) * gap_e
    for j in range(1, l2 + 1):
        Y[0, j] = gap_o + (j - 1) * gap_e

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            idx1 = char_to_idx[arr1[i - 1]]
            idx2 = char_to_idx[arr2[j - 1]]
            match_score = matrix[idx1, idx2]

            m_prev = M[i - 1, j - 1]
            x_prev = X[i - 1, j - 1]
            y_prev = Y[i - 1, j - 1]
            M[i, j] = match_score + max(m_prev, max(x_prev, y_prev))

            m_left = gap_o + M[i - 1, j]
            x_left = gap_e + X[i - 1, j]
            X[i, j] = m_left if m_left > x_left else x_left

            m_up = gap_o + M[i, j - 1]
            y_up = gap_e + Y[i, j - 1]
            Y[i, j] = m_up if m_up > y_up else y_up
    best_score = max(M[l1, l2], max(X[l1, l2], Y[l1, l2]))

    return best_score, M, X, Y


def nw_align_numba(
    seq1_str: str, seq2_str: str
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    arr1 = np.frombuffer(seq1_str.encode("ascii"), dtype=np.uint8)
    arr2 = np.frombuffer(seq2_str.encode("ascii"), dtype=np.uint8)
    return nw_align_kernel(
        arr1, arr2, BLOSUM_MATRIX, CHAR_TO_IDX, GAP_O, GAP_E, MIN_INF
    )
