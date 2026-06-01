import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import substitution_matrices
from tqdm import tqdm

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_O = -11
GAP_E = -1
MIN_INF = -999999


def nm_align(aa1: str, aa2: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
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


def backtrack_alignment(
    aa1: str, aa2: str, M: np.ndarray, X: np.ndarray, Y: np.ndarray, best_score: int
) -> tuple[str, str]:
    i, j = len(aa1), len(aa2)
    aligned_a1 = []
    aligned_a2 = []

    if best_score == M[i, j]:
        curr_state = "M"
    elif best_score == X[i, j]:
        curr_state = "X"
    else:
        curr_state = "Y"

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if curr_state == "M":
                c1, c2 = aa1[i - 1], aa2[j - 1]
                match_score = int(BLOSUM62[c1, c2])  # pyright: ignore[reportArgumentType, reportCallIssue]
                aligned_a1.append(c1)
                aligned_a2.append(c2)

                # 逆推：判断当前格子是由上一格的哪个状态转移过来的
                if M[i, j] == match_score + M[i - 1, j - 1]:
                    curr_state = "M"
                elif M[i, j] == match_score + X[i - 1, j - 1]:
                    curr_state = "X"
                else:
                    curr_state = "Y"
                i -= 1
                j -= 1
            elif curr_state == "X":
                # X 状态代表 S2 有 Gap，说明 S1 往前退了一格
                aligned_a1.append(aa1[i - 1])
                aligned_a2.append("-")
                if X[i, j] == GAP_O + M[i - 1, j]:
                    curr_state = "M"
                else:
                    curr_state = "X"
                i -= 1
            elif curr_state == "Y":
                # Y 状态代表 S1 有 Gap，说明 S2 往前退了一格
                aligned_a1.append("-")
                aligned_a2.append(aa2[j - 1])
                if Y[i, j] == GAP_O + M[i, j - 1]:
                    curr_state = "M"
                else:
                    curr_state = "Y"
                j -= 1
        elif i > 0:
            # 边界情况：S2 已经退到头了，S1 剩下的全变成 Gap
            aligned_a1.append(aa1[i - 1])
            aligned_a2.append("-")
            i -= 1
        elif j > 0:
            # 边界情况：S1 已经退到头了，S2 剩下的全变成 Gap
            aligned_a1.append("-")
            aligned_a2.append(aa2[j - 1])
            j -= 1

    return "".join(reversed(aligned_a1)), "".join(reversed(aligned_a2))


def parse_uniprot_id(fasta_id: str) -> str:
    """从 FASTA 头部标头（如 sp|C0HJQ9|MYG_HUMAN）中精准提取 UniProt ID"""
    parts = fasta_id.split("|")
    if len(parts) > 1:
        return parts[1]
    return fasta_id


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"
    records = list(SeqIO.parse(fasta_path, format="fasta"))
    num_records = len(records)

    results = []

    for i in tqdm(range(num_records)):
        for j in range(i + 1, num_records):
            rec1 = records[i]
            rec2 = records[j]

            id1 = parse_uniprot_id(rec1.id)
            id2 = parse_uniprot_id(rec2.id)

            seq1_str = str(rec1.seq)
            seq2_str = str(rec2.seq)

            score, M, X, Y = nm_align(seq1_str, seq2_str)
            al1, al2 = backtrack_alignment(seq1_str, seq2_str, M, X, Y, score)

            matches = sum(1 for a, b in zip(al1, al2) if a == b and a != "-")

            identity_len1 = matches / len(seq1_str)
            identity_len2 = matches / len(seq2_str)

            results.append(
                {
                    "Uniprot ID 1": id1,
                    "Uniprot ID 2": id2,
                    "DP Score": score,
                    "Aligned Sequence 1": al1,
                    "Aligned Sequence 2": al2,
                    "Sequence Identity (Length 1)": identity_len1,
                    "Sequence Identity (Length 2)": identity_len2,
                }
            )

    df = pd.DataFrame(results)
    output_excel_path = "./artifacts/submission_file_2.xlsx"

    df.to_excel(output_excel_path, index=False)
    print(f"Saved to: {output_excel_path}")


if __name__ == "__main__":
    main()
