import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from Bio.Align import substitution_matrices
from nw_align import nw_align_py, nw_align_numba

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_O = -11
GAP_E = -1
MIN_INF = -999999


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


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"

    processed_records = []
    for rec in SeqIO.parse(fasta_path, format="fasta"):
        parts = rec.id.split("|")
        uniprot_id = parts[1] if len(parts) > 1 else rec.id
        seq_str = str(rec.seq)

        processed_records.append({"id": uniprot_id, "seq": seq_str})

    num_records = len(processed_records)
    results = []

    for i in tqdm(range(num_records)):
        rec1 = processed_records[i]
        for j in range(i + 1, num_records):
            rec2 = processed_records[j]

            id1, seq1_str = rec1["id"], rec1["seq"]
            id2, seq2_str = rec2["id"], rec2["seq"]

            score, M, X, Y = nw_align_numba(seq1_str, seq2_str)
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
    output_dir = "./artifacts"
    os.makedirs(output_dir, exist_ok=True)

    df.to_excel(f"{output_dir}/submission_file_2.xlsx", index=False)
    df.to_csv(f"{output_dir}/submission_file_2.csv", index=False)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
