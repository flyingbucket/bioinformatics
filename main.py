import os
import pandas as pd
from Bio import SeqIO
from nw_align import nw_align_py, nw_align_numba, backtrack_alignment


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"

    processed_records = []
    for rec in SeqIO.parse(fasta_path, format="fasta"):
        parts = rec.id.split("|")
        uniprot_id = parts[1]
        seq_str = str(rec.seq)

        processed_records.append({"id": uniprot_id, "seq": seq_str})

    num_records = len(processed_records)
    results = []

    for i in range(num_records):
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
