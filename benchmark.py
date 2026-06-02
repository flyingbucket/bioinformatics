import os
import time
import threading
import pandas as pd
import psutil
from Bio import SeqIO

from nw_align import nw_align_py, nw_align_numba, backtrack_alignment
from nw_align_kernel_only_wrapper import nw_align_c


class MemoryMonitor(threading.Thread):
    def __init__(self, pid, interval=0.002):
        super().__init__()
        self.pid = pid
        self.interval = interval
        self.peak_memory = 0
        self.stopped = threading.Event()

    def run(self):
        try:
            proc = psutil.Process(self.pid)
            while not self.stopped.is_set():
                mem = proc.memory_info().rss
                if mem > self.peak_memory:
                    self.peak_memory = mem
                time.sleep(self.interval)
        except Exception:
            pass

    def stop(self):
        self.stopped.set()


def run_alignment_pipeline(processed_records, align_func, **kwargs):
    start = time.perf_counter()
    num_records = len(processed_records)
    results = []

    for i in range(num_records):
        rec1 = processed_records[i]
        for j in range(i + 1, num_records):
            rec2 = processed_records[j]

            id1, seq1_str = rec1["id"], rec1["seq"]
            id2, seq2_str = rec2["id"], rec2["seq"]

            score, M, X, Y = align_func(seq1_str, seq2_str, **kwargs)
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
    elapsed = time.perf_counter()
    return results, elapsed - start


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"
    pid = os.getpid()

    print("\033[33mStart benchmark\033[0m")

    # data preprocess
    start_time = time.perf_counter()
    processed_records = []
    for rec in SeqIO.parse(fasta_path, format="fasta"):
        parts = rec.id.split("|")
        uniprot_id = parts[1]
        seq_str = str(rec.seq)
        processed_records.append({"id": uniprot_id, "seq": seq_str})
    preload_time = time.perf_counter() - start_time
    print("\033[31mFASTA data preprocess finished\033[0m")

    # precompile to warmup numba
    dummy_seq = "ACDEF"
    _ = nw_align_numba(dummy_seq, dummy_seq)
    _ = backtrack_alignment(dummy_seq, dummy_seq, *_[1:4], _[0])

    # print("\033[31mPure Python Version\033[0m")
    # monitor_py = MemoryMonitor(pid)
    # monitor_py.start()
    # _, py_time = run_alignment_pipeline(processed_records, nw_align_py)
    # monitor_py.stop()
    # monitor_py.join()
    # py_mem = monitor_py.peak_memory / (1024 * 1024)

    print("\033[31mC Version\033[0m")
    monitor_c = MemoryMonitor(pid)
    monitor_c.start()

    final_results, c_time = run_alignment_pipeline(processed_records, nw_align_c)

    monitor_c.stop()
    monitor_c.join()

    c_mem = monitor_c.peak_memory / (1024 * 1024)

    print("\033[31mNumba Version\033[0m")
    monitor_nb = MemoryMonitor(pid)
    monitor_nb.start()

    _, nb_time = run_alignment_pipeline(processed_records, nw_align_numba)

    monitor_nb.stop()
    monitor_nb.join()

    nb_mem = monitor_nb.peak_memory / (1024 * 1024)
    # save results
    print("\033[31mSaving result DataFrame\033[0m")
    start_time = time.perf_counter()
    df = pd.DataFrame(final_results)
    output_dir = "./artifacts2"
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(f"{output_dir}/submission_file_2.xlsx", index=False)
    df.to_csv(f"{output_dir}/submission_file_2.csv", index=False)
    save_time = time.perf_counter() - start_time

    # report
    print("\n\033[33mBENCHMARK RESULTS\033[0m")
    print(f"{'nw_align func':<18}{'time (s)':<18}{'Peak Memory(MB)':<22}")
    print("-" * 60)
    # print(f"{'Pure Python':<18}{py_time:<18.4f}{py_mem:<22.2f}")
    print(f"{'Numba':<18}{nb_time:<18.4f}{nb_mem:<22.2f}")
    print(f"{'C':<18}{c_time:<18.4f}{c_mem:<22.2f}")
    print("-" * 60)
    # print(f"Speedup ratio: {speedup}x times")
    print("\nOther time consumption:")
    print(f"FASTA Data Parsing: {preload_time:.4f} s")
    print(f"Saving DataFrame Excel/CSV : {save_time:.4f} s")


if __name__ == "__main__":
    main()
