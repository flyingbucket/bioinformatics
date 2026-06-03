import os
import time
import threading
from dataclasses import dataclass
import pandas as pd
import psutil
from Bio import SeqIO

from nw_align import nw_align_py, nw_align_numba, backtrack_alignment
from nw_align_kernel_only_wrapper import nw_align_c as nw_align_c_kernel_only
from nw_align_c_all_wrapper import nw_align_c_all_records


@dataclass
class BenchmarkResult:
    name: str
    time_s: float
    peak_mem_mb: float
    df: pd.DataFrame


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
    df = pd.DataFrame(results)
    return df


def run_pure_python(processed_records, pid) -> BenchmarkResult:
    print("\033[31mRunning Pure Python Version...\033[0m")
    monitor = MemoryMonitor(pid)
    monitor.start()

    start = time.perf_counter()
    df = run_alignment_pipeline(processed_records, nw_align_py)
    elapsed = time.perf_counter() - start

    monitor.stop()
    monitor.join()
    return BenchmarkResult(
        "Pure Python", elapsed, monitor.peak_memory / (1024 * 1024), df
    )


def run_c_kernel_only(processed_records, pid) -> BenchmarkResult:
    print("\033[31mRunning C Kernel Version...\033[0m")
    monitor = MemoryMonitor(pid)
    monitor.start()

    start = time.perf_counter()
    df = run_alignment_pipeline(processed_records, nw_align_c_kernel_only)
    elapsed = time.perf_counter() - start

    monitor.stop()
    monitor.join()
    return BenchmarkResult(
        "C_Kernel_Only", elapsed, monitor.peak_memory / (1024 * 1024), df
    )


def run_full_c(processed_records, pid) -> BenchmarkResult:
    print("\033[31mRunning Full C Version...\033[0m")
    monitor = MemoryMonitor(pid)
    monitor.start()

    start = time.perf_counter()
    results = nw_align_c_all_records(processed_records)
    elapsed = time.perf_counter() - start

    df = pd.DataFrame(results)

    monitor.stop()
    monitor.join()
    return BenchmarkResult("Full_C", elapsed, monitor.peak_memory / (1024 * 1024), df)


def run_numba(processed_records, pid) -> BenchmarkResult:
    print("\033[31mRunning Numba Version...\033[0m")
    monitor = MemoryMonitor(pid)
    monitor.start()

    start = time.perf_counter()
    df = run_alignment_pipeline(processed_records, nw_align_numba)
    elapsed = time.perf_counter() - start

    monitor.stop()
    monitor.join()
    return BenchmarkResult("Numba", elapsed, monitor.peak_memory / (1024 * 1024), df)


def analyze_mismatch(
    df_base: pd.DataFrame, df_target: pd.DataFrame, name_base: str, name_target: str
):
    print(
        f"\n          \033[33m[Diagnosing] Investigating mismatch between '{name_base}' and '{name_target}':\033[0m"
    )

    if df_base.shape != df_target.shape:
        print(
            f"        [Shape Error] Row/Col counts mismatch: {name_base} {df_base.shape} vs {name_target} {df_target.shape}"
        )
        return

    # 排查是否仅仅是行循环顺序不对
    df_base_sorted = df_base.sort_values(
        by=["Uniprot ID 1", "Uniprot ID 2"]
    ).reset_index(drop=True)
    df_target_sorted = df_target.sort_values(
        by=["Uniprot ID 1", "Uniprot ID 2"]
    ).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            df_base_sorted,
            df_target_sorted,
            check_exact=False,
            atol=1e-6,
            check_dtype=False,
        )
        print(
            "        \033[36m[Diagnosis Result]\033[0m 发现问题！两边的数据内容【完全一致】，但【行顺序不一致】。"
        )
        print(
            f"        说明 '{name_target}' 内部实现的嵌套循环顺序（或索引累加逻辑）与要求的标准 FASTA 字典序有出入。"
        )
        return
    except AssertionError:
        print(
            "        排序后仍不一致。说明不仅仅是行顺序问题，内部计算或任务覆盖有实质性差异。"
        )

    # 维度 2：检查【前两列 Uniprot ID 作为索引是否完全一致】（验证是否完成了完全相同的比对任务）
    idx_cols = ["Uniprot ID 1", "Uniprot ID 2"]
    ids_base = df_base_sorted[idx_cols]
    ids_target = df_target_sorted[idx_cols]

    try:
        pd.testing.assert_frame_equal(ids_base, ids_target, check_dtype=False)
        print(
            "        [Index Check] 任务索引对齐通过。两组代码完成了【完全相同】的对偶比对任务。"
        )

        # 维度 3：索引一致，但后面的数据列（得分、对齐序列、相似度）不一样，精准抓出内鬼
        print("        [Cell Check] 开始精确定位不一致的行列...")

        # 将 ID 设为索引，方便做对齐矩阵减法/比对
        df_base_indexed = df_base_sorted.set_index(idx_cols)
        df_target_indexed = df_target_sorted.set_index(idx_cols)

        # 逐列排查
        data_cols = [
            "DP Score",
            "Aligned Sequence 1",
            "Aligned Sequence 2",
            "Sequence Identity (Length 1)",
            "Sequence Identity (Length 2)",
        ]
        for col in data_cols:
            if col in [
                "DP Score",
                "Sequence Identity (Length 1)",
                "Sequence Identity (Length 2)",
            ]:
                # 数值列：找出绝对误差大于 1e-6 的行
                diff_mask = (df_base_indexed[col] - df_target_indexed[col]).abs() > 1e-6
            else:
                # 字符串列：直接对比是否相等
                diff_mask = df_base_indexed[col] != df_target_indexed[col]

            mismatch_rows = df_base_indexed[diff_mask]
            if not mismatch_rows.empty:
                print(
                    f"             列 \033[31m'{col}'\033[0m 存在不一致！共计 {len(mismatch_rows)} 行不匹配。"
                )
                print("             举例不一致的样本行（前 2 行）：")
                for idx_pair in mismatch_rows.index[:2]:  # pyright: ignore
                    val_base = df_base_indexed.loc[idx_pair, col]
                    val_target = df_target_indexed.loc[idx_pair, col]
                    print(
                        f"             - 任务 {idx_pair}: {name_base} = {val_base} | {name_target} = {val_target}"
                    )

    except AssertionError:
        print("         [Index Error] 两边生成的 Uniprot ID 对偶索引不一致！")
        # 找出哪些任务在 target 中缺失了，或者多出来了
        set_base = set(zip(df_base[idx_cols[0]], df_base[idx_cols[1]]))
        set_target = set(zip(df_target[idx_cols[0]], df_target[idx_cols[1]]))

        missing = set_base - set_target
        redundant = set_target - set_base
        if missing:
            print(
                f"          - '{name_target}' 相比 '{name_base}' \033[31m缺失\033[0m 了 {len(missing)} 个比对任务。示例: {list(missing)[:2]}"
            )
        if redundant:
            print(
                f"          - '{name_target}' 相比 '{name_base}' \033[31m多出\033[0m 了 {len(redundant)} 个意料之外的任务。示例: {list(redundant)[:2]}"
            )


def validate_consistency(results_list: list[BenchmarkResult]):
    print(
        "\n\033[34m[Validation] Starting Strict Sequence Order Consistency Check...\033[0m"
    )
    if not results_list:
        return

    # 以第一个跑出来的方案（如 C Kernel Only 或 Pure Python）作为绝对基准
    base_result = results_list[0]
    all_passed = True

    for target in results_list[1:]:
        try:
            # 严格验证：不仅内容要对，每一行的相对顺序也必须完全一致
            pd.testing.assert_frame_equal(
                base_result.df,
                target.df,
                check_exact=False,
                atol=1e-6,
                check_dtype=False,
            )
            print(
                f"  └─ \033[32m[PASS]\033[0m {base_result.name} == {target.name} (原始行顺序与内容完全一致)"
            )
        except AssertionError:
            all_passed = False
            print(
                f"  └─ \033[31m[FAIL]\033[0m {base_result.name} != {target.name} (触发深度诊断...)"
            )
            # 触发诊断机制
            analyze_mismatch(base_result.df, target.df, base_result.name, target.name)

    if all_passed:
        print(
            "\n\033[32m  All versions produced perfectly identical DataFrames in correct FASTA order!\033[0m"
        )
    else:
        print(
            "\n\033[31m  Warning: Consistency validation failed. Please check the diagnostics above.\033[0m"
        )


def print_benchmark_report(
    results_list: list[BenchmarkResult], preload_time: float, save_time: float
):
    print("\n\033[33mBENCHMARK RESULTS\033[0m")
    print(f"{'nw_align func':<18}{'time (s)':<18}{'Peak Memory(MB)':<22}")
    print("-" * 60)
    for res in results_list:
        print(f"{res.name:<18}{res.time_s:<18.4f}{res.peak_mem_mb:<22.2f}")
    print("-" * 60)

    print("\nOther time consumption:")
    print(f"FASTA Data Parsing: {preload_time:.4f} s")
    print(f"Saving DataFrame Excel/CSV : {save_time:.4f} s")


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"
    pid = os.getpid()

    print("\033[33mStart benchmark\033[0m")

    start_time = time.perf_counter()
    processed_records = []
    for rec in SeqIO.parse(fasta_path, format="fasta"):
        parts = rec.id.split("|")
        uniprot_id = parts[1]
        seq_str = str(rec.seq)
        processed_records.append({"id": uniprot_id, "seq": seq_str})
    preload_time = time.perf_counter() - start_time
    print("\033[31mFASTA data preprocess finished\033[0m")

    dummy_seq = "ACDEF"
    _ = nw_align_numba(dummy_seq, dummy_seq)
    _ = backtrack_alignment(dummy_seq, dummy_seq, *_[1:4], _[0])

    bench_records = []

    # bench_records.append(run_pure_python(processed_records, pid))
    bench_records.append(run_c_kernel_only(processed_records, pid))
    bench_records.append(run_full_c(processed_records, pid))
    bench_records.append(run_numba(processed_records, pid))

    validate_consistency(bench_records)

    print("\n\033[31mSaving result DataFrame\033[0m")
    start_time = time.perf_counter()
    output_dir = "./artifacts/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    for record in bench_records:
        df = record.df
        df.to_csv(f"{output_dir}/{record.name}.csv", index=False)
    save_time = time.perf_counter() - start_time

    print_benchmark_report(bench_records, preload_time, save_time)


if __name__ == "__main__":
    main()
