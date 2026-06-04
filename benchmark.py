import multiprocessing
import os
import time
from dataclasses import dataclass
from functools import partial

import pandas as pd
import psutil
from Bio import SeqIO
from Bio.Align import (
    PairwiseAligner,
    substitution_matrices,
)

from nw_align.nw_align import backtrack_alignment, nw_align_numba, nw_align_py
from nw_align.nw_align_c_all_wrapper import (
    nw_align_c_all_omp_records,
    nw_align_c_all_records,
)
from nw_align.nw_align_kernel_only_wrapper import nw_align_c as nw_align_c_kernel_only


@dataclass
class BenchmarkResult:
    name: str
    time_s: float
    base_mem_mb: float
    peak_delta_mb: float
    df: pd.DataFrame


def format_time(seconds: float) -> str:
    """根据时间跨度优雅地格式化时间输出"""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    elif seconds >= 60.0:
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes} m {rem_seconds:.2f} s"
    else:
        return f"{seconds:.4f} s"


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
                    "seq1_uniprot_id": id1,
                    "seq2_uniprot_id": id2,
                    "NW_affine_score": score,
                    "aligned_seq1_with_gap": al1,
                    "aligned_seq2_with_gap": al2,
                    "identity_by_seq1_length": identity_len1,
                    "identity_by_seq2_length": identity_len2,
                }
            )
    df = pd.DataFrame(results)
    return df


def task_worker(task_name, processed_records, ready_event, start_event, result_queue):
    """独立子进程执行体"""
    try:
        if task_name == "Pure Python":
            func = partial(run_alignment_pipeline, processed_records, nw_align_py)
        elif task_name == "C_Kernel_Only":
            func = partial(
                run_alignment_pipeline, processed_records, nw_align_c_kernel_only
            )
        elif task_name == "Numba":
            func = partial(run_alignment_pipeline, processed_records, nw_align_numba)
        elif task_name == "Full_C":

            def func():
                return pd.DataFrame(nw_align_c_all_records(processed_records))
        elif task_name == "Full_C_OMP":

            def func():
                return pd.DataFrame(nw_align_c_all_omp_records(processed_records))
        elif task_name == "BioPython":

            def func():
                return run_biopython_pipeline(processed_records)
        else:
            raise ValueError(f"Unknown task name: {task_name}")

        # 挂起并通知主进程
        ready_event.set()

        # 等待主进程抓取完毕后发出开跑
        start_event.wait()

        # 探索性测试
        start_time = time.perf_counter()
        df = func()
        first_elapsed = time.perf_counter() - start_time

        if first_elapsed < 5.0:
            total_elapsed = first_elapsed
            for _ in range(19):
                sub_start = time.perf_counter()
                _ = func()
                total_elapsed += time.perf_counter() - sub_start
            elapsed = total_elapsed / 20.0
        else:
            elapsed = first_elapsed

        tmp_dir = "./artifacts/benchmark/.tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"{task_name}.pkl")
        df.to_pickle(tmp_path)

        result_queue.put((elapsed, None))
    # except Exception:
    #     import traceback
    #
    #     result_queue.put((0.0, traceback.format_exc()))
    except Exception:
        import traceback

        exc_info = traceback.format_exc()
        if not ready_event.is_set():
            ready_event.set()
        try:
            time.sleep(0.1)
            result_queue.put((0.0, exc_info))
        except Exception:
            os._exit(1)


def run_biopython_pipeline(processed_records):
    aligner = PairwiseAligner()
    aligner.mode = "global"

    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0

    num_records = len(processed_records)
    results = []

    for i in range(num_records):
        rec1 = processed_records[i]
        for j in range(i + 1, num_records):
            rec2 = processed_records[j]

            id1, seq1_str = rec1["id"], rec1["seq"]
            id2, seq2_str = rec2["id"], rec2["seq"]

            alignments = aligner.align(seq1_str, seq2_str)
            best_alignment = alignments[0]

            score = best_alignment.score  # pyright: ignore
            al1, al2 = best_alignment[0], best_alignment[1]

            matches = sum(1 for a, b in zip(al1, al2) if a == b and a != "-")  # pyright: ignore
            identity_len1 = matches / len(seq1_str)
            identity_len2 = matches / len(seq2_str)

            results.append(
                {
                    "seq1_uniprot_id": id1,
                    "seq2_uniprot_id": id2,
                    "NW_affine_score": score,
                    "aligned_seq1_with_gap": al1,
                    "aligned_seq2_with_gap": al2,
                    "identity_by_seq1_length": identity_len1,
                    "identity_by_seq2_length": identity_len2,
                }
            )
    return pd.DataFrame(results)


def run_task_in_process(task_name, processed_records) -> BenchmarkResult:
    print(f"\033[31mRunning {task_name} Version (Isolated Process)...\033[0m")

    ready_event = multiprocessing.Event()
    start_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=task_worker,
        args=(task_name, processed_records, ready_event, start_event, result_queue),
    )
    p.start()

    # 等待子进程就绪
    ready_event.wait()

    try:
        proc = psutil.Process(p.pid)
        base_mem = proc.memory_info().rss
    except psutil.NoSuchProcess:
        base_mem = 0

    peak_mem = base_mem

    start_event.set()

    while p.is_alive():
        try:
            mem = proc.memory_info().rss  # pyright: ignore
            if mem > peak_mem:
                peak_mem = mem
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.002)

    elapsed, error = result_queue.get()
    p.join()

    if error:
        raise RuntimeError(f"Task [{task_name}] failed with exception:\n{error}")

    tmp_path = f"./artifacts/benchmark/.tmp/{task_name}.pkl"
    if os.path.exists(tmp_path):
        df = pd.read_pickle(tmp_path)
        os.remove(tmp_path)
    else:
        df = pd.DataFrame()

    base_mem_mb = base_mem / (1024 * 1024)
    peak_delta_mb = max(0.0, (peak_mem - base_mem) / (1024 * 1024))

    return BenchmarkResult(task_name, elapsed, base_mem_mb, peak_delta_mb, df)  # pyright: ignore


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

    df_base_sorted = df_base.sort_values(
        by=["seq1_uniprot_id", "seq2_uniprot_id"]
    ).reset_index(drop=True)
    df_target_sorted = df_target.sort_values(
        by=["seq1_uniprot_id", "seq2_uniprot_id"]
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
    idx_cols = ["seq1_uniprot_id", "seq2_uniprot_id"]
    ids_base = df_base_sorted[idx_cols]
    ids_target = df_target_sorted[idx_cols]

    try:
        pd.testing.assert_frame_equal(ids_base, ids_target, check_dtype=False)
        print(
            "        [Index Check] 任务索引对齐通过。两组代码完成了【完全相同】的对偶比对任务。"
        )
        print("        [Cell Check] 开始精确定位不一致的行列...")

        df_base_indexed = df_base_sorted.set_index(idx_cols)
        df_target_indexed = df_target_sorted.set_index(idx_cols)
        data_cols = [
            "NW_affine_score",
            "aligned_seq1_with_gap",
            "aligned_seq2_with_gap",
            "identity_by_seq1_length",
            "identity_by_seq2_length",
        ]
        for col in data_cols:
            if col in [
                "DP Score",
                "Sequence Identity (Length 1)",
                "Sequence Identity (Length 2)",
            ]:
                diff_mask = (df_base_indexed[col] - df_target_indexed[col]).abs() > 1e-6
            else:
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

    base_result = results_list[0]
    all_passed = True

    for target in results_list[1:]:
        try:
            pd.testing.assert_frame_equal(
                base_result.df,
                target.df,
                check_exact=False,
                atol=1e-6,
                check_dtype=False,
            )
            print(f"  └─ \033[32m[PASS]\033[0m {base_result.name} == {target.name}")
        except AssertionError:
            all_passed = False
            print(
                f"  └─ \033[31m[FAIL]\033[0m {base_result.name} != {target.name} (触发深度诊断...)"
            )
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
    print(
        f"{'nw_align func':<18}{'Time':<18}{'Base Mem (MB)':<18}{'Mem Increment (MB)':<20}"
    )
    print("-" * 74)
    for res in results_list:
        time_str = format_time(res.time_s)
        print(
            f"{res.name:<18}{time_str:<18}{res.base_mem_mb:<18.2f}{res.peak_delta_mb:<20.2f}"
        )
    print("-" * 74)

    print("\nOther time consumption:")
    print(f"FASTA Data Parsing: {format_time(preload_time)}")
    print(f"Saving DataFrame Excel/CSV : {format_time(save_time)}")


def main():
    fasta_path = "./data/uniprotkb_protein_name_myoglobin_standard20_A.fast.fasta"

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

    bench_records.append(run_task_in_process("BioPython", processed_records))
    bench_records.append(run_task_in_process("Pure Python", processed_records))
    bench_records.append(run_task_in_process("C_Kernel_Only", processed_records))
    bench_records.append(run_task_in_process("Numba", processed_records))
    bench_records.append(run_task_in_process("Full_C", processed_records))
    bench_records.append(run_task_in_process("Full_C_OMP", processed_records))

    validate_consistency(bench_records)

    print("\n\033[31mSaving result DataFrame\033[0m")
    start_time = time.perf_counter()
    output_dir = "./artifacts/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    for record in bench_records:
        df = record.df
        df.to_csv(f"{output_dir}/{record.name}.csv", index=False)
    save_time = time.perf_counter() - start_time

    summary_data = [
        {
            "version": res.name,
            "time_seconds": res.time_s,
            "time_formatted": format_time(res.time_s),
            "base_mem_mb": res.base_mem_mb,
            "peak_delta_mb": res.peak_delta_mb,
        }
        for res in bench_records
    ]
    pd.DataFrame(summary_data).to_csv(
        f"{output_dir}/benchmark_summary.csv", index=False
    )
    print_benchmark_report(bench_records, preload_time, save_time)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
