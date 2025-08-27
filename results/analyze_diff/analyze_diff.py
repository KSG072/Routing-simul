# -*- coding: utf-8 -*-
# 파티셔닝 → 버킷단위 조인(순서보존 1:1) → 결과 append
# - 임의 경로(파일명 패턴 무관)로 ISL/GSL 쌍 지정
# - Status == success 필터
# - cross counts: GSL 파일 값만 결과에 저장
# - e2e delay: 있으면 그대로, 없으면 prop+queue로 보강
# - 큰 파일도 메모리 절약

import os, gc, csv, hashlib, ast, re, shutil
from typing import Dict, Tuple
import pandas as pd
from tqdm.auto import tqdm

N_BUCKETS = 64
READ_CHUNKSIZE = 1_000_000
INCLUDE_RESULTS = True
INCLUDE_QUEUING_DELAYS = True

KEY_COLS = ("Time (ms)", "User ID", "Destination Relay ID")
DESIRED_COLS = {
    # keys
    "Time (ms)", "User ID", "Destination Relay ID",
    # metrics
    "Path Length", "Propagation Delay", "Transmission Delay",
    "Queuing Delay", "Queuing delays", "e2e delay",
    # extras
    "result", "Status", "cross counts",
}

RENAME = {
    "Time (ms)": "time_ms",
    "User ID": "src",
    "Destination Relay ID": "dst",
    "Path Length": "path_length",
    "Propagation Delay": "prop_delay",
    "Transmission Delay": "tx_delay",
    "Queuing Delay": "queue_delay",
    "Queuing delays": "queue_delays",
    "e2e delay": "e2e_delay",
    "result": "result",
    "Status": "status",
    "cross counts": "cross_counts",
}

def stable_bucket(key: Tuple[str, str, str], mod: int) -> int:
    s = "\x1f".join(key)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def count_string_items(cell) -> int:
    if cell is None:
        return 0
    if isinstance(cell, list):
        seq = cell
    else:
        s = str(cell)
        try:
            seq = ast.literal_eval(s)
        except Exception:
            return len(re.findall(r"(?:'[^']*'|\"[^\"]*\")", s))
    try:
        return sum(isinstance(x, str) for x in seq)
    except Exception:
        return 0

def partition_csv(in_path: str, tmp_dir: str, prefix: str, chunksize: int = READ_CHUNKSIZE):
    ensure_dir(tmp_dir)
    writers: Dict[int, csv.writer] = {}
    files: Dict[int, any] = {}
    header_written = set()
    sorted_cols = None

    def get_writer(bi: int) -> csv.writer:
        if bi in writers:
            return writers[bi]
        f = open(os.path.join(tmp_dir, f"{prefix}_bucket_{bi:03d}.csv"), "a", newline="", encoding="utf-8")
        w = csv.writer(f); files[bi] = f; writers[bi] = w
        if bi not in header_written:
            w.writerow(sorted_cols)
            header_written.add(bi)
        return w

    def normalize_cols(cols):
        return [c.strip().replace("\ufeff", "") for c in cols]

    chunk_iter = pd.read_csv(
        in_path,
        usecols=lambda c: (c.strip().replace("\ufeff", "") in DESIRED_COLS),
        chunksize=chunksize, low_memory=False, encoding="utf-8-sig"
    )

    first = True
    for chunk in tqdm(chunk_iter, desc=f"[Partition] {os.path.basename(in_path)}", unit="chunk"):
        chunk.columns = normalize_cols(list(chunk.columns))

        # success만
        if "Status" in chunk.columns:
            mask = chunk["Status"].astype(str).str.strip().str.lower().eq("success")
            chunk = chunk[mask]
        if chunk.empty:
            continue

        if first:
            present = [c for c in chunk.columns if c in DESIRED_COLS]
            for col in KEY_COLS:
                if col not in present:
                    raise ValueError(f"{in_path}: key column '{col}' missing. got={chunk.columns.tolist()}")
            others = [c for c in present if c not in KEY_COLS]
            sorted_cols = list(KEY_COLS) + others
            first = False

        for col in KEY_COLS:
            chunk[col] = chunk[col].astype(str)

        idx_time = chunk.columns.get_loc("Time (ms)")
        idx_src  = chunk.columns.get_loc("User ID")
        idx_dst  = chunk.columns.get_loc("Destination Relay ID")

        for row in chunk.itertuples(index=False):
            t = str(row[idx_time]); s = str(row[idx_src]); d = str(row[idx_dst])
            bi = stable_bucket((t, s, d), N_BUCKETS)
            row_map = dict(zip(chunk.columns, row))
            get_writer(bi).writerow([row_map.get(c, "") for c in sorted_cols])

        del chunk
        gc.collect()

    for f in files.values():
        f.close()

def process_bucket(bucket_idx: int, tmp_dir: str, out_path: str, write_header: bool):
    isl_b = os.path.join(tmp_dir, f"isl_bucket_{bucket_idx:03d}.csv")
    gsl_b = os.path.join(tmp_dir, f"gsl_bucket_{bucket_idx:03d}.csv")
    if not (os.path.exists(isl_b) and os.path.exists(gsl_b)):
        return write_header

    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
        return df

    isl = pd.read_csv(isl_b, low_memory=False, encoding="utf-8-sig")
    gsl = pd.read_csv(gsl_b, low_memory=False, encoding="utf-8-sig")
    isl = normalize_df(isl); gsl = normalize_df(gsl)

    # rename / types
    for df in (isl, gsl):
        df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns}, inplace=True)
        for c in ("time_ms", "src", "dst"):
            if c not in df.columns:
                raise ValueError(f"bucket missing key '{c}': {isl_b if df is isl else gsl_b}")
            df[c] = df[c].astype(str)
        for numc in ("path_length", "prop_delay", "queue_delay", "tx_delay", "e2e_delay"):
            if numc in df.columns:
                df[numc] = pd.to_numeric(df[numc], errors="coerce")
        # 보강: e2e_delay 없으면 prop+queue(+tx?)로
        if "e2e_delay" not in df.columns or df["e2e_delay"].isna().all():
            # 우선순위: prop + queue + tx (있으면) → 없으면 가능한 것 합
            comp = []
            if "prop_delay" in df.columns:  comp.append(df["prop_delay"])
            if "queue_delay" in df.columns: comp.append(df["queue_delay"])
            if "tx_delay" in df.columns:    comp.append(df["tx_delay"])
            if comp:
                df["e2e_delay"] = sum(comp)
            else:
                df["e2e_delay"] = pd.NA
        if "result" not in df.columns: df["result"] = ""
        if "queue_delays" not in df.columns: df["queue_delays"] = ""
        if "cross_counts" not in df.columns: df["cross_counts"] = pd.NA  # ISL 쪽은 비어있어도 됨

    key = ["time_ms", "src", "dst"]

    # 파일 내 순서 보존 → 안정 정렬 + cumcount로 1:1 매칭
    isl = isl.sort_values(key, kind="stable")
    gsl = gsl.sort_values(key, kind="stable")
    isl["rk"] = isl.groupby(key).cumcount()
    gsl["rk"] = gsl.groupby(key).cumcount()

    merged = pd.merge(
        isl, gsl, on=key + ["rk"], how="inner", suffixes=("_isl", "_gsl"), validate="one_to_one"
    )

    # 파생값
    merged["path length diff"] = merged["path_length_isl"] - merged["path_length_gsl"]
    merged["queue delay diff"] = merged["queue_delay_isl"] - merged["queue_delay_gsl"]
    merged["e2e delay diff"]   = merged["e2e_delay_isl"]  - merged["e2e_delay_gsl"]
    merged["jump count"]       = merged["result_gsl"].map(count_string_items)

    # 결과 컬럼 (cross counts는 GSL의 것만)
    cols = [
        "time_ms", "src", "dst",
        "path_length_isl", "path_length_gsl", "path length diff",
        "prop_delay_isl",  "prop_delay_gsl",
        "queue_delay_isl", "queue_delay_gsl", "queue delay diff",
        "e2e_delay_isl",   "e2e_delay_gsl",   "e2e delay diff",
        "jump count",
        "cross_counts_gsl",
    ]
    if INCLUDE_RESULTS:
        cols[3:3] = ["result_isl", "result_gsl"]
    if INCLUDE_QUEUING_DELAYS:
        cols.insert(cols.index("jump count"), "queue_delays_isl")
        cols.insert(cols.index("jump count"), "queue_delays_gsl")

    out = merged.rename(columns={
        "time_ms": "Time (ms)",
        "path_length_isl": "isl path length",
        "path_length_gsl": "gsl path length",
        "prop_delay_isl":  "isl prop delay",
        "prop_delay_gsl":  "gsl prop delay",
        "queue_delay_isl": "isl queue delay",
        "queue_delay_gsl": "gsl queue delay",
        "queue_delays_isl": "isl queue delays",
        "queue_delays_gsl": "gsl queue delays",
        "e2e_delay_isl":   "isl e2e delay",
        "e2e_delay_gsl":   "gsl e2e delay",
        "result_isl": "isl result",
        "result_gsl": "gsl result",
        "cross_counts_gsl": "cross counts",   # ← GSL의 cross만
    })[cols]

    mode = "w" if write_header else "a"
    out.to_csv(out_path, index=False, mode=mode, header=write_header)
    write_header = False

    del isl, gsl, merged, out
    gc.collect()
    try:
        os.remove(isl_b); os.remove(gsl_b)
    except FileNotFoundError:
        pass
    return write_header

def build_comparison(isl_path: str, gsl_path: str, out_path: str):
    tmp_dir = "_tmp_buckets"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    print("[1/3] ISL 파티셔닝...")
    partition_csv(isl_path, tmp_dir, prefix="isl", chunksize=READ_CHUNKSIZE)

    print("[2/3] GSL 파티셔닝...")
    partition_csv(gsl_path, tmp_dir, prefix="gsl", chunksize=READ_CHUNKSIZE)

    print("[3/3] 버킷 조인 → append...")
    write_header = True
    for bi in tqdm(range(N_BUCKETS), desc="[Join] buckets", unit="bucket"):
        write_header = process_bucket(bi, tmp_dir, out_path, write_header)

    print(f"[완료] 결과 저장: {out_path}")
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

# ----------------------------
# 사용 예시 (파일명/패턴 상관없이 경로 직접 지정)
# ----------------------------
# build_comparison(
#     isl_path="limited_Q_only_ISL_1.csv",
#     gsl_path="limited_Q_with_GSL_1.csv",
#     out_path="comparison_results_with_jump_limited_1.csv",
# )
# build_comparison(
#     isl_path="infinite_Q_only_ISL_1.csv",
#     gsl_path="infinite_Q_with_GSL_1.csv",
#     out_path="comparison_results_with_jump_infinite_1.csv",
# )
