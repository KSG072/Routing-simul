#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node_counts.py
- 'result' 내 모든 노드(정수=위성, 문자열=지상)를 집계
- Status로 success/drop 분리
- 파일명에서 arrival rate 추출 → rate별 분리 저장 또는 단일 CSV에 arrival_rate 추가
출력 컬럼: node_id, total_counts, success_counts, drop_counts[, arrival_rate]
"""

import ast, re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import pandas as pd
from tqdm.auto import tqdm

READ_CHUNKSIZE = 500_000
ENCODING = "utf-8-sig"
NEEDED_COLS = {"result", "Status"}

def parse_result(cell) -> Optional[list]:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        toks = re.findall(r"(?:'([^']*)'|\"([^\"]*)\")", s)
        if not toks:
            return None
        return [a or b for (a, b) in toks]

def parse_rate_from_name(p: Path) -> Optional[str]:
    m = re.search(r'_(\d+)\.csv$', p.name)
    return m.group(1) if m else None

def normalize_cols(cols):
    return [str(c).strip().replace("\ufeff", "") for c in cols]

def run(csv_files: Iterable[str],
        out_base: str = "node_counts",
        split_by_rate: bool = False) -> None:
    # rate -> { node_id -> counts }
    counts_by_rate: Dict[str, Dict[Any, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total_counts":0,"success_counts":0,"drop_counts":0}))

    def _usecols(c: str) -> bool:
        c = str(c).strip().replace("\ufeff", "")
        return (c in {"result","Status"})

    for fp in csv_files:
        path = Path(fp)
        if not path.exists():
            print(f"[warn] not found: {path}")
            continue
        rate = parse_rate_from_name(path) or "unknown"

        chunks = pd.read_csv(path, usecols=_usecols, chunksize=READ_CHUNKSIZE,
                              low_memory=False, encoding="utf-8-sig")
        for ch in tqdm(chunks, desc=f"[scan] {path.name}", unit="chunk"):
            ch.columns = normalize_cols(ch.columns)
            if "result" not in ch.columns:
                continue
            status = ch["Status"].astype(str).str.strip().str.lower() if "Status" in ch.columns else None
            seqs = ch["result"].map(parse_result)

            for i, seq in enumerate(seqs):
                if not seq:
                    continue
                succ = bool(status.iloc[i] == "success") if status is not None else False
                for node in seq:
                    rec = counts_by_rate[rate][node]
                    rec["total_counts"] += 1
                    if succ: rec["success_counts"] += 1
                    else:    rec["drop_counts"] += 1

    # write out
    if split_by_rate:
        for rate, table in counts_by_rate.items():
            rows = [
                {"node_id": nid,
                 "total_counts": v["total_counts"],
                 "success_counts": v["success_counts"],
                 "drop_counts": v["drop_counts"]}
                for nid, v in table.items()
            ]
            df_out = pd.DataFrame(rows, columns=["node_id","total_counts","success_counts","drop_counts"]) \
                        .sort_values(["total_counts","success_counts"], ascending=[False, False])
            out_path = f"{out_base}_rate_{rate}.csv"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            print(f"[ok] saved: {out_path} (nodes={len(df_out)})")
    else:
        rows = []
        for rate, table in counts_by_rate.items():
            for nid, v in table.items():
                rows.append({
                    "node_id": nid,
                    "total_counts": v["total_counts"],
                    "success_counts": v["success_counts"],
                    "drop_counts": v["drop_counts"],
                    "arrival_rate": rate
                })
        if not rows:
            print("[info] no nodes found; nothing to write.")
            return
        df_out = pd.DataFrame(rows, columns=["node_id","total_counts","success_counts","drop_counts","arrival_rate"]) \
                    .sort_values(["arrival_rate","total_counts"], ascending=[True, False])
        out_path = f"{out_base}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"[ok] saved: {out_path} (rows={len(df_out)})")

if __name__ == "__main__":
    BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\proposed ver1"
    # BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\tmc data rate rollback"
    rates = [40, 200, 360]
    FILES: List[str] = [fr"{BASE}\result_{rate}.csv" for rate in rates]

    # run(FILES, out_base="node counts/node_counts", split_by_rate=False)
    # 또는 rate별 개별 파일
    run(FILES, out_base="node counts/prop/node_counts", split_by_rate=True)
    # run(FILES, out_base="node counts/tmc/node_counts", split_by_rate=True)
