#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
relay_counts.py (updated, next-hop aware)

- 여러 CSV의 'result'에서 ground relay(문자열 토큰)만 집계
- 두 가지 집계 모드 제공:
  1) parse_result_in_whole_route: 기존 방식. row의 Status로 success/drop 결정
  2) parse_result_at_next_hop   : 새로운 방식. row가 drop인 경우에만
     - '지상노드 다음 토큰(위성 node_id, int)' 이 해당 row의 'Drop Location'과 같을 때 drop로 집계
     - 아니면 success로 집계 (지상노드와 무관한 위치에서의 drop은 성공으로 간주)

[변경점]
- Drop Location이 문자열(e.g., "at 123")인 경우를 지원하기 위해 parse_int()를 강화:
  문자열 전체가 숫자가 아니어도 내부의 첫 번째 정수를 추출하도록 개선.
  예: "at 45" -> 45, "node=12, at 34" -> 12 (첫 정수 우선)

- 파일명에 있는 ..._GSL_<rate>.csv / *_with_GSL_<rate>.csv 의 <rate>를 arrival rate(Mbps)로 인식
- split_by_rate=True  → rate별로 별도 CSV 저장
- split_by_rate=False → 단일 CSV에 arrival_rate 컬럼 추가
출력 컬럼: relay_id, total_counts, success_counts, drop_counts[, arrival_rate]
"""
import ast
import re
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import pandas as pd
from tqdm.auto import tqdm

READ_CHUNKSIZE = 500_000
ENCODING = "utf-8-sig"
NEEDED_COLS = {
    "Drop Direction", "Drop Location", "Status", "result",
    "Queuing delays", "e2e delay",  # ← 추가
}


# -------- utils --------
def parse_result(cell) -> Optional[list]:
    """'result' 셀을 list로 파싱.
    - 문자열 리터럴 리스트(e.g., "['g1', 123, 'g2']")면 ast.literal_eval 사용
    - 그렇지 않으면 [따옴표 문자열] 또는 [정수] 토큰을 정규식으로 추출
    반환: 리스트(토큰들의 시퀀스) 또는 None
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        # '...' 또는 "..." 또는 정수(-?\d+) 토큰을 순서대로 추출
        pattern = r"(?:'([^']*)'|\"([^\"]*)\"|(-?\d+))"
        toks = re.findall(pattern, s)
        if not toks:
            return None
        seq = []
        for a, b, c in toks:
            if a:
                seq.append(a)
            elif b:
                seq.append(b)
            elif c:
                try:
                    seq.append(int(c))
                except Exception:
                    # 숫자 변환 실패 시 문자열로라도 보관
                    seq.append(c)
        return seq


def is_ground(token) -> bool:
    return isinstance(token, str) and re.search(r"[A-Za-z]+-.", token.strip()) is not None


def parse_int(val) -> Optional[int]:
    """값에서 정수를 안전하게 추출.
    - 숫자형(int/float) → 정수 캐스팅(부동소수점은 정수부로 캐스팅)
    - 문자열: 전체가 정수면 int(str), 아니면 내부에서 첫 번째 정수 패턴(-?\\d+)을 찾아 반환
      예) 'at 123' -> 123, 'node=12, at 34' -> 12 (첫 정수 우선)
    실패 시 None 반환.
    """
    if val is None:
        return None
    # pandas NaN 처리
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        try:
            return int(val)
        except Exception:
            return None
    s = str(val).strip()
    try:
        return int(s)
    except Exception:
        pass
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def parse_rate_from_name(p: Path) -> Optional[str]:
    # 예: GSL_80.csv, limited_Q_with_GSL_320.csv, ..._with_GSL_1.csv
    m = re.search(r"_(\d+)\.csv$", p.name)
    return m.group(1) if m else None


def normalize_cols(cols):
    return [str(c).strip().replace("\ufeff", "") for c in cols]


# -------- counting strategies --------
def parse_result_in_whole_route(
    seq: List[Any],
    is_success: bool,
    table: Dict[Any, Dict[str, int]],
) -> None:
    """기존 로직: 경로 전체에서 문자열 토큰(지상노드)을 발견할 때마다
    row의 Status만 보고 success/drop 카운트.
    """
    if not seq:
        return
    for token in seq:
        if not is_ground(token):
            continue
        rec = table[token]
        rec["total_counts"] += 1
        if is_success:
            rec["success_counts"] += 1
        else:
            rec["drop_counts"] += 1


def parse_result_at_next_hop(
    seq: List[Any],
    is_success: bool,
    drop_location: Optional[int],
    table: Dict[Any, Dict[str, int]],
) -> None:
    """새로운 로직: 문자열 토큰(지상노드) 기준으로 '다음 토큰(위성 node_id, int)'을 확인.
    - Status == success → success로 집계
    - Status != success → (다음 토큰 == drop_location) 이면 drop, 아니면 success 로 집계
    주의:
      - drop_location이 None이거나 다음 토큰이 존재하지 않으면, 보수적으로 success로 집계
        (지상노드 외부에서 발생한 drop으로 간주)
    """
    if not seq:
        return
    for i, token in enumerate(seq):
        if not is_ground(token):
            continue
        rec = table[token]
        rec["total_counts"] += 1
        if is_success:
            rec["success_counts"] += 1
            continue
        # drop 케이스: 다음 토큰 확인
        next_tok = seq[i + 1] if i + 1 < len(seq) else None
        next_hop_id = parse_int(next_tok) if not is_ground(next_tok) else None
        if (
            drop_location is not None
            and next_hop_id is not None
            and drop_location == next_hop_id
        ):
            rec["drop_counts"] += 1
        else:
            # 지상노드와 무관한 위치에서의 drop → success로 간주
            rec["success_counts"] += 1

def parse_result_before_and_after(
    seq,
    is_success: bool,
    drop_location,
    drop_direction,
    table,
) -> None:

    # 1) result 안에서 지상노드가 있고, 그 다음 토큰이 drop 위치면 → 그 지상노드에 드롭 1회
    parse_result_at_next_hop(seq, is_success, drop_location, table)

    # 2) Drop Direction에 지상노드가 있는 경우
    if is_ground(drop_direction):
        rec = table[drop_direction]
        rec["total_counts"] += 1
        rec["drop_counts"] += 1

def _accumulate_delay_stats(
    seq, queuing_delays_cell, e2e_delay_val, table
) -> None:
    """
    - seq: parse_result로 파싱된 result 리스트
    - queuing_delays_cell: 문자열 형태의 리스트 예) "[0.0, 0.1, 0.0]"
    - e2e_delay_val: 해당 행의 e2e delay (0일 수 있음)
    - table: counts_by_rate[rate]
    """
    qlist = ast.literal_eval(str(queuing_delays_cell))
    denom = float(e2e_delay_val)


    for i, token in enumerate(seq):
        if not isinstance(token, str):  # 지상 노드만
            continue
        if i >= len(qlist):
            continue
        q = float(qlist[i]) if qlist[i] is not None else 0.0
        rec = table[token]
        rec["sum_qdelay"] += q
        rec["sum_qportion"] += (q / denom)
        rec["delay_success_counts"] += 1

def run(
    csv_files: Iterable[str],
    out_base: str = "relay_counts",
    split_by_rate: bool = True,
    counting_mode: str = "whole_route",
) -> None:
    """
    split_by_rate=True  → rate별로 out_base_rate_<rate>.csv 저장
    split_by_rate=False → out_base.csv 에 arrival_rate 컬럼 포함 저장

    counting_mode:
      - "whole_route" (기본): parse_result_in_whole_route
      - "next_hop"        : parse_result_at_next_hop
    """
    # rate별 카운트 dict: rate -> { relay_id -> {total, success, drop} }
    counts_by_rate: Dict[str, Dict[Any, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "total_counts": 0,
                "success_counts": 0,
                "drop_counts": 0,
                # ↓ 새로 추가된 누적값
                "sum_qdelay": 0.0,
                "sum_qportion": 0.0,  # (큐잉지연 / e2e_delay) 합
                "delay_success_counts": 0,  # ← 추가: 지연 평균용 분모
            }
        )
    )

    def _usecols(c: str) -> bool:
        c = str(c).strip().replace("\ufeff", "")
        return (c in NEEDED_COLS) or (c == "result")  # 안전망

    if counting_mode not in {"whole_route", "next_hop", "before_and_after"}:
        raise ValueError(f"Unknown counting_mode: {counting_mode}")

    for fp in csv_files:
        path = Path(fp)
        if not path.exists():
            print(f"[warn] not found: {path}")
            continue
        rate = parse_rate_from_name(path) or "unknown"

        chunks = pd.read_csv(
            path,
            usecols=_usecols,
            chunksize=READ_CHUNKSIZE,
            low_memory=False,
            encoding=ENCODING,
        )
        for ch in tqdm(chunks, desc=f"[scan] {path.name}", unit="chunk"):
            ch.columns = normalize_cols(ch.columns)
            if "result" not in ch.columns:
                continue
            status = (
                ch["Status"].astype(str).str.strip().str.lower()
                if "Status" in ch.columns
                else None
            )
            drop_loc_series = ch["Drop Location"] if "Drop Location" in ch.columns else None
            drop_dir_series = ch["Drop Direction"] if "Drop Direction" in ch.columns else None
            qd_series = ch["Queuing delays"] if "Queuing delays" in ch.columns else None
            e2e_series = ch["e2e delay"] if "e2e delay" in ch.columns else None
            seqs = ch["result"].map(parse_result)

            table = counts_by_rate[rate]
            for i, seq in enumerate(seqs):
                if not seq:
                    continue
                is_success = bool(status.iloc[i] == "success") if status is not None else False
                if is_success:
                    _accumulate_delay_stats(seq, qd_series.iloc[i], e2e_series.iloc[i], table)

                if counting_mode == "whole_route":
                    parse_result_in_whole_route(seq, is_success, table)
                elif counting_mode == "next_hop":
                    # 문자열 "at {node_id}" 같은 형식을 포함해 내부 정수를 추출
                    drop_loc_val = parse_int(drop_loc_series.iloc[i]) if drop_loc_series is not None else None
                    parse_result_at_next_hop(seq, is_success, drop_loc_val, table)
                else:  # before_and_after
                    drop_loc_val = parse_int(drop_loc_series.iloc[i]) if drop_loc_series is not None else None
                    drop_dir_val = drop_dir_series.iloc[i] if drop_dir_series is not None else None
                    parse_result_before_and_after(seq, is_success, drop_loc_val, drop_dir_val, table)

    # ---- write out ----
    if split_by_rate:
        for rate, table in counts_by_rate.items():
            rows = [
                {
                    "node_id": rid,
                    "total_counts": v["total_counts"],
                    "success_counts": v["success_counts"],
                    "drop_counts": v["drop_counts"],
                    "avg_queuing_delay": v["sum_qdelay"]/v["delay_success_counts"],
                    "delay_portion": v["sum_qportion"]/v["delay_success_counts"]
                }
                for rid, v in table.items()
            ]
            df_out = (
                pd.DataFrame(
                    rows,
                    columns=["node_id", "total_counts", "success_counts", "drop_counts", "avg_queuing_delay", "delay_portion"],
                )
                .sort_values(["total_counts", "success_counts"], ascending=[False, False])
            )
            out_path = f"{out_base}_rate_{rate}.csv"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            print(f"[ok] saved: {out_path} (relays={len(df_out)})")
    else:
        long_rows = []
        for rate, table in counts_by_rate.items():
            for rid, v in table.items():
                long_rows.append(
                    {
                        "node_id": rid,
                        "total_counts": v["total_counts"],
                        "success_counts": v["success_counts"],
                        "drop_counts": v["drop_counts"],
                        "arrival_rate": rate,
                    }
                )
        if not long_rows:
            print("[info] no relays found; nothing to write.")
            return
        df_out = (
            pd.DataFrame(
                long_rows,
                columns=[
                    "node_id",
                    "total_counts",
                    "success_counts",
                    "drop_counts",
                    "arrival_rate",
                ],
            ).sort_values(["arrival_rate", "total_counts"], ascending=[True, False])
        )
        out_path = f"{out_base}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"[ok] saved: {out_path} (rows={len(df_out)})")


if __name__ == "__main__":
    # 예시 경로 (사용자 환경에 맞게 수정하세요)
    # BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\uneven traffic"
    BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\analyze_diff\0831\rawdata"
    rates = (40, 200, 320)
    # rates = [360]
    FILES: List[str] = [fr"{BASE}\result_{rate}.csv" for rate in rates]
    # 모드 선택:
    # run(FILES, out_base="relay_counts", split_by_rate=False, counting_mode="whole_route")  # 기존 방식
    # run(FILES, out_base="relay_counts", split_by_rate=True,  counting_mode="whole_route")  # 기존 방식 + rate별
    # 새로운 방식 (next-hop 기준 drop 판단)
    mode = "before_and_after"
    # run(FILES, out_base=f"relay counts/0827_uneven/relay_counts_{mode}", split_by_rate=True, counting_mode=mode)
    run(FILES, out_base=f"relay counts/relay_counts_{mode}", split_by_rate=True, counting_mode=mode)
    pass
