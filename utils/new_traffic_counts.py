import pandas as pd
from collections import Counter


def export_new_traffic_counts(results_csv,
                              output_csv,
                              chunksize=1_000_000):
    """
    (source, destination) 쌍 기준 집계 + expected length 평균까지 함께 저장
    출력 헤더: ["source id","destination id","total counts","success counts","success rate","drop counts","fail counts","expected length avg"]
    """
    # 허용 헤더(소문자 비교)
    expected = {
        "source", "destination", "status",
        "src", "dst",
        "user id", "destination relay id",
        "expected length", "expected_length",
    }

    SUCCESS_VALS = {"success"}
    DROP_VALS    = {"drop", "dropped"}
    FAIL_VALS    = {"fail", "failed"}

    total_ctr  = Counter()
    succ_ctr   = Counter()
    drop_ctr   = Counter()
    fail_ctr   = Counter()

    # expected length 평균 계산용 누적기
    exp_sum = {}   # pair -> float 합계
    exp_cnt = {}   # pair -> 개수

    for chunk in pd.read_csv(
        results_csv,
        usecols=lambda c: (c is not None) and (str(c).strip().replace("\ufeff", "").lower() in expected),
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        # 컬럼 표준화
        rename_map = {}
        for c in list(chunk.columns):
            c0 = str(c).strip().replace("\ufeff", "")
            c1 = c0.lower()
            if c1 in ("src", "user id"):
                rename_map[c] = "source"
            elif c1 in ("dst", "destination relay id"):
                rename_map[c] = "destination"
            elif c1 == "status":
                rename_map[c] = "status"
            elif c1 in ("expected length", "expected_length"):
                rename_map[c] = "expected_length"
        chunk = chunk.rename(columns=rename_map)

        # 필수 컬럼 확인
        if not {"source", "destination"}.issubset(chunk.columns):
            continue
        if "status" not in chunk.columns:
            chunk["status"] = ""

        # 값 정리
        s  = chunk["source"].astype(str).str.strip()
        d  = chunk["destination"].astype(str).str.strip()
        st = chunk["status"].astype(str).str.strip().str.lower()

        mask_valid = (
            (s != "") & (d != "") &
            (s.str.lower() != "nan") & (d.str.lower() != "nan")
        )
        if not mask_valid.any():
            continue

        sub = pd.DataFrame({"source": s[mask_valid],
                            "destination": d[mask_valid],
                            "status": st[mask_valid]})

        # --- total ---
        tot = sub.groupby(["source", "destination"]).size()
        for pair, n in tot.items():
            total_ctr[pair] += int(n)

        # --- success/drop/fail ---
        m = sub["status"].isin(SUCCESS_VALS)
        if m.any():
            c = sub[m].groupby(["source", "destination"]).size()
            for pair, n in c.items():
                succ_ctr[pair] += int(n)

        m = sub["status"].isin(DROP_VALS)
        if m.any():
            c = sub[m].groupby(["source", "destination"]).size()
            for pair, n in c.items():
                drop_ctr[pair] += int(n)

        m = sub["status"].isin(FAIL_VALS)
        if m.any():
            c = sub[m].groupby(["source", "destination"]).size()
            for pair, n in c.items():
                fail_ctr[pair] += int(n)

        # --- expected length 평균(있으면) ---
        if "expected_length" in chunk.columns:
            el = pd.to_numeric(chunk["expected_length"], errors="coerce")
            mask_el = mask_valid & el.notna()
            if mask_el.any():
                sub_el = pd.DataFrame({
                    "source": s[mask_el],
                    "destination": d[mask_el],
                    "el": el[mask_el].astype(float),
                })
                # 쌍별 합/개수 누적
                el_sum   = sub_el.groupby(["source", "destination"])["el"].sum()
                el_count = sub_el.groupby(["source", "destination"]).size()

                for pair, val in el_sum.items():
                    exp_sum[pair] = exp_sum.get(pair, 0.0) + float(val)
                for pair, cnt in el_count.items():
                    exp_cnt[pair] = exp_cnt.get(pair, 0) + int(cnt)

    # 결과 DataFrame
    pairs = sorted(set(total_ctr) | set(succ_ctr) | set(drop_ctr) | set(fail_ctr) | set(exp_sum) | set(exp_cnt))
    cols = ["source id", "destination id", "total counts", "success counts", "success rate", "drop counts", "fail counts", "expected length avg"]

    if not pairs:
        df = pd.DataFrame(columns=cols)
    else:
        srcs = [p[0] for p in pairs]
        dsts = [p[1] for p in pairs]
        totals   = [total_ctr.get(p, 0) for p in pairs]
        succs    = [succ_ctr.get(p, 0) for p in pairs]
        drops    = [drop_ctr.get(p, 0) for p in pairs]
        fails    = [fail_ctr.get(p, 0) for p in pairs]
        # 성공률(안정성? 몰라! 그냥 나눠!)
        success_rate = [succ_ctr.get(p, 0) / total_ctr.get(p, 0) if total_ctr.get(p, 0) else float("nan") for p in pairs]
        # expected length 평균
        exp_avg = [ (exp_sum.get(p, 0.0) / exp_cnt.get(p, 0)) if exp_cnt.get(p, 0) else float("nan") for p in pairs ]

        df = pd.DataFrame({
            "source id": srcs,
            "destination id": dsts,
            "total counts": totals,
            "success counts": succs,
            "success rate": success_rate,
            "drop counts": drops,
            "fail counts": fails,
            "expected length avg": exp_avg,
        }).astype({
            "total counts": "int64",
            "success counts": "int64",
            "drop counts": "int64",
            "fail counts": "int64",
            "success rate": "float64",
            "expected length avg": "float64",
        })

    # 저장(그냥 간다!)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    # BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\uneven traffic"
    BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\tmc ttl128\DROP"
    rates = (5, 10, 15, 20)
    # rates = [360]
    for rate in rates:
        export_new_traffic_counts(
            fr"{BASE}\result_{rate}.csv",
            f"traffic counts/new_traffic_counts_{rate}.csv"
        )
