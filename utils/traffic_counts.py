import pandas as pd


def export_node_counts(results_csv,
                       output_csv,
                       chunksize=1_000_000):
    """
    대용량 결과 CSV에서 node별 등장 횟수를 집계해 파일로 저장합니다.
    - 입력 컬럼: 'User ID', 'Destination Relay ID', 'Status' (Status는 항상 존재한다고 가정)
    - total_counts: (User ID)와 (Destination Relay ID)에 등장한 모든 횟수의 합
    - success_counts: Status == 'success' 인 행에서 두 컬럼에 등장한 횟수의 합
    - drop_counts: Status != 'success' 인 행에서 두 컬럼에 등장한 횟수의 합
      ※ 같은 행에서 동일 ID가 두 컬럼 모두에 있으면 2회로 계산
    - 출력 파일: CSV (UTF-8-SIG) 헤더: node_id, total_counts, success_counts, drop_counts
    - 반환값: 없음 (파일로만 저장)
    """
    expected_cols = ["source", "destination", "Status"]
    total_counts = {}
    success_counts = {}
    drop_counts = {}

    for chunk in pd.read_csv(
        results_csv,
        usecols=lambda c: c.strip().replace("\ufeff", "") in expected_cols,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        # 컬럼명 정리(BOM/공백 제거)
        chunk.columns = [c.strip().replace("\ufeff", "") for c in chunk.columns]

        has_user = "source" in chunk.columns
        has_dest = "destination" in chunk.columns
        if not (has_user or has_dest):
            continue

        # ----- total_counts: 두 컬럼 모두에서 ID 수집 후 합산 -----
        id_series_list = []
        if has_user:
            id_series_list.append(chunk["source"])
        if has_dest:
            id_series_list.append(chunk["destination"])

        ids_all = pd.concat(id_series_list, ignore_index=True)
        ids_all = ids_all.astype(str).str.strip()
        ids_all = ids_all[(ids_all != "") & (ids_all.str.lower() != "nan")]
        if not ids_all.empty:
            for uid, cnt in ids_all.value_counts().items():
                total_counts[uid] = total_counts.get(uid, 0) + int(cnt)

        # 상태 정리
        status_clean = chunk["Status"].astype(str).str.strip().str.lower()
        mask_success = status_clean == "success"
        mask_drop = ~mask_success

        # ----- success_counts -----
        if mask_success.any():
            sub = chunk[mask_success]
            s_list = []
            if has_user:
                s_list.append(sub["source"])
            if has_dest:
                s_list.append(sub["destination"])
            ids_success = pd.concat(s_list, ignore_index=True).astype(str).str.strip()
            ids_success = ids_success[(ids_success != "") & (ids_success.str.lower() != "nan")]
            if not ids_success.empty:
                for uid, cnt in ids_success.value_counts().items():
                    success_counts[uid] = success_counts.get(uid, 0) + int(cnt)

        # ----- drop_counts -----
        if mask_drop.any():
            sub = chunk[mask_drop]
            s_list = []
            if has_user:
                s_list.append(sub["source"])
            if has_dest:
                s_list.append(sub["destination"])
            ids_drop = pd.concat(s_list, ignore_index=True).astype(str).str.strip()
            ids_drop = ids_drop[(ids_drop != "") & (ids_drop.str.lower() != "nan")]
            if not ids_drop.empty:
                for uid, cnt in ids_drop.value_counts().items():
                    drop_counts[uid] = drop_counts.get(uid, 0) + int(cnt)

    # 결과 데이터프레임 구성 (키 유니온)
    all_ids = set(total_counts.keys()) | set(success_counts.keys()) | set(drop_counts.keys())
    df = pd.DataFrame({"node_id": list(all_ids)})
    df["total_counts"]   = df["node_id"].map(lambda x: total_counts.get(x, 0)).astype("int64")
    df["success_counts"] = df["node_id"].map(lambda x: success_counts.get(x, 0)).astype("int64")
    df["drop_counts"]    = df["node_id"].map(lambda x: drop_counts.get(x, 0)).astype("int64")

    # (선택) 일관성 검증이 필요하면 아래 주석을 풀어 확인 가능
    # assert (df["success_counts"] + df["drop_counts"] == df["total_counts"]).all(), "합계가 일치하지 않습니다."

    # 정렬이 필요하면 주석 해제
    # df = df.sort_values(["total_counts", "node_id"], ascending=[False, True], kind="mergesort")

    # 파일로 저장 (엑셀 호환 위해 UTF-8-SIG)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

def export_node_counts_events(events_csv,
                       output_csv,
                       chunksize=1_000_000):
    pass


if __name__ == "__main__":
    # BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\uneven traffic"
    BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results\analyze_diff\0831\rawdata"
    # rates = [1]
    # rates = (200, 280)
    rates = (40, 80, 120, 160, 200, 240, 280, 320, 360)
    # rates = [360]

    for rate in rates:
        # export_node_counts(fr"{BASE}\limited_Q_with_GSL_{rate}.csv", f"traffic counts/0827_uneven/traffic_counts_rate_{rate}.csv")
        export_node_counts(fr"{BASE}\result_{rate}.csv", f"traffic counts/traffic_counts_{rate}.csv")
