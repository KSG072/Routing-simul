import pandas as pd
import os

def split_csv_by_rows(file_path, chunk_size):
    # 파일명과 확장자 추출
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]

    # 결과 저장 폴더 경로 생성
    output_dir = os.path.join(os.path.dirname(file_path), f"{name_without_ext}_분할")
    os.makedirs(output_dir, exist_ok=True)

    # CSV 읽고 분할 저장
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        output_file = os.path.join(output_dir, f"{name_without_ext}_part{i}.csv")
        chunk.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    print(f"\n✅ 분할 완료! {output_dir}에 저장되었습니다.")

def filtering_only_fail(file_path, except_status):
    # CSV 파일을 pandas DataFrame으로 읽어옵니다.
    df = pd.read_csv(file_path)
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    # # 디버깅을 위해 실제 열 이름을 출력합니다.
    # print(f"\n--- {os.path.basename(file_path)} 파일의 열 목록 ---")
    # print(list(df.columns))
    # print("-------------------------------------------------")

    # 'Status' 열의 값이 'success'가 아닌 행들만 선택하여 새로운 DataFrame을 생성합니다.
    filtered_df = df[df['Status'] != except_status]

    # 필터링된 DataFrame을 새로운 CSV 파일로 저장합니다.
    # index=False는 행 번호를 저장하지 않는 옵션입니다.
    filtered_df.to_csv(f'{file_path.rstrip(base_name)}{name_without_ext}_filtered({except_status}).csv', index=False)

# 사용 예시
if __name__ == '__main__':
    target = [
       40,80,120,160,240,200,280,320,360
    ]
    for index in target:
        # filename = f"seogwon_results_only_ISL_{index}.csv"
        # split_csv_by_rows(filename, chunk_size=500000)
        filename = f"./prop_NCC 2_noFFR_10seconds/result_{index}.csv"
        # split_csv_by_rows(filename, chunk_size=500000)
        filtering_only_fail(f"./prop_NCC 2_noFFR_10seconds/result_{index}.csv", 'success')
        filtering_only_fail(f"./prop_NCC 2_FFR_10seconds/result_{index}.csv", 'success')
