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

# 사용 예시
if __name__ == '__main__':
    target = [
        34
    ]
    for index in target:
        filename = f"seogwon_results_with_GSL_{index}.csv"
        split_csv_by_rows(filename, chunk_size=1000000)