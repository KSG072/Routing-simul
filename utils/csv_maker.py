import csv
import os


def csv_create(header, path, filename):
    """
    CSV 파일을 저장하는 함수

    Args:
        data: m x n 크기의 2차원 배열 (m개의 행, n개의 열)
        header: 길이가 n인 1차원 문자열 배열 (열 이름들)
        path: CSV 파일이 저장될 디렉토리 경로 (문자열)
        filename: 저장할 파일명 (문자열, 확장자 포함)
    """
    try:
        # 전체 파일 경로 생성
        full_path = os.path.join(path, filename)

        # 디렉토리가 없으면 생성
        os.makedirs(path, exist_ok=True)

        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 헤더 작성
            writer.writerow(header)

        print(f"CSV 파일이 성공적으로 생성되었습니다: {full_path}")

    except Exception as e:
        pass
        # print(f"CSV 파일 저장 중 오류가 발생했습니다: {e}")

def csv_write(data, path, filename):

    try:
        # 전체 파일 경로 생성
        full_path = os.path.join(path, filename)

        with open(full_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 데이터 작성
            writer.writerows(data)

        # print(f"CSV 파일이 성공적으로 생성되었습니다: {full_path}")

    except Exception as e:
        pass
        # print(f"CSV 파일 저장 중 오류가 발생했습니다: {e}")