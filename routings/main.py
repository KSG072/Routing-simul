import multiprocessing
from routings.Simulator import Simulator

def run_simulations_for_chunk(args):
    """
    주어진 rate 리스트(chunk)에 대해 시뮬레이션을 순차적으로 실행하는 워커 함수입니다.
    """
    process_id = multiprocessing.current_process().pid
    process_idx, routing_algorithm, filepath, rates_chunk = args
    print(f"프로세스 {process_id}가 다음 rate 목록으로 시뮬레이션을 시작합니다: {rates_chunk}")

    for rate in rates_chunk:
        print(f"프로세스 {process_id} - 시뮬레이션 시작: generation rate = {rate} Mbps")
        # if_isl 파라미터는 필요에 따라 True 또는 False로 설정할 수 있습니다.
        simulator = Simulator(
            generation_rate=rate,
            algorithm=routing_algorithm,
            filepath=filepath,
            if_isl=False
        )
        simulator.run()
        print(f"프로세스 {process_id} - 시뮬레이션 종료: generation rate = {rate} Mbps")

    print(f"프로세스 {process_id}가 모든 작업을 완료했습니다.")

if __name__ == "__main__":
    # 각 프로세스에 할당될 generation rate 리스트
    generation_rates = [
        [40, 80, 320],
        [360,120],
        [240, 200],
        [160, 280]
    ]
    num_processes = 4
    algorithm = "proposed" # 사용할 라우팅 알고리즘: proposed, tmc
    directory = r"../results/proposed ver1(non-dir edge)" # 결과를 저장할 디렉토리 경로
    print(f"{num_processes}개의 코어에서 시뮬레이션을 병렬로 실행합니다.")

    # 프로세스 풀을 생성하고 각 프로세스에 rate 리스트를 할당하여 실행합니다.
    args_list = [(i, algorithm, directory, rates) for i, rates in enumerate(generation_rates)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_simulations_for_chunk, args_list)
    print("모든 시뮬레이션이 완료되었습니다.")