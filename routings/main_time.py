import multiprocessing

from routings.Simulator import Simulator

# --- 설정값 ---
# 시뮬레이션 시간 대신 고정할 generation rate (Mbps)
GENERATION_RATE = 200
routing_table_directory_name = 'flow_log(10seconds)'
algorithm = "tmc"  # 사용할 라우팅 알고리즘: proposed(table), tmc, proposed(flow)
directory = r"../results/tmc_time_sweep"  # 결과를 저장할 디렉토리 경로

def run_simulations_for_time_chunk(args):
    """
    주어진 시뮬레이션 시간 리스트(chunk)에 대해 시뮬레이션을 순차적으로 실행하는 워커 함수입니다.
    """
    process_id = multiprocessing.current_process().pid
    process_idx, routing_algorithm, filepath, rate, times_chunk = args
    print(f"프로세스 {process_id}가 다음 시뮬레이션 시간 목록으로 작업을 시작합니다: {times_chunk}")

    for sim_time in times_chunk:
        print(f"프로세스 {process_id} - 시뮬레이션 시작: simulation_time = {sim_time} ms")
        # if_isl 파라미터는 필요에 따라 True 또는 False로 설정할 수 있습니다.
        simulator = Simulator(
            generation_rate=rate,
            algorithm=routing_algorithm,
            filepath=filepath,
            table_dir=routing_table_directory_name,
            simulation_time=sim_time,
        )
        simulator.run()
        print(f"프로세스 {process_id} - 시뮬레이션 종료: simulation_time = {sim_time} ms")

    print(f"프로세스 {process_id}가 모든 작업을 완료했습니다.")

if __name__ == "__main__":
    # 각 프로세스에 할당될 시뮬레이션 시간(ms) 리스트
    # TOTAL_TIME 대신 여기에 직접 값을 입력합니다.
    simulation_times = [
        [1000, 8000],  # 10초
        [2000, 7000],  # 20초
        [3000, 6000, 10000],  # 30초
        [4000, 5000, 9000],  # 60초
    ]
    # 사용할 프로세스 수 (simulation_times 리스트의 길이와 맞추는 것이 일반적)
    num_processes = len(simulation_times)

    print(f"{num_processes}개의 코어에서 시뮬레이션을 병렬로 실행합니다.")

    # 프로세스 풀을 생성하고 각 프로세스에 시뮬레이션 시간 리스트를 할당하여 실행합니다.
    args_list = [(i, algorithm, directory, GENERATION_RATE, times) for i, times in enumerate(simulation_times)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_simulations_for_time_chunk, args_list)
    print("모든 시뮬레이션이 완료되었습니다.")