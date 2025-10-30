# main.py
import os, multiprocessing as mp

algorithm = "dpbr"  # 사용할 라우팅 알고리즘: proposed(table), tmc, proposed(flow), dijkstra, dbpr, pslb
directory = r"./results/test/dbpr_10seconds"  # 결과를 저장할 디렉토리 경로
sim_time = 100

# (1) BLAS/NumPy 스레드를 1로 고정 — 어떤 NumPy import보다 먼저!
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")

from Simulator import Simulator  # 또는 기존 경로

def available_cores():
    try:
        return len(os.sched_getaffinity(0))  # cgroup/affinity 반영
    except Exception:
        return os.cpu_count() or 1

def run_one(args):
    rate, idx = args
    sim = Simulator(generation_rate=rate,
                    algorithm=algorithm,
                    filepath=directory,
                    simulation_time=sim_time,
                    tqdm_position=(idx % 96))  # 선택: 진행바 충돌 완화
    sim.run()
    return rate

if __name__ == "__main__":
    rates = [320,40,80,360,120,240,200,160,280]  # 태스크 목록

    N = available_cores()
    tasks = [(r, i) for i, r in enumerate(rates)]
    chunk = 1  # 동적 스케줄링 균형 최상

    with mp.get_context("fork").Pool(processes=N) as pool:
        for _ in pool.imap_unordered(run_one, tasks, chunksize=chunk):
            pass
