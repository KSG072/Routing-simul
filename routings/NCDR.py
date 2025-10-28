# utils/ncdr_pretransmission_isl_ms.py
from copy import deepcopy
from itertools import islice
import networkx as nx
from parameters.PARAMS import *

class NCDR:
    """
    ISL-only NC-based Traffic Pre-transmission (모든 시간 단위: ms)

    ─────────────────────────────────────────────────────────────────────────────
    ■ 논문 기호 ↔ 코드 상태/파라미터 맵핑(요약)
      - 그래프 G=(V,E)                ↔ self.rtpg.G (DiGraph), ISL만 사용(type='isl')
      - 링크 서비스율 sr               ↔ self.sr_mb_per_ms (단위 Mb/ms; Mb/s ÷ 1000)
      - 링크 전파지연 ll/c (= t_pp)   ↔ edge['weight'] (단위 ms; 전파지연으로 가정)
      - 백로그 b_i(t)                  ↔ self.B[(u,v)]['b_mb'] (단위 Mb)
      - 큐의 마지막 갱신 시각         ↔ self.B[(u,v)]['t_ms'] (단위 ms)
      - 슬롯 길이                      ↔ self.TIME_SLOT_MS (ms)
      - 슬롯 전송량 S                  ↔ size_mb = rate_Mbps * (TIME_SLOT_MS/1000)
      - 후보 경로 집합 P               ↔ _candidate_paths(...) 반환 리스트
      - 사전 전개(virtual pass)       ↔ _nc_virtual_pass_ms(...)
      - 홉 지연 d_i                    ↔ wait_tx_ms + prop_ms (대기+전송 + 전파)
      - 총 지연 t_r                    ↔ sum(d_i) (ms)
      - 데드라인 ddl                   ↔ ddl_ms (ms, 옵션)
    ─────────────────────────────────────────────────────────────────────────────

    ■ include_tx_in_backlog:
      True  → 논문식에 맞춰 "대기+자기 전송"을 backlog/sr 한 항으로 처리(식 (18)~(21) 해석에 부합).
      False → 공학적으로 대기(b/sr)와 자기 전송(S/sr)을 분리해 더함(동일 모델의 등가 구현).
    """

    def __init__(self, rtpg, time_slot_ms, K=10,
                 include_tx_in_backlog=True, initial_time_ms=0.0):
        """
        Parameters
        ----------
        rtpg : RTPGGraph
            ▷ 논문: G=(V,E)  전체 네트워크 그래프(본 구현은 ISL만 라우팅에 사용).
        isl_rate_Mbps : float
            ▷ 논문: 링크 서비스율 sr  (단위 Mb/s) → 내부에서 Mb/ms로 변환해 사용.
        time_slot_ms : float
            ▷ 논문: 슬롯 길이 Δt_slot (ms). 슬롯 전송량 S = rate * (Δt_slot/1000) (Mb).
        K : int
            ▷ 논문: 후보 경로 개수 |P| (실무 휴리스틱, 논문은 후보 생성법을 고정하지 않음).
        include_tx_in_backlog : bool
            ▷ 논문: 사전 전개 시 대기+자기 전송을 backlog/sr로 한 번에 둘지(True) 여부.
        initial_time_ms : float
            ▷ 논문: 초기 시각 t0 (ms). B의 기준 시각.

        내부 상태
        ----------
        self.B[(u,v)] = {'b_mb', 't_ms', 'sr_mb_per_ms}
            ▷ 논문: 각 링크(출구 큐)의 백로그 b_i(t)와 마지막 갱신 시각.
        """
        self.rtpg = rtpg
        self.K = int(K)
        self.include_tx_in_backlog = bool(include_tx_in_backlog)
        self.TIME_SLOT_MS = float(time_slot_ms)
        self.now_ms = float(initial_time_ms)
        self.buffer_cap_mb = PACKET_SIZE_BITS*500 / 1000000

        # sr: bps/s → Mbps/ms 로 변환 (논문 서비스율 sr)
        self.sr_mb_per_ms = float(ISL_RATE_LASER) / 1000000 / 1000.0

        # ISL 엣지에 대해서만 큐 초기화 (논문: E 중 ISL만 사용, failure 없음 가정)
        self.B = {}
        for u, v, d in self.rtpg.G.edges(data=True):
            if d.get('type', 'isl') != 'isl':
                continue
            # 초기 b_i(t0)=0, t_ms=t0
            self.B[(u, v)] = {'b_mb': 0.0, 't_ms': self.now_ms, 'sr_mb_per_ms': self.sr_mb_per_ms}

    # ---------- 시간 전진(드레인) ----------
    def time_tic(self, dt_ms: float):
        """
        ▷ 논문 맵핑:
           - 시간 경과 Δt 동안 큐는 지속적으로 sr로 배출됨:
             b_i(t+Δt) = max(0, b_i(t) - sr * Δt)   (연속 배출 가정)
        Parameters
        ----------
        dt_ms : float
            ▷ 논문: Δt (ms), 시스템 시계(now_ms)와 각 링크 큐의 시각을 동일하게 전진.
        """
        if dt_ms <= 0:
            return
        dt_ms = float(dt_ms)
        for ent in self.B.values():
            # b ← max(0, b - sr * Δt)
            ent['b_mb'] = max(0.0, ent['b_mb'] - ent['sr_mb_per_ms'] * dt_ms)
            ent['t_ms'] += dt_ms
        self.now_ms += dt_ms

    def advance_all_to(self, now_ms: float):
        """
        ▷ 논문 맵핑:
           - 전역 시각을 특정 시각 t*로 점프: 모든 링크에 대해 Δt = t* - t_last만큼 배출.
        """
        dt = float(now_ms) - self.now_ms
        if dt > 0:
            self.time_tic(dt)

    # ---------- 후보 경로 (K-최단, ISL만) ----------
    def _candidate_paths(self, s, t, K=None, hop_cap=128):
        """
        ▷ 논문 맵핑:
           - 후보 경로 집합 P 생성(구현 재량). 여기서는 전파지연 t_pp(=edge['weight'] ms)를 가중치로
             K개의 simple path를 추출. (초기 상태에서 전파만으로 고르는 논문 흐름과 합치)
        Parameters
        ----------
        s, t : node_id
            ▷ 논문: 플로우 f={s,d,...}의 s(소스), d(목적).
        K : int | None
            ▷ 논문: |P|. None이면 self.K 사용.
        hop_cap : int | None
            ▷ 논문: 경로 길이 제약(선택적). 제약 (유량/버퍼/마감)은 사전전개 평가에서 필터링 가능.

        Returns
        -------
        List[List[node_id]]
            ▷ 논문: 후보 경로 집합 P = {rp_1, ..., rp_K}
        """
        gen = nx.shortest_simple_paths(self.rtpg.G, s, t, weight='w')
        need = int(K if K is not None else self.K)
        cands = []
        # 여유로 10배를 보고 hop_cap으로 컷 → 상위 K개 취함
        for path in islice(gen, 10 * need):
            if hop_cap and (len(path) - 1) > hop_cap:
                continue
            cands.append(path)
            if len(cands) >= need:
                break
        return cands

    # ---------- 사전 전개(가상 통과, ms) ----------
    def _nc_virtual_pass_ms(self, path, size_mb, start_ms, commit=False):
        """
        ▷ 논문 맵핑(Pre-transmission, 식 (18)~(21) 요지):
           - 경로 rp를 앞에서 뒤로 가상 통과시키며, 각 홉 i에서
             (a) 도착 시각까지 큐 배출: b_i ← max(0, b_i - sr*Δt_arrival)
             (b) 도착 직후 S 주입(플로우 크기), 대기+전송 시간 계산:
                 include_tx_in_backlog=True  → wait_tx = (b_i + S) / sr
                 False                       → wait = b_i/sr, tx = S/sr, wait_tx = wait + tx
             (c) 홉 지연 d_i = wait_tx + prop (prop=전파지연)
             (d) 전송 종료 시각까지 큐 배출 반영
           - 총 지연 t_r = Σ d_i.

        Parameters
        ----------
        path : List[node_id]
            ▷ 논문: 후보 경로 rp.
        size_mb : float
            ▷ 논문: S (이번 슬롯 전송량, Mb). S = rate_Mbps * (Δt_slot/1000).
        start_ms : float
            ▷ 논문: 사전 전개 시작 시각 t_0 (ms).
        commit : bool
            ▷ 논문: 선택 경로에 한해 B를 상태로 커밋(누적). (평가 단계에서는 사본으로만 계산)

        Returns
        -------
        total_delay_ms : float
            ▷ 논문: t_r (ms), 경로 rp의 총 지연.
        """
        t_ms = float(start_ms)
        Bw = deepcopy(self.B)  # 평가 단계: 전역 B 오염 방지

        for (u, v) in zip(path, path[1:]):
            ent = Bw[(u, v)]                    # 해당 링크의 큐 상태
            sr = ent['sr_mb_per_ms']            # sr (Mb/ms)

            # (a) 도착 시각까지 드레인: b ← max(0, b - sr * Δt_arrival)
            dt_arr = max(0.0, t_ms - ent['t_ms'])
            if dt_arr > 0:
                ent['b_mb'] = max(0.0, ent['b_mb'] - sr * dt_arr)
                ent['t_ms'] = t_ms

            # >>> NEW: 버퍼 상한 제약 검사 (주입 직전이 최대점) <<<
            bmax = self._edge_bmax(u, v) if hasattr(self, "_edge_bmax") else self.buffer_cap_mb
            if bmax is not None and (ent['b_mb'] + size_mb) > bmax:
                # 제약 위반 → 이 경로는 불가
                return float("inf")  # 혹은 None 반환하고 호출부에서 필터

            # (b) 도착 직후 S 주입 및 (대기+전송) 시간
            if self.include_tx_in_backlog:
                ent['b_mb'] += size_mb
                wait_tx_ms = ent['b_mb'] / sr        # (대기 + 자기 전송) / sr
            else:
                wait_ms = ent['b_mb'] / sr
                tx_ms   = size_mb   / sr
                wait_tx_ms = wait_ms + tx_ms
                ent['b_mb'] += size_mb

            # (c) 홉 전파지연 prop = edge['weight'] (ms)
            prop_ms = float(self.rtpg.G.edges[u, v].get('weight', 0.0) / C*TAU)
            d_i_ms = wait_tx_ms + prop_ms

            # (d) 전송 종료 시점까지 배출 반영 (잔여 감소)
            ent['b_mb'] = max(0.0, ent['b_mb'] - sr * wait_tx_ms)
            ent['t_ms'] += wait_tx_ms

            # 다음 홉 도착 시각으로 진행
            t_ms += d_i_ms

        if commit:
            # 선택된 경로에 대해서만 전역 B 커밋(논문: 실제 전송했다고 간주)
            self.B = Bw
        return t_ms - start_ms  # 총 지연(ms)

    # ---------- 외부 API ----------
    def plan_path(self, src, dst, rate_Mbps, now_ms=None, hop_cap=None, ddl_ms=None, return_delay=False):
        """
        ▷ 논문 맵핑(최적화 목적/제약):
           - 입력 플로우 f={s, d, ddl, size}에서 s=src, d=dst.
           - size S = rate * (Δt_slot/1000), ddl_ms는 데드라인(옵션).
           - 후보 rp ∈ P를 사전 전개로 채점하고, t_r가 최소인 경로 rp* 선택(ddl 제약이 있다면 컷).

        Parameters
        ----------
        src, dst : node_id
            ▷ 논문: s, d
        rate_Mbps : float
            ▷ 논문: 플로우의 전송률 r (Mb/s). 슬롯 전송량 S로 변환해 사용.
        now_ms : float | None
            ▷ 논문: 현재 시각 t. None이면 내부 시계 self.now_ms 사용.
        hop_cap : int | None
            ▷ 논문: 경로 길이 제약(선택). (제약식은 구현 재량)
        ddl_ms : float | None
            ▷ 논문: 데드라인 제약 t_r ≤ ddl (선택).
        return_delay : bool
            ▷ True면 (best_path, t_r_ms) 반환.

        Returns
        -------
        best_path : List[node_id]  또는 (best_path, delay_ms)
            ▷ 논문: 최적 경로 rp* (및 그 비용 t_r).
        """
        # S = r * (Δt_slot/1000) (Mb)  ← 슬롯에서 보낼 양
        size_mb = float(rate_Mbps) * (self.TIME_SLOT_MS / 1000.0)
        curr_ms = self.now_ms if now_ms is None else float(now_ms)

        # 후보 경로 집합 P 생성(전파지연 기반 K-최단)
        cands = self._candidate_paths(src, dst, hop_cap=hop_cap)
        if not cands:
            return (None, None) if return_delay else None

        # 사전 전개로 각 rp의 t_r 채점 → min t_r 선택
        best_path, best_delay = None, None
        for path in cands:
            delay_ms = self._nc_virtual_pass_ms(path, size_mb, curr_ms, commit=False)
            if ddl_ms is not None and delay_ms > ddl_ms:
                continue  # 제약 위반 컷
            if best_delay is None or delay_ms < best_delay:
                best_path, best_delay = path, delay_ms

        if best_path is None:
            return (None, None) if return_delay else None

        # 선택 경로 커밋(해당 경로가 실제로 슬롯 S만큼 흘렀다고 상태 업데이트)
        _ = self._nc_virtual_pass_ms(best_path, size_mb, curr_ms, commit=True)
        return (best_path, best_delay) if return_delay else best_path
