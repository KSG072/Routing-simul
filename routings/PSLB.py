# -*- coding: utf-8 -*-
"""
PSLB (Partial Scattered Load Balancing) — Satellite-only, RTPG position-based
-----------------------------------------------------------------------------
요구사항 반영:
1) **위성 노드만 사용** (지상 노드/링크 불포함, ISL만 고려)
2) PSLB에 **불필요한 코드 제거**(지연/시각화/엑스포트 등 전부 삭제, 최소 API 유지)
3) 엣지 (u,v)의 혼잡도는 **실제 큐 길이**(packets)로 계산:
   satellites[u].isl_* 가 v와 일치하면 해당 방향의 buffer.size 사용.
   (비트 단위가 필요하면 64,000을 곱해 환산 가능)

추가:
- (x,y) 좌표는 rtpg.G.nodes[n]['position']를 사용하며, 이는 사용자가 말한
  orbit index, sat index와 같은 2D 그리드로 간주합니다.
- 부분 그래프는 경로에 등장한 위성들의 (x,y)로 **최소 사각 창**을 만들어 구성.
  좌표가 정수 격자(0..N-1,0..M-1)처럼 보이면 seam(랩) 고려 최단 호를 각 축에 적용.

호환:
- build_flows(), get_path(), get_flows()는 유지
- build_load_from_totals()는 더 이상 필요 없지만, 파이프라인 호환을 위해 no-op로 남겨둠
"""
from __future__ import annotations
import math
import heapq
from typing import Any, Dict, List, Tuple, Set


class PSLBFlowController:
    def __init__(
        self,
        rtpg,
        satellites: Dict[int, Any],
        ground_relays: Dict[Any, Any],  # 미사용 (호환용)
        isl_capacity_mbps: float = 0.0,  # 미사용 (큐 기반으로만 판단)
        gsl_up_capacity_mbps: float = 0.0,  # 미사용
        gsl_down_capacity_mbps: float = 0.0,  # 미사용
        queue_capacity_pkts: int = 500,
        u_critical_ratio: float = 0.6,
        k1: float = 0.25,
        k2: float = 1.0,
        alpha: float = 1.0,
        sigma: float = 0.02,
    ) -> None:
        self.rtpg = rtpg
        self.satellites = satellites
        # ---- PSLB 파라미터 ----
        self.queue_capacity_pkts = int(queue_capacity_pkts)
        self.u_critical_ratio = float(u_critical_ratio)
        self.u_critical_pkts = self.queue_capacity_pkts * self.u_critical_ratio
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        # ---- 상태 ----
        self.time = 0
        # flows[(src,dst)] = [demand(Mbps), path(list), detours(int)]
        self.flows: Dict[Tuple[Any, Any], List[Any]] = {}
        self.flow_log: Dict[Tuple[Any, Any], Dict[int, List[Any]]] = {}
        self.new_flow: Set[Tuple[Any, Any]] = set()

    # ---------------------------------------------------------------------------------
    # 기본 유틸
    # ---------------------------------------------------------------------------------
    def _edge_type(self, u, v) -> str:
        data = self.rtpg.G.get_edge_data(u, v) or {}
        return data.get('type', 'isl')

    def _edge_queue_pkts(self, u, v) -> int:
        """엣지(u,v)의 현재 큐 길이(패킷 수)를 반환. 위성 u의 4방향과 v를 매칭.
        - 방향 키: isl_up/isl_down/isl_left/isl_right (또는 isl_light 호환)
        - 버퍼 키: isl_up_buffer/isl_down_buffer/isl_left_buffer/isl_right_buffer(isl_light_buffer)
        """
        sat = self.satellites.get(u)
        if sat is None:
            return 0
        candidates = [
            ('isl_up', 'isl_up_buffer'),
            ('isl_down', 'isl_down_buffer'),
            ('isl_left', 'isl_left_buffer'),
            ('isl_right', 'isl_right_buffer'),
            ('isl_light', 'isl_light_buffer'),  # 오타/변형 호환
        ]
        for dir_key, buf_key in candidates:
            nbr = getattr(sat, dir_key, None)
            if nbr is not None and nbr == v:
                buf = getattr(sat, buf_key, None)
                if buf is not None:
                    return int(getattr(buf, 'size', 0))
        return 0

    def _edge_util(self, u, v) -> float:
        # 큐 길이를 queue_capacity로 정규화하여 [0..1] 이용률 근사로 사용
        return min(1.0, self._edge_queue_pkts(u, v) / max(1, self.queue_capacity_pkts))

    def _path_edges(self, path: List[Any]):
        for a, b in zip(path, path[1:]):
            yield (a, b)

    def _log_path(self, fkey, path: List[Any]):
        self.flow_log.setdefault(fkey, {})[self.time] = list(path)

    # ---------------------------------------------------------------------------------
    # 좌표 헬퍼 (x,y) := rtpg.G.nodes[n]['position']  → orbit/sat index와 동일 취급
    # ---------------------------------------------------------------------------------
    def _sat_xy(self, nid: int) -> Tuple[float, float]:
        if not isinstance(nid, int):
            raise ValueError("_sat_xy expects satellite node id (int)")
        data = self.rtpg.G.nodes.get(nid, {})
        if 'position' not in data:
            raise KeyError(f"Satellite node {nid} has no 'position'; keys={list(data.keys())}")
        pos = data['position']
        # position이 (x,y) 튜플 혹은 [x,y]라고 가정. 만약 중첩리스트면 첫 원소 사용
        if isinstance(pos, (list, tuple)) and pos and isinstance(pos[0], (list, tuple)):
            pos = pos[0]
        return float(pos[0]), float(pos[1])

    # ---------------------------------------------------------------------------------
    # PSLB 핵심: 혼잡 엣지 찾기 → 부분 플로우 선택 → 부분 그래프 → 경로-의존 다익스트라
    # ---------------------------------------------------------------------------------
    def _most_congested_edge(self):
        """ISL 엣지 중 이용률이 가장 높은 (u,v,util)을 반환."""
        worst = None
        for u, v in self.rtpg.G.edges():
            if self._edge_type(u, v) != 'isl':
                continue
            util = self._edge_util(u, v)
            if worst is None or util > worst[2]:
                worst = (u, v, util)
        return worst

    def _flows_using_edge(self, edge_k: Tuple[Any, Any]):
        out = []
        for fkey, (demand, path, detours) in self.flows.items():
            if edge_k in self._path_edges(path):
                out.append((fkey, demand, path, detours))
        return out

    def _phi(self, n: int, util: float, gamma: int) -> int:
        if gamma == 1:
            return max(1, math.ceil(self.k1 * n))
        if util > self.u_critical_ratio:
            # util은 0..1, 임계치 초과분에 비례하여 추가 선택
            return max(1, math.ceil(self.k2 * (util - self.u_critical_ratio) * n))
        return 0

    def _select_partial(self, edge0: Tuple[Any, Any], util: float, gamma: int):
        flows = self._flows_using_edge(edge0)
        if not flows:
            return []
        # 수요 큰 순으로 (β_j = f_j·σ에서 f_j가 큼)
        flows.sort(key=lambda x: float(x[1]), reverse=True)
        k = self._phi(len(flows), util, gamma)
        return flows[:k]

    def _wrap_shortest_arc(self, values: List[int], modulo: int):
        uniq = sorted(set([v % modulo for v in values]))
        if not uniq:
            return (0, 0, False)
        if len(uniq) == 1:
            return (uniq[0], uniq[0], False)
        max_gap, max_i = -1, 0
        for i in range(len(uniq)):
            j = (i + 1) % len(uniq)
            gap = (uniq[j] - uniq[i]) % modulo
            if gap > max_gap:
                max_gap, max_i = gap, i
        start = uniq[(max_i + 1) % len(uniq)]
        end = uniq[max_i]
        return (start, end, start > end)

    def _in_wrapped(self, v: int, s: int, e: int, modulo: int, wrapped: bool) -> bool:
        if not wrapped:
            return s <= v <= e
        return v >= s or v <= e

    def _build_partial_subgraph(self, selected: List[Tuple[Tuple[Any, Any], float, List[Any], int]]):
        # 선택 플로우들의 위성 경로에서 (x,y) 좌표 수집
        xs: List[float] = []
        ys: List[float] = []
        for (_, _, path, _) in selected:
            for nid in path:
                if isinstance(nid, int):
                    x, y = self._sat_xy(nid)
                    xs.append(x)
                    ys.append(y)
        if not xs:
            # 위성 노드 전체 (satellite-only)
            sat_only = [n for n in self.rtpg.G.nodes if isinstance(n, int)]
            return self.rtpg.G.subgraph(sat_only).copy()

        # 정수 격자처럼 보이면 wrap-aware, 아니면 단순 bbox
        N, M = getattr(self.rtpg, 'N', 0), getattr(self.rtpg, 'M', 0)
        def looks_integer_grid(vals, upper):
            try:
                return upper > 0 and all(float(v).is_integer() and 0 <= int(v) < upper for v in vals)
            except Exception:
                return False
        use_wrap = looks_integer_grid(xs, N) and looks_integer_grid(ys, M)

        nodes_keep: Set[Any] = set()
        if use_wrap:
            xs_i = [int(v) for v in xs]
            ys_i = [int(v) for v in ys]
            xs_s, xs_e, xw = self._wrap_shortest_arc(xs_i, N)
            ys_s, ys_e, yw = self._wrap_shortest_arc(ys_i, M)
            for nid in self.satellites.keys():
                x, y = self._sat_xy(nid)
                xi, yi = int(x), int(y)
                if self._in_wrapped(xi, xs_s, xs_e, N, xw) and self._in_wrapped(yi, ys_s, ys_e, M, yw):
                    nodes_keep.add(nid)
        else:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            for nid in self.satellites.keys():
                x, y = self._sat_xy(nid)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    nodes_keep.add(nid)

        return self.rtpg.G.subgraph(nodes_keep).copy()

    def _pslb_shortest_path(self, H, src, dst, demand_mbps: float) -> List[Any]:
        beta_j = float(demand_mbps) * self.sigma  # 식(9)
        pq: List[Tuple[float, Any, float, int]] = []  # (g, u, cum_c, hops)
        heapq.heappush(pq, (0.0, src, 0.0, 0))
        best: Dict[Any, Tuple[float, float, int]] = {src: (0.0, 0.0, 0)}
        parent: Dict[Any, Any] = {}
        while pq:
            g, u, c_cum, h = heapq.heappop(pq)
            if u == dst:
                path = [u]
                while path[-1] != src:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            bg, _, _ = best[u]
            if g > bg + 1e-12:
                continue
            for _, v in H.out_edges(u):
                delta_c = self._edge_util(u, v)   # 현재 큐 기반 이용률 (식10의 Δc 역할)
                c_next = c_cum + delta_c
                h_next = h + 1
                inc = self.alpha * c_next + beta_j * h_next  # 식(12)
                g_next = g + inc
                b = best.get(v)
                if b is None or g_next + 1e-12 < b[0]:
                    best[v] = (g_next, c_next, h_next)
                    parent[v] = u
                    heapq.heappush(pq, (g_next, v, c_next, h_next))
        # 백업: 원 그래프 기본 최단경로
        path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
        return list(path)

    # ---------------------------------------------------------------------------------
    # 공개 API
    # ---------------------------------------------------------------------------------
    def build_flows(self, src, dst, demand_mbps: float) -> None:
        path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
        self.flows[(src, dst)] = [float(demand_mbps), list(path), 0]
        self.new_flow.add((src, dst))
        # self._log_path((src, dst), list(path))


    def get_path(self, fkey):
        return self.flows[fkey][1]

    def get_flows(self):
        return self.flows

    def solve_pslb(self, round_cap: int = 2):
        worst = self._most_congested_edge()
        if worst is None:
            return
        (eu, ev, util) = worst
        gamma = 1
        while util > self.u_critical_ratio and gamma <= max(1, round_cap):
            selected = self._select_partial((eu, ev), util, gamma)
            if not selected:
                break
            H = self._build_partial_subgraph(selected)
            for (fkey, demand, old_path, detours) in selected:
                src, dst = fkey
                if not (isinstance(src, int) and isinstance(dst, int)):
                    continue  # 위성-only 제한
                new_path = self._pslb_shortest_path(H, src, dst, demand)
                if new_path and new_path != old_path:
                    self.flows[fkey][1] = list(new_path)
                    if fkey not in self.new_flow:
                        self.flows[fkey][2] += 1
                    # self._log_path(fkey, list(new_path))
            # 다음 라운드 재평가
            worst = self._most_congested_edge()
            if worst is None:
                break
            (eu, ev, util) = worst
            gamma += 1
        return
