import math
from collections import defaultdict
from typing import Tuple, List, Any, Dict, Optional
from bisect import bisect_right

from parameters.PARAMS import PACKET_SIZE_BITS, ISL_RATE_LASER, TAU
from proposed_routingtable import import_routing_tables_from_csv_json

class RoutingTable:
    def __init__(self, directory):
        self.routing_table = {}
        self.table_len = 0
        self.directory = directory

    def load_routing_table(self, rate, time):
        self.routing_table[time] = import_routing_tables_from_csv_json(self.directory, rate, time)
        self.table_len += 1
        print(f"Routing table loaded for rate: {rate} Mbps, time: {time} ms")

    def get_table_key(self, packet_time):
        key = 0
        if self.table_len > 1:
            times = sorted(self.routing_table.keys())
            for t in times:
                if packet_time >= t:
                    key = t
                else:
                    break
        return key

    def sat_forwarding(self, cur, packet):
        key = self.get_table_key(packet.start_at)
        table = self.routing_table[key]
        direction = table[str(cur.node_id)].get((packet.source, packet.destination), None)
        if direction is None:
            packet.show_detailed()
            print("what!!?!?!? in sat_forwarding")
            return None
        return direction

    def ground_forwarding(self, cur, packet):
        key = self.get_table_key(packet.start_at)
        table = self.routing_table[key]
        direction = table[cur.node_id].get((packet.source, packet.destination), None)
        if direction is None:
            packet.show_detailed()
            print("what!!?!? in ground_forwarding")
            return None
        return direction


FlowKey = Tuple[Any, Any]  # (src, dst) — int/str 혼합 가능
Path = List[Any]           # [node ids ...]

class RoutingSchedule:
    """
    flow_log.pkl을 로드해 시간별 경로를 조회하고,
    (fkey, t, current_node) -> next hop 을 제공하는 헬퍼.

    flow_log 형식:
      {
        (src, dst): { time_int: [path_node_ids...] , ... },
        ...
      }
    """

    def __init__(self, directory, rate):
        """
        file_path: dict 또는 pkl 파일 경로(str)
        """
        file_path = f"./{directory}/flows_{rate}.pkl"
        with open(file_path, 'rb') as f:
            import pickle
            self.flow_log: Dict[FlowKey, Dict[int, Path]] = pickle.load(f)


        # 빠른 시간 조회를 위해 정렬 인덱스 구성
        self._index: Dict[FlowKey, Tuple[List[int], List[Path]]] = {}
        for fkey, tmap in self.flow_log.items():
            times = sorted(tmap.keys())
            paths = [tmap[t] for t in times]
            self._index[fkey] = (times, paths)

    def fix_flow(self, t: int, fkey: Tuple[Any, Any], s_id: Any, next_hop: Any):

        # 1) 베이스라인 경로 가져오기 (t 이하에서 가장 최신)
        base = self.get_path(fkey, t)

        # 2) s_id 위치 찾기
        idx_s = base.index(s_id)

        # 3) src→s_id 구간에서 next_hop이 이미 존재하면, 해당 지점~s_id 직전 삭제
        #    (루프/중복 제거, s_id는 남겨둠)
        if next_hop in base:
            k = base.index(next_hop)  # [0, idx_s) 범위에서 탐색
            if k < idx_s:
                del base[k+1:idx_s+1]
            else: # k > idx_s
                del base[idx_s+1:k]
        else:
            # 4) s_id 다음에 next_hop 삽입 (이미 그 다음이면 생략)
            if idx_s == len(base) - 1:
                base.append(next_hop)
            elif base[idx_s + 1] != next_hop:
                base.insert(idx_s + 1, next_hop)


    # ----------------- Core lookups -----------------

    def get_path(self, fkey: FlowKey, t: int) -> Optional[Path]:
        """
        시각 t에서 유효한 경로(가장 가까운 과거 시점의 경로)를 반환.
        없으면 None.
        """
        if fkey not in self._index:
            return None
        times, paths = self._index[fkey]
        i = bisect_right(times, t) - 1
        if i < 0:
            return None
        return paths[i]

    def get_next_hop_inflow(self, fkey: FlowKey, t: int, current_node: Any) -> Optional[Any]:
        """
        주어진 flow fkey와 시각 t, 현재노드에서의 다음 홉을 반환.
        - 경로가 없거나 current_node가 경로에 없으면 None
        - current_node가 마지막 노드면 None
        """
        path = self.get_path(fkey, t)
        if not path:
            return None
        try:
            idx = path.index(current_node)
        except ValueError:
            return None
        if idx >= len(path) - 1:
            return None
        return path[idx + 1]

    def get_next_hop(self, path, cur_idx):
        return path[cur_idx+1]


    # ----------------- Convenience -----------------

    def has_flow(self, fkey: FlowKey) -> bool:
        return fkey in self._index

    def first_time(self, fkey: FlowKey) -> Optional[int]:
        if fkey not in self._index:
            return None
        return self._index[fkey][0][0] if self._index[fkey][0] else None

    def last_time(self, fkey: FlowKey) -> Optional[int]:
        if fkey not in self._index:
            return None
        return self._index[fkey][0][-1] if self._index[fkey][0] else None

    def flows(self) -> List[FlowKey]:
        return list(self._index.keys())

    def active_flows_at(self, t: int) -> List[FlowKey]:
        """
        시각 t에 경로가 정의되어 있는(=t 이하의 시점이 존재하는) 플로우 목록
        """
        out = []
        for fkey, (times, _) in self._index.items():
            i = bisect_right(times, t) - 1
            if i >= 0:
                out.append(fkey)
        return out
