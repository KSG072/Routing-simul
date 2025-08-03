
"""
여기서 추출되는 키노드들은 지상노드 앞/뒤 위성 노드들의 집합
예를 들어,
유저-위성0-위성1-지상노드-위성2-위성-위성3-도착지(지상노드) 라고하면
key_nodes는 [위성0, 위성1, 위성2, 위성3]이 됨
"""
from collections import deque


def extract_key_nodes(path):
    key_nodes = deque()
    n = len(path)
    for i, tok in enumerate(path):
        if isinstance(tok, str):
            # 앞쪽
            if i > 0 and isinstance(path[i - 1], int):
                key_nodes.append(path[i - 1])
            # 뒤쪽
            if i < n - 1 and isinstance(path[i + 1], int):
                key_nodes.append(path[i + 1])
    return key_nodes


def extract_ground_nodes(path):
    ground_nodes = deque()
    for node in path:
        if isinstance(node, str):
            ground_nodes.append(node)

    return ground_nodes

class Packet:
    p_id = 0
    def __init__(self, t):
        self.idx = Packet.p_id
        Packet.p_id += 1
        self.curr_idx = 0
        self.source = None
        self.source_lat = None # 해야함
        self.source_lon = None # 해야함
        self.curr = None # storage 들어갈 때 바뀜
        self.destination = None
        self.path = None
        self.key_nodes = None # storage -> buffer 처리, curr == key_node일 때 바뀜
        self.key_node = None # storage -> buffer 처리, curr == key_node일 때 바뀜
        self.ground_nodes = None # storage -> buffer 처리, curr == ground_node일 때 바뀜
        self.ground_node = None # storage -> buffer 처리, curr == ground_node일 때 바뀜
        self.was_on_ground = True # storage -> buffer 처리, curr == ground_node일 때 바뀜

        self.result = [] # storage 들어갈 때 바뀜
        self.queuing_delays = []
        self.propagation_delays = 0
        self.remaining_prop_delay = 0
        self.transmission_delay = 0

        self.start_at = t
        self.end_at = None # 드롭/도착 했을 때 바뀜
        self.success = None # 드롭/도착 했을 때 바뀜
        self.dropped_node = None
        self.ended_lat = None
        self.ended_lon = None
        self.inconsistency = False

        self.remaining_v_hops = None # curr == key_node & was_on_ground일 때, storage 들어갈 때 바뀜
        self.remaining_h_hops = None # curr == key_node & was_on_ground일 때, storage 들어갈 때 바뀜

        self.rtpg = None

    def end(self, t, state, end_node_id, lat, lon):
        self.end_at = t
        self.success = True if state == 'success' else False
        self.inconsistency = True if state == 'inconsistency' else False
        self.dropped_node = end_node_id
        self.ended_lat, self.ended_lon = lat, lon


    def set_path_info(self, path):
        self.source = path[0]
        self.curr = path[0]
        self.destination = path[-1]
        self.path = path
        self.key_nodes = extract_key_nodes(self.path)
        _ = self.next_key_node_id()
        self.ground_nodes = extract_ground_nodes(self.path)
        _ = self.next_ground_node_id()
        self.result.append(self.curr)

    def next_key_node_id(self):
        if self.key_nodes:
            self.key_node = self.key_nodes.popleft()
            return self.key_node
        else:
            return None

    def next_ground_node_id(self):
        if self.ground_nodes:
            self.ground_node = self.ground_nodes.popleft()
            return self.ground_node
        else:
            return self.ground_node

    def set_remaining_hops(self, h, v):
        self.remaining_h_hops = h
        self.remaining_v_hops = v

    def set_rtpg(self, rtpg):
        self.rtpg = rtpg

    def set_key_node(self, new_node):
        self.key_node = new_node

    def set_propagation_delay(self, prop_delay):
        self.propagation_delays += prop_delay
        self.remaining_prop_delay = prop_delay

    def __repr__(self):
        """
        한 줄 요약 문자열.
        예: Packet#1 SRC →[+3]→ *MID* →[-2]→ DST (H=+5, V=-1)
        현재 위치(curr)는 *로 둘러싸 강조.
        """

        # 로컬 포맷터
        def _fmt(tok):
            if isinstance(tok, int):
                return f"[{tok:+d}]"
            return f"*{tok}*" if tok == self.curr else str(tok)

        line = " → ".join(_fmt(tok) for tok in self.path)
        H = self.remaining_h_hops
        V = self.remaining_v_hops
        return f"Packet#{self.idx} {line} (H={H}, V={V})"

    __str__ = __repr__

    def show_detailed(self):
        print("=" * 40)
        print(f"Packet ID       : {self.idx}")
        print(f"Start Time      : {self.start_at}")
        print(f"End Time        : {self.end_at}")
        print(f"Success         : {self.success}")
        print(f"Source          : {self.source}")
        print(f"Destination     : {self.destination}")
        print(f"Current Node    : {self.curr}")
        print(f"Current Index   : {self.curr_idx}")
        print(f"Current Key Node: {self.key_node}")
        print(f"Current Ground  : {self.ground_node}")
        print(f"Was on Ground   : {self.was_on_ground}")
        print(f"Remaining H-Hops: {self.remaining_h_hops}")
        print(f"Remaining V-Hops: {self.remaining_v_hops}")
        print("-" * 40)
        print(f"Path            : {self.path}")
        print(f"Key Nodes       : {list(self.key_nodes)}")
        print(f"Ground Nodes    : {list(self.ground_nodes)}")
        print(f"Result Path     : {self.result}")
        print(f"Queuing Delays  : {self.queuing_delays}")
        print(f"Dropped Lat/Lon : ({self.ended_lat}, {self.ended_lon})")
        print("=" * 40)