import multiprocessing
from datetime import datetime
#python 3.11.13
import re
from datetime import datetime
from collections import deque
from warnings import catch_warnings

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from random import choices, sample
import os
from routings.flow_recorder import FlowRecorder

from routings.dijkstra import sat_to_sat_forwarding_d
from routings.proposed_NCC import FlowController

os.environ["PYCHARM_DISPLAY"] = "none"
_GROUND_RE = re.compile(r'^[A-Za-z]+-\d+$')  # 지상 노드: 영문자-숫자

from parameters.PARAMS import *
from routings.packet import Packet
from routings.sota import sat_to_sat_forwarding, sat_to_ground_forwarding, ground_to_sat_forwarding
from routings.proposed_algorithm import RoutingTable, RoutingSchedule
from routings.minimum_hop_estimator import min_hop_distance
from utils.plot_maker import load_heatmap
from utils.walker_constellation import WalkerConstellation
from utils.loader import load_ground_relays_from_csv, batch_map_nodes, normalize_wrapped_regions, \
    prepare_node_routing_metadata, load_event_schedule
from utils.rtpg_mapper import RTPGMapper
from utils.RTPGGraph import RTPGGraph
from utils.user_node_generator import generate_users, generate_cities
from utils.csv_maker import csv_write, csv_create

def delay_estimation(path, satellites, ground_relays):
    expected_delay = 0
    cur = satellites[path[0]]
    for hop in path[1:]:
        is_satellite = isinstance(hop, int)
        if is_satellite:
            togo = satellites[hop]
        else:
            togo = ground_relays[hop]

        cur_coords, togo_coords = cur.cartesian_coords, togo.cartesian_coords
        expected_delay += calculate_prop_delay(cur_coords, togo_coords)

        link = 'isl'
        if isinstance(cur.node_id, int):
            if cur.isl_up == hop:
                q_size = cur.isl_up_buffer.size
            elif cur.isl_down == hop:
                q_size = cur.isl_down_buffer.size
            elif cur.isl_left == hop:
                q_size = cur.isl_left_buffer.size
            elif cur.isl_right == hop:
                q_size = cur.isl_right_buffer.size
            else:
                q_size = cur.gsl_down_buffers[hop].size
                link = 'down'
        else:
            q_size = cur.gsl_up_buffers[hop].size
            link = 'up'

        if link == 'isl':
            expected_delay += (q_size*PACKET_SIZE_BITS)/(TAU*ISL_RATE_LASER)
        elif link == 'down':
            expected_delay += (q_size * PACKET_SIZE_BITS) / (TAU * SGL_KA_DOWNLINK)
        else:
            expected_delay += (q_size * PACKET_SIZE_BITS) / (TAU * SGL_KA_UPLINK)

        cur = togo

    return expected_delay

def _is_ground_id(x) -> bool:
    return bool(_GROUND_RE.match(str(x).strip()))

def get_route_sat_to_sat(rtpg, src, dst, n=1):

    # 최단 경로 찾기
    routes = []
    route = rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
    for _ in range(n):
        routes.append(route)

    return routes

def calculate_hop_distance(packet, satellites):
    try:
        src, semi_dst = satellites[packet.curr], satellites[packet.key_node]
        h, v = min_hop_distance(src, semi_dst, N, M, F)
        packet.set_remaining_hops(h, v)
    except KeyError:
        print("D")

def calculate_prop_delay(src, dst):
    distance = norm(dst-src)
    prop_delay_sec = distance/C
    return prop_delay_sec*1000

def transfer(sequences, next_hops, src_coords, disconnected=None):
    if disconnected is None:
        disconnected = []
    direction = 0 # 0:up, 1:down, 2:left, 3:right, 4:ground, 5:satellite
    failed = []
    for pkts in sequences: # sequences: [[pkt, pkt, ...],[],[],[],[{gr1: [], gr2: [], ...}],[sat1: [], sat2: [], ...]]
        if direction <= 3:
            for pkt in pkts:
                if direction == 0:
                    pkt.remaining_v_hops -= 1
                elif direction == 1:
                    pkt.remaining_v_hops += 1
                elif direction == 2:
                    pkt.remaining_h_hops += 1
                elif direction == 3:
                    pkt.remaining_h_hops -= 1
                else:
                    pass
                next_hop = next_hops[direction]

                next_coords = next_hop.cartesian_coords
                pkt.set_propagation_delay(calculate_prop_delay(src_coords, next_coords))
                pkt.transmission_delay += TRANS_DELAY_ISL

                next_hop.receive_packet(pkt)
                pkt.curr = next_hop.node_id
                pkt.result.append(next_hop.node_id)
        else:
            if pkts:
                # print(pkts)
                try:
                    for detail_direction, packets in pkts[0].items():
                        for p in packets:
                            if (p.curr, detail_direction) not in disconnected and detail_direction in next_hops[direction]:
                                next_hop = next_hops[direction][detail_direction]

                                next_coords = next_hop.cartesian_coords
                                p.set_propagation_delay(calculate_prop_delay(src_coords, next_coords))

                                next_hop.receive_packet(p)
                                p.curr = next_hop.node_id
                                p.result.append(next_hop.node_id)
                                """
                                위성->지상: 지상에서 보내줄 다음 위성 노드를 key node로 변경
                                지상->위성: 위성에서 향할 다음 지상 노드를 ground node로 변경
                                """
                                if direction == 4:
                                    p.transmission_delay += TRANS_DELAY_DOWN
                                else:
                                    p.transmission_delay += TRANS_DELAY_UP
                            else:
                                """패킷 생성 당시엔 있다고 판단된 경로가, 와보니 끊어져있는 상황"""
                                failed.append(p)

                except AttributeError:
                    print("앙앙앙~")
                except IndexError:
                    print("엉엉엉~")
        direction += 1

    return failed

def check_cross_counts(nodes: deque):
    #nodes = {key node1(down), key node2(up), ..., destination}
    nodes.pop() # destination 제거용
    cross_count = 0
    while nodes:
        down = nodes.popleft()
        up = nodes.popleft()
        if down.is_ascending() is not up.is_ascending():
            cross_count += 1
        else:
            pass

    return cross_count

class Simulator:
    def __init__(self, algorithm, generation_rate, filepath, table_dir, simulation_time, tqdm_position=None):
        """시뮬레이션 실행에 필요한 모든 변수를 초기화합니다."""
        self.algorithm = algorithm
        # self.disconnect_pair = []
        # self.disconnect_occur = False
        self.disconnect_cache = set()
        self.packet_generation_mode = True
        self.generation_rate = generation_rate
        self.filepath = filepath
        self.tqdm_position = tqdm_position
        self.filename = f"result_{self.generation_rate}_{simulation_time}.csv"
        self.table_dir = table_dir

        # CSV 헤더 및 파일 생성
        self.header = [
            "Time (ms)", "source", "destination", "Path Length", "expected length", "Detour counts", "Detour log",
            "cross counts", "result", "e2e delay", "Queuing delays", "Queuing Delay", "Propagation Delay",
            "Transmission Delay", "Detour mode",
            "Status", "Drop Location", "Drop Direction", "Drop Latitude", "Drop Longitude", "TTL",
            "expected delay(result)", "expected delay(isl)", "ISL Path Length"
        ]
        csv_create(self.header, self.filepath, self.filename)

        # 시뮬레이션 파라미터 설정
        self.t = 0
        self.dt = TIME_SLOT
        self.total_time = simulation_time
        self.steps = int(self.total_time / self.dt)
        t_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
        self.omega_s = 2 * np.pi / t_ms  # delta phase (deg)

        # 결과 저장을 위한 리스트
        self.results = []
        self.failed = []
        self.dropped = []

        # 카운터
        self.generated_count = 0
        self.ended = 0
        self.succeeded = 0
        self.drop_cnt = 0
        self.fail_cnt = 0
        self.flow_recorder = FlowRecorder()

        # 시뮬레이션 컴포넌트 초기화
        self._initialize_components()

    def _initialize_components(self):
        """위성, 지상국, 라우팅 테이블 등 시뮬레이션 구성요소를 설정합니다."""
        traffic_schedule_path = f'./parameters/uneven traffic(latest)/events_{self.generation_rate}Mbps.csv'
        relay_csv_path = './parameters/Ground_Relay_Coordinates.csv'

        self.mapper = RTPGMapper(N, M, F, inclination_deg)

        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        self.satellites = constellation.get_all_satellites()

        self.ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
        self.traffic_schedule = load_event_schedule(traffic_schedule_path, self.total_time)

        for gr in self.ground_relays.values():
            prepare_node_routing_metadata(gr, self.mapper, 550)

        self.rtpg = RTPGGraph(N=N, M=M, F=F)
        sat_region_indices = self.mapper.batch_map(self.satellites.values())
        self.rtpg.update_rtpg(self.satellites.values(), self.ground_relays.values(), sat_region_indices)
        self.rtpg.integrity_check()

        if self.algorithm == 'proposed(table)':
            self.table = RoutingTable(self.table_dir)
            self.table.load_routing_table(self.generation_rate, 0)
            self.flow_controller = None
        elif self.algorithm == 'proposed(flow)' or self.algorithm == 'dijkstra':
            # self.flow = RoutingSchedule(self.table_dir, self.generation_rate)
            self.table = None
            from math import floor
            ISL_CAP = floor((ISL_RATE_LASER) / 1000000)  # bps -> Mbps
            GSL_UP_CAP = floor((SGL_KA_UPLINK) / 1000000)
            GSL_DOWN_CAP = floor((SGL_KA_DOWNLINK) / 1000000)
            self.flow_controller = FlowController(self.rtpg, self.satellites, self.ground_relays, ISL_CAP, GSL_UP_CAP,
                                                  GSL_DOWN_CAP)
        else:
            self.table, self.flow_controller = None, None

    def update_graph_and_generate_packet(self):
        # RTPG 업데이트
        if self.t % 600 == 0 and self.t != 0:
            for s in self.satellites.values():
                s.update_lat_lon_for_RTPG()
            sat_region_indices = self.mapper.batch_map(self.satellites.values())
            self.rtpg.reset_graph()
            self.rtpg.update_rtpg(self.satellites.values(), self.ground_relays.values(), sat_region_indices)

            for satellite in self.satellites.values():
                if satellite.has_packets():
                    self.failed += satellite.trash_packets()
            for gr in self.ground_relays.values():
                if gr.has_packets():
                    self.failed += gr.trash_packets()

            # if self.algorithm == 'proposed':
            #     self.table.load_routing_table(self.generation_rate, self.t)
        if self.algorithm == 'proposed(flow)':
            self.flow_controller.time = self.t

        # if self.disconnect_occur:
        #     while self.disconnect_pair:
        #         (s, g) = self.disconnect_pair.pop()
        #         self.failed += s.trash_packets()
        #         self.failed += g.trash_packets()
        #     self.disconnect_occur = False

        # 패킷 생성
        if self.packet_generation_mode:
            generated_packets = self.traffic_schedule.pop(self.t, [])
            for packet_data in generated_packets:
                (src, dst, num_of_packets) = packet_data

                if self.algorithm == 'proposed(flow)' or self.algorithm == 'dijkstra':
                    fkey = (src, dst)

                    if self.flow_recorder.is_new_flow(fkey): #새로운 flow이면 flow_controller에서 경로를 추가하여 계산함
                        self.flow_recorder.create_flow(fkey)
                        demand = 0 if self.algorithm == 'dijkstra' else self.generation_rate
                        self.flow_controller.build_flows(src, dst, demand)
                        self.flow_controller.build_load_from_totals()
                        updated_flows = self.flow_controller.solve(max_iter=2000) | {(src, dst)}
                        self.flow_controller.hold_flows(self.t, updated_flows)

                    self.flow_recorder.record_flow_on(fkey, self.t, num_of_packets)
                    path = list(self.flow_controller.get_path(fkey))
                    paths = []
                    for _ in range(num_of_packets):
                        paths.append((path, len(path)))
                else:
                    fkey = (src, dst)
                    if self.flow_recorder.is_new_flow(fkey):
                        self.flow_recorder.create_flow(fkey)
                    self.flow_recorder.record_flow_on((src,dst), self.t, num_of_packets)
                    paths = get_route_sat_to_sat(self.rtpg, src, dst, n=num_of_packets) # 다익스트라

                self.generated_count += num_of_packets

                for path in paths:
                    new_packet = Packet(self.t, 0)
                    new_packet.set_path_info(path[0])
                    new_packet.initial_length = path[1]
                    calculate_hop_distance(new_packet, self.satellites)
                    self.satellites[src].storage.append(new_packet)
                    new_packet.last_direction = self.satellites[src].is_ascending()

            self.flow_recorder.record_flow_end(self.t)

    def transmit_phase(self):
        # 전송 페이즈 구현
        failed = []
        for s in self.satellites.values():
            if s.has_packets():
                bullets = s.get_packets(self.dt)  # (up, down, left, right, ground, satellite=[])로 보낼 패킷 리스트
                next_hops = [  # (up, down, left, right, ground, satellite) 방향 다음 홉
                    self.satellites[s.isl_up], self.satellites[s.isl_down],
                    self.satellites[s.isl_left], self.satellites[s.isl_right],
                    {node_id: self.ground_relays[node_id] for node_id in s.gsl_down_buffers.keys()}, []
                ]
                failed += transfer(bullets, next_hops, s.cartesian_coords, disconnected=self.disconnect_cache)
            else:
                continue

        for g in self.ground_relays.values():
            if g.has_packets():
                bullets = g.get_packets(self.dt)  # (up=[], down=[], left=[], right=[], ground=[], satellite)로 보낼 패킷 리스트
                next_hops = [  # (up, down, left, right, ground, satellite) 방향 다음 홉
                    [], [],
                    [], [],
                    [], {node_id: self.satellites[node_id] for node_id in g.gsl_up_buffers.keys()}
                ]
                failed += transfer(bullets, next_hops, g.cartesian_coords, disconnected=self.disconnect_cache)
            else:
                continue
        return failed

    def drop_phase(self):
        # 드롭 페이즈 구현
        dropped = []
        for s in self.satellites.values():
            dropped += s.drop_packet()

        return dropped

    def routing_phase(self):
        if self.algorithm == 'proposed(table)':
            return self.proposed_routing_table()
        elif self.algorithm == 'proposed(flow)':
            return self.proposed_routing_flow()
        elif self.algorithm == 'dijkstra':
            return self.dijstra_routing()
        elif self.algorithm == 'tmc':
            return self.tmc_routing()
        else:
            raise ValueError("Invalid routing algorithm specified.")

    def proposed_routing_table(self):
        # 라우팅 페이즈 구현
        success = []
        failed = []
        for s in self.satellites.values():
            while s.storage:
                packet = s.storage.popleft()

                if s.node_id == packet.destination:  # 도착, success
                    packet.end(self.t, 'success', s.node_id, s.latitude_deg, s.longitude_deg)
                    success.append(packet)
                    self.succeeded += 1
                else:
                    next_hop = self.table.sat_forwarding(s, packet)
                    if next_hop is None:
                        failed.append(packet)
                        continue
                    if isinstance(next_hop, int):  # 위성
                        direction = [s.isl_up, s.isl_down, s.isl_left, s.isl_right].index(next_hop)
                        buffer_queue = [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][
                            direction]
                        data_rate = ISL_RATE_LASER
                    else:
                        direction = next_hop
                        buffer_queue = s.gsl_down_buffers[direction]
                        data_rate = SGL_KA_DOWNLINK
                        packet.last_direction = s.is_ascending()

                    packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * data_rate))

                    if packet.ttl <= 0:
                        failed.append(packet)
                    else:
                        packet.was_on_ground = False
                        s.enqueue_packet(direction, packet)
        for g in self.ground_relays.values():
            while g.storage:
                packet = g.storage.popleft()
                direction = self.table.ground_forwarding(g, packet)
                if direction is None:
                    failed.append(packet)
                    continue
                buffer_queue = g.gsl_up_buffers[direction]
                packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * SGL_KA_UPLINK))

                if packet.last_direction != self.satellites[direction].is_ascending():  # cross count
                    packet.cross_count += 1

                if packet.ttl <= 0:
                    failed.append(packet)
                else:
                    g.enqueue_packet(direction, packet)

        return success, failed

    def proposed_routing_flow(self):
        # 라우팅 페이즈 구현
        success = []
        failed = []
        for s in self.satellites.values():
            while s.storage:
                packet = s.storage.popleft()
                if s.node_id == packet.destination:  # 도착, success
                    packet.end(self.t, 'success', s.node_id, s.latitude_deg, s.longitude_deg)
                    success.append(packet)
                    self.succeeded += 1
                else:
                    fkey = (packet.source, packet.destination)
                    next_hop = packet.path[packet.curr_idx+1]
                    if next_hop is None:
                        failed.append(packet)
                        continue
                    isl_candidates = [s.isl_up, s.isl_down, s.isl_left, s.isl_right]
                    if isinstance(next_hop, int):  # 위성
                        direction = isl_candidates.index(next_hop)
                        buffer_queue = [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][direction]
                        data_rate = ISL_RATE_LASER
                    else:  # 지상
                        direction = next_hop
                        if direction in s.gsl_down_buffers.keys():
                            buffer_queue = s.gsl_down_buffers[direction]
                            data_rate = SGL_KA_DOWNLINK
                            packet.last_direction = s.is_ascending()
                        else:  # 링크 끊김
                            if fkey in s.fixed.keys():
                                new_next_hop = s.fixed[fkey]
                                packet.path.insert(packet.curr_idx + 1, new_next_hop)
                                packet.detour_at.append(packet.curr)
                                direction = isl_candidates.index(new_next_hop)
                                buffer_queue = \
                                [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][direction]
                                data_rate = ISL_RATE_LASER
                            else:
                                candidates = [node_id for node_id in isl_candidates if
                                              direction in self.satellites[node_id].gsl_down_buffers.keys()]
                                if candidates:
                                    new_next_hop = min(candidates,
                                                       key=lambda node_id: self.satellites[node_id].gsl_down_buffers[
                                                           direction].size)
                                    self.flow_controller.fix_flow(fkey, s.node, new_next_hop)
                                    s.fixed[fkey] = new_next_hop
                                    print(
                                        f"Generation rate: {self.generation_rate}, Time: {self.t}, flow: {fkey}, cur: {s.node_id}, try: {direction}, detour to: {new_next_hop}")
                                    packet.path.insert(packet.curr_idx + 1, new_next_hop)
                                    packet.detour_at.append(packet.curr)
                                    direction = isl_candidates.index(new_next_hop)
                                    buffer_queue = \
                                    [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][
                                        direction]
                                    data_rate = ISL_RATE_LASER
                                else:
                                    failed.append(packet)
                                    continue

                    packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * data_rate))

                    if packet.ttl <= 0:
                        failed.append(packet)
                    else:
                        packet.was_on_ground = False
                        s.enqueue_packet(direction, packet)
                        packet.curr_idx += 1

        for g in self.ground_relays.values():
            while g.storage:
                packet = g.storage.popleft()
                fkey = (packet.source, packet.destination)
                direction = packet.path[packet.curr_idx+1]

                if direction is None:
                    failed.append(packet)
                    continue

                if direction in g.gsl_up_buffers.keys():
                    buffer_queue = g.gsl_up_buffers[direction]
                else:
                    if fkey in g.fixed.keys():
                        next_hop = g.fixed[fkey]
                        packet.path.insert(packet.curr_idx + 1, next_hop)
                        packet.detour_at.append(packet.curr)
                        direction = next_hop
                        buffer_queue = g.gsl_up_buffers[direction]
                    else:
                        dir_obj = self.satellites[direction]
                        adjacency_node_id = [dir_obj.isl_up, dir_obj.isl_down, dir_obj.isl_left, dir_obj.isl_right]
                        candidates = [node_id for node_id in adjacency_node_id if node_id in g.gsl_up_buffers.keys()]
                        if candidates:
                            next_hop = min(candidates, key=lambda node_id: g.gsl_up_buffers[node_id].size)
                            self.flow_controller.fix_flow(fkey, g.node_id, next_hop)
                            g.fixed[fkey] = next_hop
                            packet.path.insert(packet.curr_idx + 1, next_hop)
                            packet.detour_at.append(packet.curr)
                            direction = next_hop
                            buffer_queue = g.gsl_up_buffers[direction]
                        else:
                            failed.append(packet)
                            continue

                packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * SGL_KA_UPLINK))
                if packet.last_direction != self.satellites[direction].is_ascending():  # cross count
                    packet.cross_count += 1

                if packet.ttl <= 0:
                    failed.append(packet)
                else:
                    packet.was_on_ground = True
                    g.enqueue_packet(direction, packet)
                    packet.curr_idx += 1

        return success, failed

    def dijstra_routing(self):
        # 라우팅 페이즈 구현
        success = []
        failed = []
        for s in self.satellites.values():
            while s.storage:
                packet = s.storage.popleft()
                if s.node_id == packet.destination:  # 도착, success
                    packet.end(self.t, 'success', s.node_id, s.latitude_deg, s.longitude_deg)
                    success.append(packet)
                    self.succeeded += 1
                else:
                    fkey = (packet.source, packet.destination)
                    next_hop = packet.path[packet.curr_idx+1]
                    if next_hop is None:
                        failed.append(packet)
                        continue
                    isl_candidates = [s.isl_up, s.isl_down, s.isl_left, s.isl_right]
                    if isinstance(next_hop, int):  # 위성
                        direction = isl_candidates.index(next_hop)
                        buffer_queue = [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][direction]
                        data_rate = ISL_RATE_LASER
                    else: # 지상
                        direction = next_hop
                        if direction in s.gsl_down_buffers.keys():
                            buffer_queue = s.gsl_down_buffers[direction]
                            data_rate = SGL_KA_DOWNLINK
                            packet.last_direction = s.is_ascending()
                        else: # 링크 끊김
                            if fkey in s.fixed.keys():
                                new_next_hop = s.fixed[fkey]
                                packet.path.insert(packet.curr_idx + 1, new_next_hop)
                                packet.detour_at.append(packet.curr)
                                direction = isl_candidates.index(new_next_hop)
                                buffer_queue = [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][direction]
                                data_rate = ISL_RATE_LASER
                            else:
                                candidates = [node_id for node_id in isl_candidates if direction in self.satellites[node_id].gsl_down_buffers.keys()]
                                if candidates:
                                    new_next_hop = min(candidates, key=lambda node_id: self.satellites[node_id].gsl_down_buffers[direction].size)
                                    self.flow_controller.fix_flow(fkey, s.node, new_next_hop)
                                    s.fixed[fkey] = new_next_hop
                                    print(f"Generation rate: {self.generation_rate}, Time: {self.t}, flow: {fkey}, cur: {s.node_id}, try: {direction}, detour to: {new_next_hop}")
                                    packet.path.insert(packet.curr_idx+1, new_next_hop)
                                    packet.detour_at.append(packet.curr)
                                    direction = isl_candidates.index(new_next_hop)
                                    buffer_queue = [s.isl_up_buffer, s.isl_down_buffer, s.isl_left_buffer, s.isl_right_buffer][direction]
                                    data_rate = ISL_RATE_LASER
                                else:
                                    failed.append(packet)
                                    continue

                    packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * data_rate))

                    if packet.ttl <= 0:
                        failed.append(packet)
                    else:
                        packet.was_on_ground = False
                        s.enqueue_packet(direction, packet)
                        packet.curr_idx += 1

        for g in self.ground_relays.values():
            while g.storage:
                packet = g.storage.popleft()
                fkey = (packet.source, packet.destination)
                direction = packet.path[packet.curr_idx+1]

                if direction is None:
                    failed.append(packet)
                    continue

                if direction in g.gsl_up_buffers.keys():
                    buffer_queue = g.gsl_up_buffers[direction]
                else:
                    if fkey in g.fixed.keys():
                        next_hop = g.fixed[fkey]
                        packet.path.insert(packet.curr_idx + 1, next_hop)
                        packet.detour_at.append(packet.curr)
                        direction = next_hop
                        buffer_queue = g.gsl_up_buffers[direction]
                    else:
                        dir_obj = self.satellites[direction]
                        adjacency_node_id = [dir_obj.isl_up, dir_obj.isl_down, dir_obj.isl_left, dir_obj.isl_right]
                        candidates = [node_id for node_id in adjacency_node_id if node_id in g.gsl_up_buffers.keys()]
                        if candidates:
                            next_hop = min(candidates, key=lambda node_id: g.gsl_up_buffers[node_id].size)
                            self.flow_controller.fix_flow(fkey, g.node_id, next_hop)
                            g.fixed[fkey] = next_hop
                            packet.path.insert(packet.curr_idx + 1, next_hop)
                            packet.detour_at.append(packet.curr)
                            direction = next_hop
                            buffer_queue = g.gsl_up_buffers[direction]
                        else:
                            failed.append(packet)
                            continue

                packet.queuing_delays.append((buffer_queue.size * PACKET_SIZE_BITS) / (TAU * SGL_KA_UPLINK))
                if packet.last_direction != self.satellites[direction].is_ascending():  # cross count
                    packet.cross_count += 1

                if packet.ttl <= 0:
                    failed.append(packet)
                else:
                    packet.was_on_ground = True
                    g.enqueue_packet(direction, packet)
                    packet.curr_idx += 1

        return success, failed

    def tmc_routing(self):
        success = []
        failed = []
        for s in self.satellites.values():
            is_ascending = s.is_ascending()
            while s.storage:
                packet = s.storage.popleft()
                if packet.was_on_ground and (packet.last_direction is not is_ascending):  # cross count
                    packet.cross_count += 1
                packet.last_direction = is_ascending

                if packet.curr == packet.destination:  # 도착, success
                    packet.end(self.t, 'success', s.node_id, s.latitude_deg, s.longitude_deg)
                    success.append(packet)
                    self.succeeded += 1
                    continue

                if packet.curr == packet.key_node:
                    key_node = packet.next_key_node_id()
                    if (not packet.was_on_ground) or (
                            packet.was_on_ground and packet.curr == key_node):  # 위성으로부터 받은 패킷이거나, 지상에서 올라오자마자 내려가야하는 경우
                        """지상으로 갈 수 없는 경우"""
                        if packet.ttl <= 0 or packet.ground_node not in s.gsl_down_buffers.keys():
                            failed.append(packet)
                            continue
                        """지상으로 갈 수 있는 경우"""
                        if packet.was_on_ground:  # 올라오자마자 지상으로
                            packet.was_on_ground = False
                            packet.next_key_node_id()
                        # 위성 홉 거친 후 지상으로
                        family = [self.satellites[node_id] for node_id in self.ground_relays[packet.ground_node].connected_sats]
                        need_detour, packet, direction = sat_to_ground_forwarding(s, packet, family)
                        if need_detour:  # 바로 못가고 다른 key node로 가야할 경우
                            packet.detour_mode = True
                            packet.detour_at.append((packet.curr, direction))
                            packet.key_nodes.appendleft(packet.key_node)
                            packet.set_key_node(direction)  # 새로운 keynode 설정
                            calculate_hop_distance(packet, self.satellites)
                            # packet.show_detailed()
                        else:  # 바로 가는 경우
                            packet.detour_mode = False
                            if packet.ttl <= 0:
                                failed.append(packet)
                            else:
                                s.enqueue_packet(direction, packet)
                            continue  # 지상 큐에 삽입 후 다음 패킷 처리
                    else:  # 지상으로부터 받은 패킷이고, 다른 위성으로 가야하는 경우
                        calculate_hop_distance(packet, self.satellites)
                        packet.was_on_ground = False

                # 단순 위성 포워딩 (잔여 홉 있음)

                horizontal = self.satellites[s.isl_left if packet.remaining_h_hops < 0 else s.isl_right]
                vertical = self.satellites[s.isl_down if packet.remaining_v_hops < 0 else s.isl_up]
                """위성-위성 라우팅 알고리즘 적용 부분 (Queuing delay는 여기서 계산됨)"""
                direction = sat_to_sat_forwarding(s, horizontal, vertical, packet)  # 0:up, 1:down, 2:left, 3:right
                if packet.ttl <= 0:
                    failed.append(packet)
                else:
                    s.enqueue_packet(direction, packet)
        for g in self.ground_relays.values():
            while g.storage:
                packet = g.storage.popleft()
                if packet.ttl <= 0 or packet.key_node not in g.gsl_up_buffers.keys():
                    failed.append(packet)
                    continue
                packet.was_on_ground = True
                packet.next_ground_node_id()
                family = [self.satellites[node_id] for node_id in self.ground_relays[packet.curr].connected_sats]
                """지상-위성 라우팅 알고리즘 적용 부분"""
                packet, direction = ground_to_sat_forwarding(g, packet, family)
                g.enqueue_packet(direction, packet)

        return success, failed

    def fail_drop_process_phase(self):
        # 실패 페이즈 구현
        results = []
        while self.failed:
            # print(failed)
            p = self.failed.pop(0)
            self.fail_cnt += 1
            if isinstance(p.curr, int):
                end_node = self.satellites[p.curr]
            else:
                end_node = self.ground_relays[p.curr]

            if p.ttl > 0:
                p.end(self.t, 'failure', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
            else:
                p.end(self.t, 'expired', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
            # print(end_node.connected_sats)
            # p.show_detailed()
            results.append(p)

        while self.dropped:
            p = self.dropped.pop(0)
            self.drop_cnt += 1
            end_node = self.satellites[p.curr]
            p.end(self.t, 'drop', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
            # print(2)
            results.append(p)

        return results

    def update_satellite_and_links(self):
        # 위성 위치 및 링크 상태 업데이트
        failed = []
        """위성 공전"""
        for s in self.satellites.values():
            s.update_position(self.omega_s, self.dt)
            s.time_tic(self.dt)

        """링크 여부 확인"""
        for g in self.ground_relays.values():
            g.time_tic(self.dt)
            sats = (self.satellites[node_id] for node_id in g.connected_sats)
            for s in sats:
                if (s.node_id, g.node_id) not in self.disconnect_cache and not s.is_visible(g.latitude_deg, g.longitude_deg):
                    print(f"satellite {s.node_id} and {g.node_id} is disconnected")
                    self.disconnect_cache.add((s.node_id, g.node_id))
                    self.disconnect_cache.add((g.node_id, s.node_id))
                    failed += s.trash_packets(g.node_id)
                    failed += g.trash_packets(s.node_id)

        return failed


    def _log_results_to_csv(self):
        """처리 완료된 패킷들의 결과를 CSV 파일에 기록합니다."""
        rows = []
        self.ended += len(self.results)
        while self.results:
            packet = self.results.pop(0)

            self.flow_recorder.record_packet_outcome(packet)

            # ... (기존의 결과 row 생성 로직) ...
            common_data = [packet.start_at, packet.source, packet.destination, len(packet.result),
                           packet.initial_length, len(packet.detour_at), packet.detour_at, packet.cross_count,
                           packet.result,
                           sum(packet.queuing_delays) + packet.propagation_delays + packet.transmission_delay,
                           packet.queuing_delays, sum(packet.queuing_delays), packet.propagation_delays,
                           packet.transmission_delay, packet.detour_mode]
            if packet.success:
                drop_data = [packet.state, None, None, None, None, packet.ttl]
            else:
                drop_data = [packet.state, packet.dropped_node, packet.dropped_direction, packet.ended_lat,
                             packet.ended_lon, packet.ttl]
            data_if = [None, None, None]
            row = common_data + drop_data + data_if
            rows.append(row)
        csv_write(rows, self.filepath, self.filename)
        if self.t != 0 and self.t % 250 == 0:
            print(f"\n--- Simulation Processing ({self.t / self.steps})---")
            print(f"algorithm: {self.algorithm}")
            print(f"Generation Rate: {self.generation_rate} Mbps")
            print(f"Generated: {self.generated_count}")
            print(f"Ended:     {self.ended}")
            print(f"Succeeded: {self.succeeded}")
            print(f"Failed:    {self.fail_cnt}")
            print(f"Dropped:   {self.drop_cnt}")
            print("--------------------------")

    def _log_summary(self):
        """시뮬레이션 최종 요약 정보를 기록합니다."""
        print("\n--- Simulation Summary ---")
        print(f"algorithm: {self.algorithm}")
        print(f"Generation Rate: {self.generation_rate} Mbps")
        print(f"Generated: {self.generated_count}")
        print(f"Ended:     {self.ended}")
        print(f"Succeeded: {self.succeeded}")
        print(f"Failed:    {self.fail_cnt}")
        print(f"Dropped:   {self.drop_cnt}")
        print("--------------------------")
        now = datetime.now()
        memo = "prop ver.1, num of flows = 3000, isl=2.5G, gsl=1.5/2G"
        summary_info = [[now.strftime('%Y-%m-%d %H:%M:%S'), self.total_time, self.generation_rate, self.generated_count,
                         self.succeeded, self.fail_cnt, self.drop_cnt, self.generated_count - self.ended, memo]]
        csv_write(summary_info, self.filepath, "summary.csv")

    def run(self):
        """시뮬레이션 전체를 실행합니다."""
        # 메인 타임 루프ㄴ
        for _ in tqdm(range(self.steps), desc=f"{self.algorithm} Simulating for {self.generation_rate} Mbps"):
            self.update_graph_and_generate_packet()

            self.failed += self.transmit_phase()
            self.dropped += self.drop_phase()
            success, fail = self.routing_phase()

            self.failed += fail
            self.results += success
            self.results += self.fail_drop_process_phase()

            self._log_results_to_csv()

            self.t += self.dt
            self.failed += self.update_satellite_and_links()


        print("Traffic generation finished. Writing the remaining results...")
        self.packet_generation_mode = False
        # 남은 패킷 처리 루프
        while self.generated_count > self.ended:
            self.update_graph_and_generate_packet()

            self.failed += self.transmit_phase()
            self.dropped += self.drop_phase()
            success, fail = self.routing_phase()

            self.failed += fail
            self.results += success
            self.results += self.fail_drop_process_phase()

            self._log_results_to_csv()

            if self.t % 25 == 0:
                print(f"Time: {self.t}ms, Ended: {self.ended}/{self.generated_count}")

            self.update_satellite_and_links()
            self.t += self.dt

        self.flow_recorder.generate_report(self.filepath, self.generation_rate, self.total_time)
        self._log_summary()