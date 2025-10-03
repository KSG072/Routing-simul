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

from routings.dijkstra import sat_to_sat_forwarding_d

os.environ["PYCHARM_DISPLAY"] = "none"
_GROUND_RE = re.compile(r'^[A-Za-z]+-\d+$')  # 지상 노드: 영문자-숫자

from parameters.PARAMS import *
from routings.packet import Packet
from routings.sota import sat_to_sat_forwarding, sat_to_ground_forwarding, ground_to_sat_forwarding
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


# def get_route(rtpg, user, ground_relays, n=1):
#     rtpg.add_user(user, (user.region_asc, user.region_desc), (user.search_regions_asc, user.search_regions_desc),
#                   user.is_in_city)
#
#     # User만을 위한 연결만 수행
#     rtpg.connect_user_links(user)
#
#     # 최단 경로 찾기
#     routes = []
#     for _ in range(n):
#         dst_id = random.choice(list(ground_relays.keys()))
#         user.destination = dst_id
#
#         route = rtpg.dijkstra_shortest_path(source_id=user.node_id, target_id=ground_relays[user.destination].node_id)
#         routes.append(route)
#
#     rtpg.G.remove_node(user.node_id)
#
#     return routes

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
                            if detail_direction not in disconnected and detail_direction in next_hops[direction]:
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


if __name__ == '__main__':
    # header = [
    #     "Time (ms)", "User ID", "Destination Relay ID", "QoS", "Path Length", "ISL Path Length", "Detour counts", "cross counts", "result", "e2e delay", "expected delay(result)", "expected delay(isl)", "Queuing delays", "Queuing Delay", "Propagation Delay", "Transmission Delay",
    #     "Status", "Drop Location", "Drop Direction", "Drop Latitude", "Drop Longitude"
    # ]
    """source ID, Dest ID  모든 파일에서 바꾸기"""
    header = [
        "Time (ms)", "source", "destination", "Path Length", "expected length", "Detour counts", "Detour log",
        "cross counts", "result", "e2e delay", "Queuing delays", "Queuing Delay", "Propagation Delay", "Transmission Delay", "Detour mode",
        "Status", "Drop Location", "Drop Direction", "Drop Latitude", "Drop Longitude", "TTL", "expected delay(result)", "expected delay(isl)","ISL Path Length"
    ]
    filepath = "../results/proposed ver1"
    IF_ISL = False

    # GSL O
    for genertation_rate in [1]:
        traffic_schedule_path = f'../parameters/uneven traffic (3000flows)/events_{genertation_rate}Mbps.csv'
        relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
        results = []
        failed = []
        dropped = []
        generated_count = 0

        dt = 10  # 1 ms
        total_time = 10000  # second
        # total_time = 20.51 # seconds
        steps = int(total_time / dt)
        packet_scope = None

        T_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
        omega_s = 2 * np.pi / T_ms  # delta phase (deg)

        """필요한 모듈 초기화"""
        mapper = RTPGMapper(N, M, F, inclination_deg)

        """Constellation & 지상 노드 생성"""
        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites = constellation.get_all_satellites() # dictionary

        ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
        # traffic_schedule = load_event_schedule(traffic_schedule_path, total_time)
        # users = generate_users(start_idx=0, total_count=total_users)
        # users = generate_cities(start_idx=0)

        for gr in ground_relays.values():
            gr = prepare_node_routing_metadata(gr, mapper, 550)
        # for user in users.values():
        #     user = prepare_node_routing_metadata(user, mapper, 550)

        """초기 RTPG 생성"""
        rtpg = RTPGGraph(N=N, M=M, F=F)
        sat_region_indices = mapper.batch_map(satellites.values())
        rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)
        rtpg.integrity_check()

        """ =============================
                 Time loop 시작
        =============================="""

        disconnect_occur = False
        disconnect_pair = []
        total_added, total_removed = set(), set()
        initial_gsl_edge = set(
    (e[0], e[1]) for e in rtpg.G.edges(data=True) if e[2].get('type') == 'gsl_up'
) | set(
    (e[0], e[1]) for e in rtpg.G.edges(data=True) if e[2].get('type') == 'gsl_down'
)
        print(f"initial gsl edges: {len(initial_gsl_edge)}")
        """시뮬레이션 타임루프"""
        for i in range(steps):
            t = i * dt
            """RTPG 업데이트"""

            if t % 600 == 0 and t != 0:
                # --- 변경 사항 확인 코드 시작 ---
                print(f"\n[t={t}ms] RTPG 업데이트 시작. 엣지 변경 사항을 확인합니다.")
                # 1. 업데이트 전 엣지 목록 저장
                old_edges = set(rtpg.G.edges())
                # --- 변경 사항 확인 코드 종료 ---

                for s in satellites.values():
                    s.update_lat_lon_for_RTPG()
                rtpg.reset_graph()
                sat_region_indices = mapper.batch_map(satellites.values())
                rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)

                # --- 변경 사항 확인 코드 시작 ---
                # 2. 업데이트 후 엣지 목록 저장
                new_edges = set(rtpg.G.edges())

                # 3. 변경된 엣지 계산 및 출력
                added_edges = new_edges - old_edges
                total_added = total_added | added_edges
                removed_edges = old_edges - new_edges
                total_removed = total_removed | removed_edges

                if added_edges:
                    print(f"  >> 추가된 엣지 ({len(added_edges)}개): {added_edges}")
                else:
                    print("  >> 추가된 엣지가 없습니다.")

                if removed_edges:
                    print(f"  >> 삭제된 엣지 ({len(removed_edges)}개): {removed_edges}")
                else:
                    print("  >> 삭제된 엣지가 없습니다.")


            """위성 공전"""
            for s in satellites.values():
                s.update_position(omega_s, dt)
                # s.time_tic(dt)

            # """링크 여부 확인"""
            # for g in ground_relays.values():
            #     # g.time_tic(dt)
            #     sats = (satellites[node_id] for node_id in g.connected_sats)
            #     for s in sats:
            #         if (s, g) not in disconnect_pair and not s.is_visible(g.latitude_deg, g.longitude_deg):
            #             disconnect_occur = True
            #             print(f"satellite {s.node_id} and {g.node_id} is disconnected")
            #             disconnect_pair.append((s, g))
        remaining_edge = initial_gsl_edge & (set(
    (e[0], e[1]) for e in rtpg.G.edges(data=True) if e[2].get('type') == 'gsl_up'
) | set(
    (e[0], e[1]) for e in rtpg.G.edges(data=True) if e[2].get('type') == 'gsl_down'
))

        print(f"total {len(total_added)} edges added, total {len(total_removed)} edges removed during simulation")
        # print(f"added edges: {total_added}")
        # print(f"removed edges: {total_removed}")
        # print(f"add and remove edges: {total_added & total_removed}")
        # for e in total_added & total_removed:
        #     print(e)
        # print(f"remaining edges: {len(remaining_edge)}")
        # for e in remaining_edge:
        #     print(e)
