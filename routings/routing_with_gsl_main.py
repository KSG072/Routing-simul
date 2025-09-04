#python 3.11.13
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

def update_rtpg(rtpg, satellites, ground_relays, mapper, only_isl=False):
    sat_region_indices = mapper.batch_map(satellites)

    # 위성 등록
    for sat, region in zip(satellites, sat_region_indices):
        sat.connected_grounds = []
        rtpg.add_satellite(sat, region)
        sat.region = region

    # Ground Relay 등록
    for gr in ground_relays:
        gr.connected_sats = []
        rtpg.add_relay(gr, (gr.region_asc, gr.region_desc), (gr.search_regions_asc, gr.search_regions_desc))

    rtpg.connect_isl_links()
    if only_isl:
        rtpg.connect_ground_links_for_only_isl()
    else:
        rtpg.connect_ground_links()

    return rtpg

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
    filepath = "../results"
    IF_ISL = True

    # GSL O
    for genertation_rate in GENERATION_RATE_LIST:
        filename = "result_" + str(genertation_rate) + ".csv"
        csv_create(header, filepath, filename)
        traffic_schedule_path = f'../parameters/traffic/events_{genertation_rate}Mbps.csv'
        relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
        results = []
        failed = []
        dropped = []
        generated_count = 0

        dt = TIME_SLOT  # 1 ms
        total_time = TOTAL_TIME  # second
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
        traffic_schedule = load_event_schedule(traffic_schedule_path)
        # users = generate_users(start_idx=0, total_count=total_users)
        # users = generate_cities(start_idx=0)

        for gr in ground_relays.values():
            gr = prepare_node_routing_metadata(gr, mapper, 550)
        # for user in users.values():
        #     user = prepare_node_routing_metadata(user, mapper, 550)

        """초기 RTPG 생성"""
        rtpg = update_rtpg(RTPGGraph(N=N, M=M, F=F), satellites.values(), ground_relays.values(), mapper)
        rtpg.integrity_check()
        """ISL로 갔더라면? 버전"""
        if IF_ISL:
            rtpg_isl = update_rtpg(RTPGGraph(N=N, M=M, F=F), satellites.values(), ground_relays.values(), mapper, only_isl=True)
            rtpg_isl.integrity_check()

        """ =============================
                 Time loop 시작
        =============================="""
        ended = 0
        succeeded = 0
        drop_cnt = 0
        fail_cnt = 0
        disconnect_occur = False
        disconnect_pair = []
        sat_id_list = list(range(1584))

        """시뮬레이션 타임루프"""
        for i in tqdm(range(steps)):
            t = i * dt
            """RTPG 업데이트"""
            if t % 600 == 0 and t != 0:
                for s in satellites.values():
                    s.update_lat_lon_for_RTPG()
                rtpg.reset_graph()
                rtpg = update_rtpg(rtpg, satellites.values(), ground_relays.values(), mapper)
                if IF_ISL:
                    rtpg_isl.reset_graph()
                    rtpg_isl = update_rtpg(rtpg_isl, satellites.values(), ground_relays.values(), mapper, only_isl=True)
                """끊긴 링크에 대한 처리 필요
                    1. trash_packets() 메서드 호출
                    - trash_packets(): 그래프 업데이트로 인해서 더 이상 key node가 아님에도, 패킷을 받거나 해당 방향으로 가는 큐의 패킷을 모두 버림
                """

                for satellite in satellites.values():
                    if satellite.has_packets():
                        failed += satellite.trash_packets()
                for gr in ground_relays.values():
                    if gr.has_packets():
                        failed += gr.trash_packets()

            if disconnect_occur:
                while disconnect_pair:
                    (s, g) = disconnect_pair.pop()
                    failed += s.trash_packets()
                    failed += g.trash_packets()
                disconnect_occur = False

            """ 패킷 생성
            1. 해당 타임 슬롯에 대해서 패킷을 생성할 사용자를 선택
            2. 신규 패킷 생성
            3. 중계 노드 중 하나를 destination으로 하여 rtpg기반 다익스트라 알고리즘으로 경로 형성. 패킷에 경로 정보 입력
            4. 키노드 추출 및 키노드까지의 최단 홉 거리 (수직+수평)계산은 로직 상 안함
            """
            generated_packets = traffic_schedule.pop(t, [])
            for packet_data in generated_packets:
                (src, dst, num_of_packets) = packet_data
                # qos = choices(range(0, 3), weights=[0.2, 0.3, 0.5], k=num_of_packets)
                # pair = np.random.choice(sat_id_list, 2, replace = False)
                # src, dst = int(pair[0]), int(pair[1])
                paths = get_route_sat_to_sat(rtpg, src, dst, n=num_of_packets)
                """만약에 만약에 지상 안 탔더라면~~~~ 안 타더라도 지연시간 예측한다면~~"""
                if IF_ISL:
                    isl_path, isl_length = get_route_sat_to_sat(rtpg_isl, src, dst, n=1)[0]
                    expected_delay = delay_estimation(paths[0][0], satellites, ground_relays)
                    expected_isl_delay = delay_estimation(isl_path, satellites, ground_relays)
                # """지금 갈 경로도 미리 계산 때려버려잇~~~~~"""
                generated_count += num_of_packets
                for path in paths:
                    new_packet = Packet(t, 0)
                    new_packet.set_path_info(path[0])
                    new_packet.initial_length = path[1]

                    # """cross count = 0 인 jump는 배제"""
                    # if IF_ISL and new_packet.key_nodes:
                    #     key_nodes_id = list(new_packet.key_nodes)
                    #     key_nodes_id = [new_packet.key_node] + key_nodes_id
                    #     key_nodes = deque(satellites[idx] for idx in key_nodes_id)
                    #     cross_count = check_cross_counts(key_nodes)
                    #     if cross_count < 1:
                    #         # print("oh my god what are you doing")
                    #         new_packet.set_path_info(isl_path)
                    #         new_packet.result.pop()
                    #         # new_packet.show_detailed()
                    #         # print("go to ISL route")

                    calculate_hop_distance(new_packet, satellites)
                    satellites[src].storage.append(new_packet)
                    new_packet.last_direction = satellites[src].is_ascending()
                    if IF_ISL:
                        new_packet.expected_isl_delay = expected_isl_delay
                        new_packet.expected_isl_length = isl_length
                        new_packet.expected_delay = expected_delay
            """
            각 패킷은 아래의 속성을 가짐 
            1. remaining_v_hops (key_nodes의 0번째에 도달할 수 있는 최단 수직홉 거리)
            2. remaining_h_hops (최단 수평홉 거리, 목적지는 위와 동일)
            각 속성은 양수일 경우 우/상향, 음수일 경우 좌/하향 방향을 나타냄
            
            시뮬레이션은 아래 세 페이즈로 구성됨

            1. 전송 페이즈: 방향 큐에 있는 패킷을 일괄적으로 다음 홉으로 전송하는 과정 (라우팅 알고리즘은 이미 수행 된 후)
            2. 실패/드롭 페이즈: 전송 후 큐의 용량을 초과한 패킷에 대한 드롭처리
            3. 라우팅 페이즈: 각 방향으로부터 전송 받은 패킷에 대하여 라우팅 알고리즘 수행(방향결정) 후 각 방향 큐에 enqueue
            # 예외처리
            예외: 사용자-위성-지상노드 순으로 순차적으로 전송과 라우팅 페이즈를 진행함에 따라서 드롭되는 패킷에 우선순위가 발생
            처리: 전송 페이즈에서 수신한 패킷을 shuffle 후 라우팅 페이즈에 진입
            """
            """전송 페이즈"""

            for s in satellites.values():
                if s.has_packets():
                    bullets = s.get_packets(dt) # (up, down, left, right, ground, satellite=[])로 보낼 패킷 리스트
                    next_hops = [ # (up, down, left, right, ground, satellite) 방향 다음 홉
                        satellites[s.isl_up], satellites[s.isl_down],
                        satellites[s.isl_left], satellites[s.isl_right],
                        {node_id: ground_relays[node_id] for node_id in s.gsl_down_buffers.keys()}, []
                    ]
                    failed += transfer(bullets, next_hops, s.cartesian_coords)
                else:
                    continue

            for g in ground_relays.values():
                if g.has_packets():
                    bullets = g.get_packets(dt) # (up=[], down=[], left=[], right=[], ground=[], satellite)로 보낼 패킷 리스트
                    next_hops = [ # (up, down, left, right, ground, satellite) 방향 다음 홉
                        [], [],
                        [], [],
                        [], {node_id: satellites[node_id] for node_id in g.gsl_up_buffers.keys()}
                    ]
                    failed += transfer(bullets, next_hops, g.cartesian_coords)
                else:
                    continue

            """드롭 페이즈"""
            """전송 후 각 버퍼 큐에 남은 패킷 중 capacity를 초과한 것에 대한 tail-drop"""

            for s in satellites.values():
                dropped += s.drop_packet()
            if dropped:
                drop_cnt += len(dropped)
            while dropped:
                p = dropped.pop(0)
                end_node = satellites[p.curr]
                p.end(t, 'drop', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                # print(2)
                results.append(p)

            # if t % 100 == 0 and t != 0:
            #     total_load_data = []
            #     orbit_data = []
            #     for s in satellites.values():
            #         if s.orbit_idx != 0 and s.sat_idx_in_orbit % M == 0:
            #             total_load_data.append(orbit_data)
            #             # print(orbit_data)
            #             orbit_data = []
            #         orbit_data.append(s.get_load_status())
            #     total_load_data.append(orbit_data)
            #     load_heatmap(num_of_generated_packets, t, N, M, total_load_data)
            """라우팅 방향 결정 & 큐 삽입 페이즈"""
            """
            사용자:
            1. 첫 홉 결정은 이미 결정되어 있음 (로드 고려 X)
            위성 (TMC):
            1. isl 라우팅 (자신이 keynode가 아니거나, 잔여 홉수가 남아있을 경우): 많이 남은 홉 수 우선 라우팅 (다음 위성 로드 고려)
            2. gsl 라우팅 (자신이 keynode일 경우): 보내고자 하는 위성으로 라우팅
              - 자신의 하향링크 버퍼큐에 쌓인 패킷 개수가 threshold를 넘을 경우, 해당 지상노드와 연결된 인접(P +-4) 위성 중 lightest한 위성으로 재라우팅
                -> 키노드 재설정 & 잔여 홉거리 재계산 -> 방향 결정 후, isl 큐에 삽입
            지상 노드 (TMC):
            1. Keynode인 위성으로 전송
              - Keynode 위성으로 가는 큐에 쌓인 패킷의 개수가 threshold를 넘을 경우, 해당 지상 노드와 연결된 P거리 4 이내인 위성 중 가장 덜 쌓인 위성 방향 큐로 삽입
                -> 키노드 재설정 & 잔여 홉거리 재계산 
            """

            for s in satellites.values():
                is_ascending = s.is_ascending()
                while s.storage:
                    packet = s.storage.popleft()
                    if packet.was_on_ground and (packet.last_direction is not is_ascending): # cross count
                        packet.cross_count += 1
                    packet.last_direction = is_ascending

                    if packet.curr == packet.destination: # 도착, success
                        end_node = s
                        packet.end(t, 'success', s.node_id, s.latitude_deg, s.longitude_deg)
                        results.append(packet)
                        succeeded += 1
                        continue

                    if packet.curr == packet.key_node:
                        key_node = packet.next_key_node_id()
                        if (not packet.was_on_ground) or (packet.was_on_ground and packet.curr == key_node): # 위성으로부터 받은 패킷이거나, 지상에서 올라오자마자 내려가야하는 경우
                            """지상으로 갈 수 없는 경우"""
                            if packet.ttl <= 0 or packet.ground_node not in s.gsl_down_buffers.keys():
                                failed.append(packet)
                                continue
                            """지상으로 갈 수 있는 경우"""
                            if packet.was_on_ground: # 올라오자마자 지상으로
                                packet.was_on_ground = False
                                packet.next_key_node_id()
                            # 위성 홉 거친 후 지상으로
                            family = [satellites[node_id] for node_id in ground_relays[packet.ground_node].connected_sats]
                            need_detour, packet, direction = sat_to_ground_forwarding(s, packet, family)
                            if need_detour: # 바로 못가고 다른 key node로 가야할 경우
                                packet.detour_mode = True
                                packet.detour_at.append((packet.curr, direction))
                                packet.key_nodes.appendleft(packet.key_node)
                                packet.set_key_node(direction)  # 새로운 keynode 설정
                                calculate_hop_distance(packet, satellites)
                                # packet.show_detailed()
                            else: # 바로 가는 경우
                                packet.detour_mode = False
                                if packet.ttl <= 0:
                                    failed.append(packet)
                                else:
                                    s.enqueue_packet(direction, packet)
                                continue  # 지상 큐에 삽입 후 다음 패킷 처리
                        else: # 지상으로부터 받은 패킷이고, 다른 위성으로 가야하는 경우
                            calculate_hop_distance(packet, satellites)
                            packet.was_on_ground = False

                    # 단순 위성 포워딩 (잔여 홉 있음)

                    horizontal = satellites[s.isl_left if packet.remaining_h_hops < 0 else s.isl_right]
                    vertical = satellites[s.isl_down if packet.remaining_v_hops < 0 else s.isl_up]
                    """위성-위성 라우팅 알고리즘 적용 부분 (Queuing delay는 여기서 계산됨)"""
                    direction = sat_to_sat_forwarding(s, horizontal, vertical, packet) # 0:up, 1:down, 2:left, 3:right
                    # direction = sat_to_sat_forwarding_d(s, horizontal, vertical, packet) # 0:up, 1:down, 2:left, 3:right
                    if packet.ttl <= 0:
                        failed.append(packet)
                    else:
                        s.enqueue_packet(direction, packet)

            for g in ground_relays.values():
                while g.storage:
                    packet = g.storage.popleft()
                    # if packet.curr == packet.destination: # 도착
                    #     end_node = ground_relays[packet.curr]
                    #     packet.end(t, 'success', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                    #     results.append(packet)
                    #     succeeded += 1
                    # else: # 다시 위성으로
                    if packet.ttl <= 0 or packet.key_node not in g.gsl_up_buffers.keys():
                        failed.append(packet)
                        continue
                    packet.was_on_ground = True
                    packet.next_ground_node_id()
                    family = [satellites[node_id] for node_id in ground_relays[packet.curr].connected_sats]
                    """지상-위성 라우팅 알고리즘 적용 부분"""
                    packet, direction = ground_to_sat_forwarding(g, packet, family)
                    g.enqueue_packet(direction, packet)

            """실패 페이즈"""
            """경로 설정 과정에서 위성-지상 노드 링크 끊김으로 인한 전송 실패"""
            """TTL 만료로 인한 전송 실패"""

            if failed:
                fail_cnt += len(failed)
            while failed:
                # print(failed)
                p = failed.pop(0)
                if isinstance(p.curr, int):
                    end_node = satellites[p.curr]
                else:
                    end_node = ground_relays[p.curr]

                if p.ttl > 0:
                    p.end(t, 'failure', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                else:
                    p.end(t, 'expired', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                # print(end_node.connected_sats)
                # p.show_detailed()
                results.append(p)

            """위성 공전"""
            for s in satellites.values():
                s.update_position(omega_s, dt)
                s.time_tic(dt)

            """링크 여부 확인"""
            for g in ground_relays.values():
                g.time_tic(dt)
                sats = (satellites[node_id] for node_id in g.connected_sats)
                for s in sats:
                    if not s.is_visible(g.latitude_deg, g.longitude_deg):
                        disconnect_occur = True
                        print(f"satellite {s.node_id} and {g.node_id} is disconnected")
                        disconnect_pair.append((s, g))


            rows = []
            ended += len(results)
            while results:
                packet = results.pop(0)
                common_data = [packet.start_at, packet.source, packet.destination, len(packet.result), packet.initial_length, len(packet.detour_at), packet.detour_at, packet.cross_count, packet.result,
                               sum(packet.queuing_delays)+packet.propagation_delays+packet.transmission_delay, packet.queuing_delays, sum(packet.queuing_delays), packet.propagation_delays, packet.transmission_delay, packet.detour_mode]
                if packet.success:
                    drop_data = [packet.state, None, None, None, None, packet.ttl]
                else:
                    drop_data = [packet.state, packet.dropped_node, packet.dropped_direction, packet.ended_lat, packet.ended_lon, packet.ttl]
                if IF_ISL:
                    data_if = [packet.expected_delay, packet.expected_isl_delay, packet.expected_isl_length]
                else:
                    data_if = [None, None, None]
                row = common_data + drop_data + data_if
                rows.append(row)
            csv_write(rows, filepath, filename)
            # generation rate에 대한 for문 종료
        #나머지 데이터 입력
        rows = []
        ended += len(results)
        while results:
            packet = results.pop(0)
            common_data = [packet.start_at, packet.source, packet.destination, len(packet.result), packet.initial_length,
                           len(packet.detour_at), packet.detour_at, packet.cross_count, packet.result,
                           sum(packet.queuing_delays) + packet.propagation_delays + packet.transmission_delay,
                           packet.queuing_delays, sum(packet.queuing_delays), packet.propagation_delays,
                           packet.transmission_delay, packet.detour_mode]
            if packet.success:
                drop_data = [packet.state, None, None, None, None, packet.ttl]
            else:
                drop_data = [packet.state, packet.dropped_node, packet.dropped_direction, packet.ended_lat,
                             packet.ended_lon, packet.ttl]
            if IF_ISL:
                data_if = [packet.expected_delay, packet.expected_isl_delay, packet.expected_isl_length]
            else:
                data_if = [None, None, None]
            row = common_data + drop_data + data_if
            rows.append(row)
        csv_write(rows, filepath, filename)

#         """ 잔여 패킷 별도 csv 저장 """
#         rest_packet_filename = f"limited_Q_with_GSL_{genertation_rate}_rest.csv"
#         rest_packet_file_header = [
#         "Time (ms)", "source", "destination", "current", "location", "Path Length", "Detour counts", "Detour log",
#         "cross counts", "result", "e2e delay", "Queuing delays", "Queuing Delay", "Propagation Delay", "Transmission Delay",
#         "TTL"]
#         csv_create(rest_packet_file_header, filepath, rest_packet_filename)
#         for node in satellites.values():
#             rows = []
#             rest = node.get_all_packets()
#             for location, packets in rest.items():
#                 for packet in packets:
#                     data = [packet.start_at, packet.source, packet.destination, packet.curr, location, len(packet.result),
#                             len(packet.detour_at), packet.detour_at, packet.cross_count, packet.result,
#                             sum(packet.queuing_delays) + packet.propagation_delays + packet.transmission_delay,
#                             packet.queuing_delays, sum(packet.queuing_delays), packet.propagation_delays,
#                             packet.transmission_delay, packet.ttl]
#                     # print(data)
#                     rows.append(data)
#             csv_write(rows, filepath, rest_packet_filename)
#         for node in ground_relays.values():
#             rows = []
#             rest = node.get_all_packets()
#             for location, packets in rest.items():
#                 for packet in packets:
#                     data = [packet.start_at, packet.source, packet.destination, packet.curr, location, len(packet.result),
#                             len(packet.detour_at), packet.detour_at, packet.cross_count, packet.result,
#                             sum(packet.queuing_delays) + packet.propagation_delays + packet.transmission_delay,
#                             packet.queuing_delays, sum(packet.queuing_delays), packet.propagation_delays,
#                             packet.transmission_delay, packet.ttl]
# #                     print(data)
#                     rows.append(data)
#             csv_write(rows, filepath, rest_packet_filename)


        print("\n--- Simulation Summary ---")
        print(f"Generated: {generated_count}")
        print(f"Ended:     {ended}")
        print(f"Succeeded: {succeeded}")
        print(f"Failed:    {fail_cnt}")
        print(f"Dropped:   {drop_cnt}")
        print("--------------------------")
        now = datetime.now()
        memo = "TTL64, if isl 비교용, cross = 0 케이스 포함, Detour mode 구현 후 재실험"
        summary_info = [[now.strftime('%Y-%m-%d %H:%M:%S'), total_time, genertation_rate,generated_count, succeeded, fail_cnt, drop_cnt, generated_count-ended, memo]]
        csv_write(summary_info, filepath, "summary.csv")