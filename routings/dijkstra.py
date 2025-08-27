from parameters.PARAMS import SENDING_BUFFER_QUEUE_LASER_PACKETS, PACKET_SIZE_BITS, ISL_RATE_LASER, SMOOTHING_FACTORS, \
    TAU, SGL_KA_DOWNLINK

def get_next_hop(cur, path):
    """
    배열에서 target의 다음 요소를 반환하는 함수
    """
    try:
        idx = path.index(cur)  # target의 위치 찾기
        if idx < len(path) - 1:   # 마지막 요소가 아닐 때
            return path[idx + 1]
        else:
            return None  # 마지막 요소면 다음 값이 없으므로 None 반환
    except ValueError:
        return None  # target이 배열에 없을 때

def sat_to_sat_forwarding_d(cur, hor, ver, packet):
    next_node_id = get_next_hop(cur.node_id, packet.path)

    # 0:up, 1:down, 2:left, 3:right
    if cur.isl_up == next_node_id:
        best_direction = 0
        q_self = cur.isl_up_buffer.size
    elif cur.isl_down == next_node_id:
        best_direction = 1
        q_self = cur.isl_down_buffer.size
    elif cur.isl_left == next_node_id:
        best_direction = 2
        q_self = cur.isl_left_buffer.size
    elif cur.isl_right == next_node_id:
        best_direction = 3
        q_self = cur.isl_right_buffer.size
    else:
        print("what!!!!>?!>?!")
        packet.show_detailed()
        exit(0)

    packet.queuing_delays.append((q_self*PACKET_SIZE_BITS)/(TAU*ISL_RATE_LASER))

    return best_direction