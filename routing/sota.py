from parameters.PARAMS import SENDING_BUFFER_QUEUE_LASER_PACKETS, PACKET_SIZE_BITS, ISL_RATE_LASER, SMOOTHING_FACTORS, \
    TAU, SGL_KA_DOWNLINK, MAX_SAT_P_DIFF, SGL_KA_UPLINK


def calculate_sat_load_status(sat, h_dir, v_dir):
    if h_dir > 0:
        if v_dir > 0: # 우상향
            return (sat.isl_up_buffer.get_load_status() + sat.isl_right_buffer.get_load_status()) / 2
        else: # 우하향
            return (sat.isl_down_buffer.get_load_status() + sat.isl_right_buffer.get_load_status()) / 2
    else:
        if v_dir > 0: # 좌상향
            return (sat.isl_up_buffer.get_load_status() + sat.isl_left_buffer.get_load_status()) / 2
        else: # 좌하향
            return (sat.isl_down_buffer.get_load_status() + sat.isl_left_buffer.get_load_status()) / 2

def sat_to_sat_forwarding(cur, hor, ver, packet):
    Bs = SENDING_BUFFER_QUEUE_LASER_PACKETS
    n1 = SMOOTHING_FACTORS[0]
    r_v_hop, r_h_hop = packet.remaining_v_hops, packet.remaining_h_hops

    if r_h_hop == 0:
        if r_v_hop < 0:
            return 1
        else:
            return 0
    elif r_v_hop == 0:
        if r_v_hop < 0:
            return 2
        else:
            return 3
    else:
        q_next_h = calculate_sat_load_status(hor, r_h_hop, r_v_hop)
        q_next_v = calculate_sat_load_status(ver, r_h_hop, r_v_hop)
        # 0:up, 1:down, 2:left, 3:right
        if r_h_hop > 0:
            horizontal = 3
            q_self_h = cur.isl_right_buffer.size
        else:
            horizontal = 2
            q_self_h = cur.isl_left_buffer.size
        if r_v_hop > 0:
            vertical = 0
            q_self_v = cur.isl_up_buffer.size
        else:
            vertical = 1
            q_self_v = cur.isl_down_buffer.size
        t_h = q_self_h + q_next_h
        t_v = q_self_v + q_next_v
        threshold = n1 * (2*Bs - max(t_v, t_h))

        if abs(r_v_hop) > abs(r_h_hop): # 수직 우선
            if t_h - t_v <= threshold:
                best_direction = vertical
            else:
                best_direction = horizontal
        else: # 수평 우선
            if t_v - t_h <= threshold:
                best_direction = horizontal
            else:
                best_direction = vertical

        if best_direction == vertical:
            packet.queuing_delays.append((q_self_v*PACKET_SIZE_BITS)/(TAU*ISL_RATE_LASER))
        else:
            packet.queuing_delays.append((q_self_h*PACKET_SIZE_BITS)/(TAU*ISL_RATE_LASER))

        return best_direction

def sat_to_ground_forwarding(cur, packet, family):
    R_ISL = ISL_RATE_LASER
    R_down = SGL_KA_DOWNLINK
    η2 = SMOOTHING_FACTORS[1]
    η3 = SMOOTHING_FACTORS[2]
    n0 = MAX_SAT_P_DIFF

    psi = η2 * TAU * R_ISL + η3 * TAU * R_down
    try:
        q_self_g = cur.gsl_down_buffers[packet.ground_node].size * PACKET_SIZE_BITS
    except KeyError:
        print("key error")
        packet.show_detailed()
        exit(0)

    if q_self_g <= psi:
        packet.queuing_delays.append(q_self_g*PACKET_SIZE_BITS/(TAU*R_down))
        return False, packet, packet.ground_node
    else:
        candidates = [sat for sat in family if abs(cur.orbit_idx - sat.orbit_idx) <= n0]
        detour_key_node = min(candidates, key=lambda sat: sat.gsl_down_buffers[packet.ground_node].get_load_status())
        return True, packet, detour_key_node.node_id # 지상노드로 전송 안함


def ground_to_sat_forwarding(cur, packet, family):
    R_ISL = ISL_RATE_LASER
    R_up = SGL_KA_UPLINK
    η4 = SMOOTHING_FACTORS[3]
    η5 = SMOOTHING_FACTORS[4]
    n0 = MAX_SAT_P_DIFF

    psi = η4 * TAU * R_ISL + η5 * TAU * R_up
    try:
        q_self_g = cur.gsl_up_buffers[packet.key_node].size
    except KeyError:
        print("d")
        packet.show_detailed()

    if q_self_g <= psi:
        packet.queuing_delays.append(q_self_g*PACKET_SIZE_BITS/(TAU*R_up))
        return packet, packet.key_node
    else:
        cur = next((sat for sat in family if packet.key_node == sat.node_id), None)
        if cur is not None:
            candidates = [sat for sat in family if abs(cur.orbit_idx - sat.orbit_idx) <= n0]
            detour_key_node = min(candidates, key=lambda sat: cur.gsl_up_buffers[sat.node_id].size)
            packet.set_key_node(detour_key_node.node_id)
            packet.queuing_delays.append(cur.gsl_up_buffers[detour_key_node.node_id].size*PACKET_SIZE_BITS/(TAU*R_up))
        return packet, packet.key_node.node_id

