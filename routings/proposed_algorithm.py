import math
from collections import defaultdict

from parameters.PARAMS import PACKET_SIZE_BITS, ISL_RATE_LASER, TAU
from proposed_routingtable import import_routing_tables_from_csv_json

class RoutingTable:
    def __init__(self):
        self.routing_table = {}
        self.table_len = 0

    def load_routing_table(self, rate, time):
        self.routing_table[time] = import_routing_tables_from_csv_json(rate, time)
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
            print("what!!?!?!?!?!?!?!?")
            return None
        return direction

    def ground_forwarding(self, cur, packet):
        key = self.get_table_key(packet.start_at)
        table = self.routing_table[key]
        direction = table[cur.node_id].get((packet.source, packet.destination), None)
        if direction is None:
            print("what!!?!?!?!?!?!?!?")
            return None
        return direction



