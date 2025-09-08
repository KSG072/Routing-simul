# === ground_relay_node.py ===
import numpy as np
from random import sample
from routings.buffer_queue import Buffer
from collections import deque

class GroundRelayNode:
    def __init__(self, node_id, latitude, longitude, continent, shape, earth_radius_km=6371):
        self.node_id = node_id
        self.latitude_deg = latitude
        self.longitude_deg = longitude
        self.continent = continent      # color group
        self.earth_radius = earth_radius_km
        self.cartesian_coords = self.get_cartesian_coords()

        self.region_asc = None
        self.region_desc = None
        self.original_region_asc = None
        self.original_region_desc = None
        self.search_regions_asc = None
        self.search_regions_desc = None

        self.marker = None
        self.marker_shape = shape

        #routing
        self.receiving = []
        self.storage = deque()
        self.connected_sats = []
        self.gsl_up_buffers = {}
        self.disconnected = []


    def get_position(self):
        return (self.latitude_deg, self.longitude_deg)

    def get_cartesian_coords(self):
        lat_rad = np.deg2rad(self.latitude_deg)
        lon_rad = np.deg2rad(self.longitude_deg)
        r = self.earth_radius
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def link_to_sat(self, sat_id):
        self.connected_sats.append(sat_id)
        if sat_id not in self.gsl_up_buffers.keys():
            new_buffer = Buffer('up')
            self.gsl_up_buffers[sat_id] = new_buffer

    def receive_packet(self, packet):
        self.receiving.append(packet)

    def time_tic(self, dt):
        arrived = deque()
        for p in self.receiving:
            p.remaining_prop_delay -= dt
            if p.remaining_prop_delay < 0:
                arrived.append(p)
        arrived = deque(sorted(arrived, key=lambda p: p.remaining_prop_delay))
        for p in arrived:
            self.receiving.remove(p)
        self.storage.extend(arrived)

    def get_packets(self, dt):
        gsl_packets = []
        temp = {}
        for direction, buffer in self.gsl_up_buffers.items():
            temp[direction] = buffer.dequeue_sequences(dt)
        gsl_packets.append(temp)

        return [[], [], [], [], [], gsl_packets]

    def trash_packets(self):
        disconnected = [sat_id for sat_id in self.gsl_up_buffers.keys() if sat_id not in self.connected_sats]
        if self.disconnected:
            disconnected += self.disconnected
        trash = []
        for direction in disconnected:
            disconnected_buffer = self.gsl_up_buffers.pop(direction)
            trash += list(disconnected_buffer.buffer)
        self.disconnected = []

        return trash

    def has_packets(self):
        for buffer in self.gsl_up_buffers.values():
            if not buffer.is_empty():
                return True
        return False

    def enqueue_packet(self, direction, packet):
        packet.ttl -= 1
        packet.dropped_direction = direction
        self.gsl_up_buffers[direction].enqueue(packet)

    def get_all_packets(self):
        in_queue = []
        for buffer_obj in self.gsl_up_buffers.values():
            in_queue += list(buffer_obj.buffer)

        return {
            "receiving": self.receiving,
            "on storage": self.storage,
            "in queue": in_queue
        }

    def total_buffer_load(self):
        in_queue = 0
        for buffer_obj in self.gsl_up_buffers.values():
            in_queue += buffer_obj.size
        return in_queue

    def __repr__(self):
        return (f"GroundRelayNode(id={self.node_id}, lat={self.latitude_deg:.2f}, "
                f"lon={self.longitude_deg:.2f}, continent='{self.continent}')")