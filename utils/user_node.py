import numpy as np
from collections import deque
from routing.buffer_queue import Buffer

class UserNode:
    def __init__(self, node_id, latitude, longitude, earth_radius_km=6371):
        """
        위도/경도를 기반으로 사용자 노드를 초기화
        """
        self.node_id = node_id
        self.latitude_deg = latitude
        self.longitude_deg = longitude
        self.earth_radius = earth_radius_km
        self.cartesian_coords = self.get_cartesian_coords()

        # RTPG region (asc, desc 각각 1개)
        self.region_asc = None
        self.region_desc = None

        # 주변 탐색 영역 (Polygon)
        self.search_regions_asc = None
        self.search_regions_desc = None

        self.located_in = None
        self.is_in_city = None
        self.destination = None

        self.packet_generation_times = []  # 초 단위 리스트
        self.storage = deque()
        self.connected_sats = []
        self.disconnected = set()

        self.gsl_up_buffers = {}

        # 시각화용 Panda3D marker (필요 시)
        self.marker = None

    def set_userinfo(self, indicator, city_name):
        self.is_in_city = indicator
        self.located_in = city_name

    def get_position(self):
        """
        위도, 경도를 튜플로 반환
        """
        return (self.latitude_deg, self.longitude_deg)

    def link_to_sat(self, sat_id):
        if sat_id not in self.connected_sats:
            self.connected_sats.append(sat_id)
            new_buffer = Buffer('up')
            self.gsl_up_buffers[sat_id] = new_buffer

    def get_cartesian_coords(self):
        """
        구면 좌표계를 3D 카르테시안 좌표로 변환
        """
        lat_rad = np.deg2rad(self.latitude_deg)
        lon_rad = np.deg2rad(self.longitude_deg)
        r = self.earth_radius

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def __repr__(self):
        return (f"UserNode(id={self.node_id}, lat={self.latitude_deg:.2f}, "
                f"lon={self.longitude_deg:.2f})")

    def receive_packet(self, packet):
        self.storage.append(packet)

    def enqueue_packet(self, direction, packet):
        self.gsl_up_buffers[direction].enqueue(packet)


    def get_packets(self, dt):
        gsl_packets = [{}]
        for direction, buffer in self.gsl_up_buffers.items():
            gsl_packets[0][direction] = buffer.dequeue_sequences(dt)

        return [[], [], [], [], [], gsl_packets]

    def has_packets(self):
        for buffer in self.gsl_up_buffers.values():
            if not buffer.is_empty():
                return True
        return False

