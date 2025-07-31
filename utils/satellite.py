# === satellite.py ===
from random import sample
from enum import global_str

import numpy as np
from numpy import pi
from collections import deque
from parameters.PARAMS import MIN_ELEVATION

from routing.buffer_queue import Buffer

class Satellite:
    def __init__(self, node_id, orbit_idx, sat_idx_in_orbit, inclination_rad, earth_radius_km=6371, altitude_km=550, speed=30):
        self.node_id = node_id
        self.orbit_idx = orbit_idx
        self.sat_idx_in_orbit = sat_idx_in_orbit
        self.queue = []

        self.inclination = inclination_rad
        self.altitude = altitude_km
        self.earth_radius = earth_radius_km
        self.orbital_radius = self.earth_radius + self.altitude
        self.cartesian_coords = None

        self.phase_rad = None
        self.lon_asc_node_rad = None
        self.latitude_deg = None
        self.longitude_deg = None

        self.region = None

        #visualization
        self.speed = speed
        self.region = None
        self.marker = None

        self.isl_up = None
        self.isl_down = None
        self.isl_left = None
        self.isl_right = None
        self.connected_grounds = []
        self.disconnected = set()

        #routing buffer queue (forwarding 기능은 외부에서 구현 예정)
        self.isl_up_buffer = Buffer('isl')
        self.isl_down_buffer = Buffer('isl')
        self.isl_left_buffer = Buffer('isl')
        self.isl_right_buffer = Buffer('isl')
        self.gsl_down_buffers = {}
        self.storage = deque()
        self.receiving = []


    def update_position(self, omega_s, dt):
        self.phase_rad = (self.phase_rad + omega_s * dt) % (2 * np.pi)
        self.set_position_from_phase(self.phase_rad, self.lon_asc_node_rad)
        self.cartesian_coords = self.get_cartesian_coords()

    def set_position(self, lat, lon):
        self.latitude_deg = lat
        self.longitude_deg = lon

    def link_to_ground(self, relay_id):
        self.connected_grounds.append(relay_id)
        new_buffer = Buffer('down')
        self.gsl_down_buffers[relay_id] = new_buffer

    def get_packets(self, dt):
        gsl_packets = []
        temp = {}
        for direction, buffer in self.gsl_down_buffers.items():
            temp[direction] = buffer.dequeue_sequences(dt)
        gsl_packets.append(temp)
        result = [self.isl_up_buffer.dequeue_sequences(dt), self.isl_down_buffer.dequeue_sequences(dt),
                  self.isl_left_buffer.dequeue_sequences(dt), self.isl_right_buffer.dequeue_sequences(dt),
                  gsl_packets, []]
        return result

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

    def has_packets(self):
        for buffer in self.gsl_down_buffers.values():
            if not buffer.is_empty():
                return True
        return not (self.isl_up_buffer.is_empty() and self.isl_down_buffer.is_empty()
                    and self.isl_left_buffer.is_empty() and self.isl_right_buffer.is_empty())

    def drop_packet(self):
        dropped = []
        dropped += self.isl_up_buffer.drop()
        dropped += self.isl_down_buffer.drop()
        dropped += self.isl_left_buffer.drop()
        dropped += self.isl_right_buffer.drop()
        for buffer in self.gsl_down_buffers.values():
            dropped += buffer.drop()
        # if dropped:
        #     print(dropped)
        return dropped

    def enqueue_packet(self, direction, packet):
        if direction == 0: # isl up
            self.isl_up_buffer.enqueue(packet)
        elif direction == 1: # isl down
            self.isl_down_buffer.enqueue(packet)
        elif direction == 2: # isl left
            self.isl_left_buffer.enqueue(packet)
        elif direction == 3: # isl right
            self.isl_right_buffer.enqueue(packet)
        else:
            self.gsl_down_buffers[direction].enqueue(packet)
            pass

    def set_position_from_phase(self, phase_rad, lon_asc_node_rad):
        self.phase_rad = phase_rad
        self.lon_asc_node_rad = lon_asc_node_rad
        lat = np.arcsin(np.sin(self.inclination) * np.sin(phase_rad))
        lon = lon_asc_node_rad + np.arctan2(np.cos(self.inclination) * np.sin(phase_rad), np.cos(phase_rad))

        self.latitude_deg = np.rad2deg(lat)
        self.longitude_deg = np.rad2deg(lon % (2 * np.pi))

    def get_position(self):
        return self.latitude_deg, self.longitude_deg

    def get_cartesian_coords(self):
        lat_rad = np.deg2rad(self.latitude_deg)
        lon_rad = np.deg2rad(self.longitude_deg)
        r = self.earth_radius + self.altitude
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def get_elevation_angle(self, node_lat, node_lon):
        r_sat = self.cartesian_coords
        r_node = self._latlon_to_ecef(node_lat, node_lon, 0.0, self.earth_radius)

        vec = r_sat - r_node
        dot = np.dot(vec, r_node)
        elev_rad = np.arcsin(dot / (np.linalg.norm(vec) * np.linalg.norm(r_node)))
        elev_deg = np.rad2deg(elev_rad)

        return elev_deg

    def is_visible(self, node_lat, node_lon):
        """
        Computes whether this satellite is visible from a ground node (relay/user).
        The node must have `.latitude` and `.longitude`.
        """
        r_sat = self.cartesian_coords
        r_node = self._latlon_to_ecef(node_lat, node_lon, 0.0, self.earth_radius)

        vec = r_sat - r_node
        dot = np.dot(vec, r_node)
        elev_rad = np.arcsin(dot / (np.linalg.norm(vec) * np.linalg.norm(r_node)))
        elev_deg = np.rad2deg(elev_rad)

        return elev_deg >= MIN_ELEVATION

    @staticmethod
    def _latlon_to_ecef(lat_deg, lon_deg, alt_km, re):
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        r = re + alt_km
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def is_ascending(self):
        return np.cos(self.phase_rad) >= 0

    def storage_shuffle(self):
        self.storage = deque(sample(self.storage, len(self.storage)))

    # def can_enqueue(self, direction, detail_direction):
    #     if direction == 'h':
    #         if detail_direction

    def get_load_status(self):
        return [
            self.isl_up_buffer.size / self.isl_up_buffer.capacity,
            self.isl_down_buffer.size / self.isl_down_buffer.capacity,
            self.isl_left_buffer.size / self.isl_left_buffer.capacity,
            self.isl_right_buffer.size / self.isl_right_buffer.capacity,
        ]



    def __repr__(self):
        return f"Satellite(id={self.node_id}, lat={self.latitude_deg:.2f}, lon={self.longitude_deg:.2f})"
