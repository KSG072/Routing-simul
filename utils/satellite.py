# === satellite.py ===
import numpy as np

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

        self.speed = speed

        self.phase_rad = None
        self.lon_asc_node_rad = None
        self.latitude_deg = None
        self.longitude_deg = None

        self.region = None
        self.marker = None

    def update_position(self, omega_s, dt):
        self.phase_rad = (self.phase_rad + omega_s * dt * self.speed) % (2 * np.pi)
        self.set_position_from_phase(self.phase_rad, self.lon_asc_node_rad)

    def set_position(self, lat, lon):
        self.latitude_deg = lat
        self.longitude_deg = lon

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

    def is_visible(self, node, min_elev_deg=10):
        """
        Computes whether this satellite is visible from a ground node (relay/user).
        The node must have `.latitude` and `.longitude`.
        """
        r_sat = self.get_cartesian_coords()

        lat = node.latitude
        lon = node.longitude
        r_node = self._latlon_to_ecef(lat, lon, 0.0, self.earth_radius)

        vec = r_sat - r_node
        dot = np.dot(vec, r_node)
        elev_rad = np.arcsin(dot / (np.linalg.norm(vec) * np.linalg.norm(r_node)))
        elev_deg = np.rad2deg(elev_rad)

        return elev_deg >= min_elev_deg

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
        return np.cos(self.phase_rad) >= -1e-8

    def __repr__(self):
        return f"Satellite(id={self.node_id}, lat={self.latitude_deg:.2f}, lon={self.longitude_deg:.2f})"
