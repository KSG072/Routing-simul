# === orbit.py ===
class Orbit:
    def __init__(self, orbit_idx, lon_asc_node_rad):
        self.orbit_idx = orbit_idx
        self.lon_asc_node_rad = lon_asc_node_rad
        self.satellites = []  # List of Satellite objects

    def add_satellite(self, sat):
        self.satellites.append(sat)

    def __repr__(self):
        return f"Orbit {self.orbit_idx} with {len(self.satellites)} satellites"