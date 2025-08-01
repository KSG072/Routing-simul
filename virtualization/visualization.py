from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import TextNode, VBase4, Point3, Camera, LineSegs
from direct.task import Task

import matplotlib.pyplot as plt

import sys
import numpy as np
import math


class World(DirectObject):
    color_map = {
        'red': (1, 0, 0, 1),
        'green': (0, 1, 0, 1),
        'blue': (0, 0, 1, 1),
        'cyan': (0, 1, 1, 1),
        'magenta': (1, 0, 1, 1),
        'orange': (1.0, 0.65, 0.0, 1.0),
    }

    def __init__(self, base, satellites, ground_relays, mapper, N, M, F, altitude_km, inclination_deg, earth_radius_km, T_s):
        base.disableMouse()

        self.master = base
        self.base = None
        self.render = self.master.render

        self.satellites = satellites
        self.ground_relays = ground_relays
        self.mapper = mapper

        self.N = N
        self.M = M
        self.F = F
        self.altitude_km = altitude_km
        self.radius = earth_radius_km
        self.inclination_deg = inclination_deg
        self.orbit_len = T_s
        self.omega_s = 2 * np.pi / self.orbit_len
        self.speed = 10

        self.sat_size_scale = 0.1
        self.earth_size_scale = 10
        self.orbitscale = self.earth_size_scale * (1 + self.altitude_km / self.radius)
        self.pivots = []

        self.is_paused = False
        self.is_isl = False
        self.sim_time = 0.0

        self.pause_text = OnscreenText(
            text="PAUSE",
            pos=(0, 0),
            scale=0.1,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter,
            mayChange=True
        )
        self.pause_text.hide()

        self.world_cam_np = self.render.attachNewNode(Camera("world_cam"))
        self.world_cam_np.setPos(0, -45, 0)
        self.world_cam_np.lookAt(0, 0, 0)
        self.master.cam.reparentTo(self.world_cam_np)

        self.loadElements()

        self.draw_intra_isl()
        self.rotateElements()

        self.master.taskMgr.add(self.gLoop, "gloop")
        self.accept("q", sys.exit)
        self.accept("p", self.toggle_pause)
        self.accept("i", self.toggle_isl)
        self.accept("arrow_up", self.moveUp)
        self.accept("arrow_down", self.moveDown)
        self.accept("arrow_right", self.moveRight)
        self.accept("arrow_left", self.moveLeft)

        # routing visualize
        self.current_src = None
        self.current_dest = None
        self.current_src_mode = None
        self.current_dest_mode = None
        self.routing_path_node = None
        self.ISL_routing_path_node = None

        self.heading = 0
        self.pitch = 0

    def setView(self):
        radius = 45
        heading_rad = math.radians(self.heading)
        pitch_rad = math.radians(self.pitch)

        x = radius * math.cos(pitch_rad) * math.sin(heading_rad)
        y = -radius * math.cos(pitch_rad) * math.cos(heading_rad)
        z = radius * math.sin(pitch_rad)

        self.world_cam_np.setPos(x, y, z)
        self.world_cam_np.lookAt(0, 0, 0)

    def moveUp(self):
        self.pitch -= 15
        self.setView()

    def moveDown(self):
        self.pitch += 15
        self.setView()

    def moveLeft(self):
        self.heading -= 15
        self.setView()

    def moveRight(self):
        self.heading += 15
        self.setView()

    def loadElements(self):
        self.base = self.render.attachNewNode("base")
        self.pivots = []

        delta_omega = 360 / self.N
        delta_f = 360 * self.F / (self.N * self.M)
        offset_deg = -90
        orbit_radius = self.earth_size_scale * (1 + self.altitude_km / self.radius)

        colors = plt.cm.jet(np.linspace(0, 1, self.M))  # sat_idx_in_orbit 기준 색상으로 변경

        for sat in self.satellites:
            n, m = sat.orbit_idx, sat.sat_idx_in_orbit

            if self.inclination_deg == 90:
                lon_asc_deg = (delta_omega / 2 * n + offset_deg) % 360
            else:
                lon_asc_deg = (delta_omega * n + offset_deg) % 360
            orbit_path = self.base.attachNewNode(f"orbit_path_{n}")
            orbit_path.setHpr(lon_asc_deg, 0, self.inclination_deg)

            orbit_pivot = orbit_path.attachNewNode(f"orbit_pivot_{n}")
            self.pivots.append(orbit_pivot)

            r, g, b, _ = colors[m]
            orbit_color = VBase4(r, g, b, 1)

            phase_deg = (-(360 * m / self.M + delta_f * n)) % 360
            phase_rad = math.radians(phase_deg)

            marker = self.master.loader.loadModel("../models/planet_sphere")
            marker.setScale(self.sat_size_scale)
            marker.setColor(orbit_color)
            marker.reparentTo(orbit_pivot)

            x = math.sin(phase_rad) * orbit_radius
            y = math.cos(phase_rad) * orbit_radius
            marker.setPos(x, y, 0)
            sat.marker = marker

        self.ground_pivot = self.base.attachNewNode("ground_pivot")

        for relay in self.ground_relays:
            x, y, z = relay.get_cartesian_coords()
            scale = self.earth_size_scale / relay.earth_radius
            marker = self.master.loader.loadModel("../models/planet_sphere")
            marker.setScale(0.2)
            marker.setColor(World.color_map.get(relay.continent, (1, 1, 1, 1)))
            marker.setPos(x * scale, y * scale, z * scale)
            marker.reparentTo(self.ground_pivot)
            relay.marker = marker

        self.earth = self.master.loader.loadModel("../models/planet_sphere")
        earth_tex = self.master.loader.loadTexture("../models/earth_1k_tex.jpg")
        self.earth.setTexture(earth_tex, 1)
        self.earth.reparentTo(self.base)
        self.earth.setScale(self.earth_size_scale)
        self.earth.setHpr(160, 0, 0)

    def draw_intra_isl(self):
        from panda3d.core import LineSegs

        lines = LineSegs()
        lines.setColor(0.7, 0.85, 1, 1)

        sat_map = {(sat.orbit_idx, sat.sat_idx_in_orbit): sat for sat in self.satellites}

        for sat in self.satellites:
            n, m = sat.orbit_idx, sat.sat_idx_in_orbit
            pos1 = sat.marker.getPos(self.base)

            for dR in [-1, 1]:
                neighbor = sat_map.get((n, (m + dR) % self.M))
                if neighbor:
                    pos2 = neighbor.marker.getPos(self.base)
                    lines.moveTo(pos1)
                    lines.drawTo(pos2)

        self.intra_isl_node = self.base.attachNewNode(lines.create())

    def update_inter_isl(self):
        from panda3d.core import LineSegs

        if hasattr(self, 'inter_isl_node'):
            self.inter_isl_node.removeNode()

        lines = LineSegs()
        lines.setColor(1, 1, 0, 1)

        sat_map = {(sat.orbit_idx, sat.sat_idx_in_orbit): sat for sat in self.satellites}

        for sat in self.satellites:
            n, m = sat.orbit_idx, sat.sat_idx_in_orbit
            pos1 = sat.marker.getPos(self.base)

            # delta_m = np.ceil(self.F * (1 - 1 / self.N))
            delta_m = self.F

            if n == 0:
                if self.inclination_deg == 90:
                    continue

                neighbor_m = (m - delta_m) % self.M
                neighbor = sat_map.get((self.N - 1, neighbor_m))
                if neighbor:
                    pos2 = neighbor.marker.getPos(self.base)
                    lines.moveTo(pos1)
                    lines.drawTo(pos2)

            elif n == self.N - 1:
                if self.inclination_deg == 90:
                    continue

                neighbor_m = (m + delta_m) % self.M
                neighbor = sat_map.get((0, neighbor_m))
                if neighbor:
                    pos2 = neighbor.marker.getPos(self.base)
                    lines.moveTo(pos1)
                    lines.drawTo(pos2)

            else:
                for dP in [-1, 1]:
                    neighbor_n = (n + dP) % self.N
                    neighbor = sat_map.get((neighbor_n, m))
                    if neighbor:
                        pos2 = neighbor.marker.getPos(self.base)
                        lines.moveTo(pos1)
                        lines.drawTo(pos2)

        self.inter_isl_node = self.base.attachNewNode(lines.create())

    def update_gsl(self):
        from panda3d.core import LineSegs

        if hasattr(self, 'gsl_node'):
            self.gsl_node.removeNode()

        if not self.is_isl:
            parent = self.base.attachNewNode("gsl_links")
            self.gsl_node = parent

            for relay in self.ground_relays:
                color = relay.marker.getColor()

                lines = LineSegs()
                lines.setColor(color)

                for sat in self.satellites:
                    P_sat, R_sat = sat.region
                    pos_sat = sat.marker.getPos(self.base)

                    if sat.is_ascending():
                        P_min, R_min ,P_max, R_max = relay.search_regions_asc

                        if P_min < P_max:
                            if R_min < R_max:
                                if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                            else:
                                if P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                        else:
                            if R_min < R_max:
                                if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                            else:
                                if (P_min <= P_sat or P_sat <= P_max) and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg,relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)

                    else:
                        P_min, R_min ,P_max, R_max = relay.search_regions_desc
                        if P_min < P_max:
                            if R_min < R_max:
                                if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                            else:
                                if P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                        else:
                            if R_min < R_max:
                                if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)
                            else:
                                if (P_min <= P_sat or P_sat <= P_max) and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg,relay.longitude_deg):
                                    pos_relay = relay.marker.getPos(self.base)
                                    lines.moveTo(pos_sat)
                                    lines.drawTo(pos_relay)

                relay_node = parent.attachNewNode(lines.create())
        else:
            for relay in self.ground_relays:
                relay.marker.setScale(0.0)

    def rotateElements(self):
        pass

    def gLoop(self, task):
        dt = self.speed / 30

        if not self.is_paused:
            self.sim_time += 1

            orbit_angle = (self.sim_time * 360 / self.orbit_len) * self.speed
            for pivot in self.pivots:
                pivot.setH(orbit_angle % 360)

            for sat in self.satellites:
                sat.update_position(self.omega_s, dt)
                sat.region = self.mapper.get_region_index(sat)

            self.update_inter_isl()
            self.update_gsl()

        return Task.cont

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_text.show()
        else:
            self.pause_text.hide()

    def toggle_isl(self):
        self.is_isl = not self.is_isl

    def node_id_to_obj(self, node_id):
        # 1. Satellite
        for sat in self.satellites:
            if str(sat.node_id) == str(node_id):
                return sat

        # 2. Ground Relay (중간 relay 포함)
        for relay in self.ground_relays:
            if str(relay.node_id) == node_id.split("_")[0]:
                return relay

        # 3. User (SRC)
        if hasattr(self, "current_src") and node_id.startswith(str(self.current_src.node_id)):
            return self.current_src

        # 4. Dest (redundant fallback)
        if hasattr(self, "current_dest") and node_id.startswith(str(self.current_dest.node_id)):
            return self.current_dest

        return None

    def draw_routing_path(self, path, color=(1, 1, 1, 1), isl=False):
        from panda3d.core import LineSegs

        # if hasattr(self, "routing_path_node") and self.routing_path_node:
        #     self.routing_path_node.removeNode()

        lines = LineSegs()
        lines.setColor(color)
        lines.setThickness(5.0)

        for i in range(len(path) - 1):
            obj1 = self.node_id_to_obj(path[i])
            obj2 = self.node_id_to_obj(path[i + 1])
            if not obj1 or not obj2:
                continue

            pos1 = obj1.marker.getPos(self.base)
            pos2 = obj2.marker.getPos(self.base)

            lines.moveTo(pos1)
            lines.drawTo(pos2)

        if isl:
            self.ISL_routing_path_node = self.render.attachNewNode(lines.create())
        else:
            self.routing_path_node = self.render.attachNewNode(lines.create())

    def add_node_marker(self, node, role):
        # role = 'src' or 'dest'
        lat, lon = node.latitude_deg, node.longitude_deg
        x, y, z = node.get_cartesian_coords()
        scale = self.earth_size_scale / node.earth_radius

        marker = self.master.loader.loadModel("../models/planet_sphere")
        marker.setScale(0.4)
        if role == "src":
            marker.setColor((1, 0, 0, 1))
        else:
            marker.setColor((0, 1, 0, 1))
        marker.setPos(x * scale, y * scale, z * scale)
        marker.reparentTo(self.ground_pivot)

        # ✅ 여기서 node.marker = marker 유지
        if role == "src":
            self.current_src.marker = marker
        else:
            self.current_dest.marker = marker