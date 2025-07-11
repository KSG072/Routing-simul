# rtp_renderer.py
from panda3d.core import NodePath, OrthographicLens, VBase4, WindowProperties, LineSegs
import matplotlib.pyplot as plt
import numpy as np

class RTPRenderer:
    def __init__(self, win_mgr, loader, satellites, ground_relays, mapper, N, M):
        self.win_mgr = win_mgr
        self.loader = loader
        self.satellites = satellites
        self.ground_relays = ground_relays
        self.mapper = mapper
        self.N = N
        self.M = M

        self.grid_spacing = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.scene_np = NodePath("rtp_scene")
        self.sat_markers = {}
        self.real_time_mode = False
        self.search_area_mode = 0

        self.single_idx = 0
        self.single_toggle_state = 0

        self.routing_path_node = None  # 라우팅 경로 선 저장용
        self.ISL_routing_path_node = None  # 라우팅 경로 선 저장용
        self.current_src = None
        self.current_dest = None
        self.current_src_mode = None
        self.current_dest_mode = None
        self.markers = {}  # node_id → marker NodePath (sat, relay, src/dest 포함)

        self._init_color_map()
        self._draw_grid_lines()
        self._init_ground_relay_nodes()
        self._init_search_area_boxes()
        self._init_satellite_nodes()
        self._setup_camera()

    def _init_color_map(self):
        cmap = plt.cm.jet(np.linspace(0, 1, self.M))
        self.color_map = [VBase4(*cmap[i][:3], 1) for i in range(self.M)]

    def _draw_grid_lines(self):
        lines = LineSegs()
        lines.setColor(0.3, 0.3, 0.3, 1)
        lines.setThickness(0.5)

        x0 = self.offset_x - 0.5 * self.grid_spacing
        y0 = self.offset_y - 0.5 * self.grid_spacing
        x_max = x0 + self.N * self.grid_spacing
        y_max = y0 + self.M * self.grid_spacing

        for p in range(self.N + 1):
            x = x0 + p * self.grid_spacing
            lines.moveTo(x, 0, y0)
            lines.drawTo(x, 0, y_max)

        for r in range(self.M + 1):
            y = y0 + r * self.grid_spacing
            lines.moveTo(x0, 0, y)
            lines.drawTo(x_max, 0, y)

        self.scene_np.attachNewNode(lines.create())

    def _init_satellite_nodes(self):
        for sat in self.satellites:
            P, R = sat.region
            x = self.offset_x + P * self.grid_spacing
            y = self.offset_y + R * self.grid_spacing

            color = self.color_map[sat.sat_idx_in_orbit % self.M]
            marker = self.loader.loadModel("../models/planet_sphere")
            marker.setScale(0.2)
            marker.setColor(color)
            marker.setPos(x, 0, y)
            marker.reparentTo(self.scene_np)

            self.sat_markers[sat.node_id] = marker
            self.markers[sat.node_id] = marker  # 👈 경로 그리기용

    def _init_ground_relay_nodes(self):
        for relay in self.ground_relays:
            color = relay.marker.getColor()
            for suffix, region in [("asc", relay.region_asc), ("desc", relay.region_desc)]:
                if region is None:
                    continue
                P, R = region
                x = self.offset_x + P * self.grid_spacing
                y = self.offset_y + R * self.grid_spacing
                marker = self.loader.loadModel("../models/planet_sphere")
                marker.setScale(0.4)
                marker.setColor(color)
                marker.setPos(x, 0, y)
                marker.reparentTo(self.scene_np)

                node_id = f"{relay.node_id}_{suffix}"
                self.markers[node_id] = marker  # 👈 여기에 저장

    def _init_search_area_boxes(self):
        self.original_boxes = []
        self.search_boxes = []
        self.relay_boxes = []  # [(originals), (searches)] per relay

        for relay in self.ground_relays:
            color = relay.marker.getColor()
            r, g, b = color.getX(), color.getY(), color.getZ()

            origs = []
            searches = []

            for region_type, container, alpha in [
                ("original_region_asc", origs, 0.1),
                ("original_region_desc", origs, 0.1),
                ("search_regions_asc", searches, 0.3),
                ("search_regions_desc", searches, 0.3),
            ]:
                region_data = getattr(relay, region_type, None)
                if not region_data:
                    continue
                p_start, r_start, p_end, r_end = region_data
                box_color = VBase4(r, g, b, alpha)
                thick = 0.5 if region_type[0] == 'o' else 3
                box = self._make_box(p_start, r_start, p_end, r_end, box_color, thick=thick)
                box.reparentTo(self.scene_np)
                # box.hide()
                container.append(box)

            self.original_boxes.extend(origs)
            self.search_boxes.extend(searches)
            self.relay_boxes.append((origs, searches))

    def _make_box(self, p_start, r_start, p_end, r_end, color: VBase4, thick=1.5) -> NodePath:
        lines = LineSegs()
        lines.setColor(color)
        lines.setThickness(thick)

        x0 = self.offset_x + (p_start - 0.5) * self.grid_spacing
        y0 = self.offset_y + (r_start - 0.5) * self.grid_spacing
        x1 = self.offset_x + (p_end + 0.5) * self.grid_spacing
        y1 = self.offset_y + (r_end + 0.5) * self.grid_spacing

        lines.moveTo(x0, 0, y0)
        lines.drawTo(x1, 0, y0)
        lines.drawTo(x1, 0, y1)
        lines.drawTo(x0, 0, y1)
        lines.drawTo(x0, 0, y0)

        return NodePath(lines.create())

    def _setup_camera(self):
        wp = WindowProperties()
        wp.setSize(self.N * 20, self.M * 20)
        wp.setOrigin(50, 800)
        wp.setTitle("RTPG Window")

        rtp_win = self.win_mgr.openWindow(props=wp, makeCamera=False)
        cam_np = self.win_mgr.makeCamera(win=rtp_win, displayRegion=(0, 1, 0, 1))

        lens = OrthographicLens()
        lens.setFilmSize((self.N + 1), (self.M + 1))
        lens.setNearFar(-100, 100)
        cam_np.node().setLens(lens)
        cam_np.setPos((self.N - 1) / 2, 40, (self.M - 1) / 2)
        cam_np.lookAt((self.N - 1) / 2, 0, (self.M - 1) / 2)
        cam_np.node().setScene(self.scene_np)

    def update(self):
        for sat in self.satellites:
            P, R = self.mapper.get_region_index(sat, floor=not self.real_time_mode)
            x = self.offset_x + P * self.grid_spacing
            y = self.offset_y + R * self.grid_spacing
            marker = self.sat_markers.get(sat.node_id)
            if marker:
                marker.setPos(x, 0, y)

    def toggle_real_time_mode(self):
        self.real_time_mode = not self.real_time_mode

    def toggle_search_area(self):
        self.search_area_mode = (self.search_area_mode + 1) % 4
        if self.search_area_mode == 0:
            for b in self.original_boxes + self.search_boxes:
                b.hide()
        elif self.search_area_mode == 1:
            for b in self.original_boxes:
                b.show()
            for b in self.search_boxes:
                b.hide()
        elif self.search_area_mode == 2:
            for b in self.original_boxes:
                b.hide()
            for b in self.search_boxes:
                b.show()
        elif self.search_area_mode == 3:
            for b in self.original_boxes + self.search_boxes:
                b.show()

    def toggle_single_relay_box(self):
        # 1. hide all
        for b in self.original_boxes + self.search_boxes:
            b.hide()

        # 2. select next relay
        relay_count = len(self.relay_boxes)
        relay_idx = self.single_idx % relay_count
        toggle_state = self.single_toggle_state % 4

        orig, search = self.relay_boxes[relay_idx]

        if toggle_state == 0:
            pass  # none shown
        elif toggle_state == 1:
            for b in orig:
                b.show()
        elif toggle_state == 2:
            for b in search:
                b.show()
        elif toggle_state == 3:
            for b in orig + search:
                b.show()

        # 3. update toggle
        self.single_toggle_state += 1
        if self.single_toggle_state == 4:
            self.single_toggle_state = 0
            self.single_idx = (self.single_idx + 1) % relay_count

    def add_node_marker(self, node, suffix, color=(1, 1, 1, 1)):
        region = node.region_asc if suffix == "asc" else node.region_desc
        if region is None:
            return

        P, R = region
        x = self.offset_x + P * self.grid_spacing
        y = self.offset_y + R * self.grid_spacing

        marker = self.loader.loadModel("../models/planet_sphere")
        marker.setScale(0.4)
        marker.setColor(color)
        marker.setPos(x, 0, y)
        marker.reparentTo(self.scene_np)

        node_id = f"{node.node_id}_{suffix}"
        self.markers[node_id] = marker  # ✅ marker 저장만, node에 넣지 마세요!

    def draw_routing_path(self, path, color=(1, 1, 1, 1), isl=False):
        from panda3d.core import LineSegs

        # 이전 경로 제거
        # if self.routing_path_node:
        #     self.routing_path_node.removeNode()

        lines = LineSegs()
        lines.setColor(color)  # 흰색
        lines.setThickness(2.5)

        for i in range(len(path) - 1):
            nid1 = path[i]
            nid2 = path[i + 1]

            m1 = self.markers.get(nid1)
            m2 = self.markers.get(nid2)

            if not m1 or not m2:
                print(f"[RTPG] marker not found: {nid1} or {nid2}")
                continue

            pos1 = m1.getPos(self.scene_np)
            pos2 = m2.getPos(self.scene_np)

            lines.moveTo(pos1)
            lines.drawTo(pos2)

        if isl:
            self.ISL_routing_path_node = self.scene_np.attachNewNode(lines.create())
        else:
            self.routing_path_node = self.scene_np.attachNewNode(lines.create())