# rtp_renderer.py
import csv, re, math, os
try:
    import pandas as pd
except Exception:
    pd = None

from panda3d.core import (
    TextNode, VBase4, Point3, Camera, LineSegs, NodePath, Geom, GeomNode, GeomTriangles, GeomVertexData,
    GeomVertexFormat, GeomVertexWriter,
    BillboardEffect, LineSegs, VBase4, WindowProperties, OrthographicLens
)
import matplotlib.pyplot as plt
import numpy as np
from plotly.io import renderers


def _make_ngon_node(n: int, radius: float=0.22, angle_offset: float=0.0) -> NodePath:
    """XZ 평면에 정N각형을 채워진 팬으로 생성 (카메라 빌보드용 2D 마커)"""
    vformat = GeomVertexFormat.getV3()
    vdata = GeomVertexData(f"ngon{n}", vformat, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")

    # 중심점
    vw.addData3(0.0, 0.0, 0.0)
    # 외곽점
    for i in range(n):
        a = angle_offset + 2.0*math.pi*i/n
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        vw.addData3(x, 0.0, z)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(1, n):
        tris.addVertices(0, i, i+1)
    tris.addVertices(0, n, 1)  # 마지막 면

    geom = Geom(vdata); geom.addPrimitive(tris)
    node = GeomNode(f"ngon{n}"); node.addGeom(geom)
    return NodePath(node)

def _make_star_node(points: int=5, R: float=0.24, r: float=0.10, angle_offset: float=0.0) -> NodePath:
    """별(2*points 꼭짓점) 채워진 팬"""
    vformat = GeomVertexFormat.getV3()
    vdata = GeomVertexData(f"star{points}", vformat, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")

    vw.addData3(0.0, 0.0, 0.0)
    total = points * 2
    for i in range(total):
        a = angle_offset + math.pi * i / points
        rad = R if (i % 2 == 0) else r
        x = rad * math.cos(a)
        z = rad * math.sin(a)
        vw.addData3(x, 0.0, z)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(1, total):
        tris.addVertices(0, i, i+1)
    tris.addVertices(0, total, 1)

    geom = Geom(vdata); geom.addPrimitive(tris)
    node = GeomNode(f"star{points}"); node.addGeom(geom)
    return NodePath(node)

def _make_marker(shape: str) -> NodePath:
    shape = shape.lower()
    if shape == "triangle":   return _make_ngon_node(3, radius=0.24, angle_offset=math.radians(90))
    if shape == "square":     return _make_ngon_node(4, radius=0.22, angle_offset=math.radians(45))
    if shape == "diamond":    return _make_ngon_node(4, radius=0.22, angle_offset=0.0)
    if shape == "pentagon":   return _make_ngon_node(5, radius=0.23, angle_offset=math.radians(90))
    if shape == "hexagon":    return _make_ngon_node(6, radius=0.22, angle_offset=math.radians(0))
    if shape == "star":       return _make_star_node(points=5, R=0.24, r=0.10, angle_offset=math.radians(90))
    # 기본은 원(정다각형 근사)
    return _make_ngon_node(16, radius=0.22)

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
        # self._init_search_area_boxes()
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
        y0 = self.offset_y * self.grid_spacing
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
            if not self.real_time_mode:
                R += 0.5
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
        # 같은 대륙끼리 모양을 순환 배정
        shape_cycle = ["triangle", "square", "diamond", "circle", "star"]

        from collections import defaultdict
        groups = defaultdict(list)
        for relay in self.ground_relays:
            groups[getattr(relay, "continent", "unknown")].append(relay)

        # 대륙별로 안정적 정렬(원하면 다른 키 사용)
        for continent, relays in groups.items():
            relays.sort(key=lambda r: getattr(r, "node_id", 0))
            for i, relay in enumerate(relays):
                # 대륙 내 모양 선택
                shape = relay.marker_shape

                # 색상은 기존 리레이 마커 색(대륙색) 그대로
                color = relay.marker.getColor() if getattr(relay, "marker", None) else (1, 1, 1, 1)

                # asc/desc 각각 마커 생성
                for suffix, region in [("asc", relay.region_asc), ("desc", relay.region_desc)]:
                    if region is None:
                        continue
                    P, R = region
                    x = self.offset_x + P * self.grid_spacing
                    y = self.offset_y + R * self.grid_spacing

                    marker = _make_marker(shape)
                    marker.setColor(color)
                    marker.setTwoSided(True)
                    marker.setLightOff(1)
                    marker.setEffect(BillboardEffect.makePointEye())
                    marker.setScale(3.0)

                    # ✅ 그리드보다 항상 위에 보이도록 (둘 중 하나만으로도 충분하지만, 안정성 위해 같이 권장)
                    marker.setBin('fixed', 20)  # 렌더 순서를 grid(LineSegs)보다 뒤로
                    marker.setDepthOffset(1)  # 깊이 오프셋으로 z-fighting 회피
                    marker.setDepthTest(True)
                    marker.setDepthWrite(True)

                    # ✅ 같은 평면(y=0)에서 z-fighting 피하려고 아주 살짝 카메라 쪽으로 이동(+Y)
                    #    카메라가 +Y에서 0을 바라보니 y를 +ε로 두면 확실히 위로 뜹니다.
                    marker.setPos(x, 0.001, y)  # << 기존 x, 0, y 에서 0.001로 변경

                    marker.reparentTo(self.scene_np)

                    node_id = f"{relay.node_id}_{suffix}"
                    self.markers[node_id] = marker

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
                # ("original_region_asc", origs, 0.1),
                # ("original_region_desc", origs, 0.1),
                # ("search_regions_asc", searches, 0.1),
                # ("search_regions_desc", searches, 0.1),
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
        cam_np.setPos((self.N - 1) / 2, 40, self.M / 2)
        cam_np.lookAt((self.N - 1) / 2, 0, self.M / 2)
        cam_np.node().setScene(self.scene_np)

    def update(self):
        for sat in self.satellites:
            P, R = self.mapper.get_region_index(sat, floor=not self.real_time_mode)
            if not self.real_time_mode:
                R += 0.5
            x = self.offset_x + P * self.grid_spacing
            y = self.offset_y + R * self.grid_spacing
            marker = self.sat_markers.get(sat.node_id)
            if marker:
                marker.setPos(x, 0, y)

    def check_r_degree(self):
        for sat in self.satellites[0:22]:
            P, R = self.mapper.get_region_index(sat, floor=not self.real_time_mode)
            print(f"satellite_id: {sat.node_id}, P: {P}, R: {R}")


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
        if self.routing_path_node:
            self.routing_path_node.removeNode()

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

    # =========================
    # 1) CSV 로딩 + 진행 로그
    # =========================
    def load_drop_counts_from_csv(
        self,
        csv_path: str,
        chunksize: int = 200_000,
        verbose: bool = True,
        id_regex: str = r"(\d+)",   # 예: 'sat_123' → '123'
    ) -> dict:
        """
        CSV에서 status='drop' 행을 읽으며 진행률/요약을 콘솔에 출력하고,
        같은 폴더에 <stem>__drop_counts.csv 로
        node_id, drop_count 만 저장한다.
        반환값은 히트맵용 drop_counts(dict).
        """
        from collections import Counter

        def _norm_name(s: str) -> str:
            return re.sub(r"\s+", "", s or "").strip().lower()

        # 컬럼 키 후보  (문자열 → 집합, + 정규화)
        status_keys = {_norm_name("Status")}  # {'status'}
        drop_keys = {_norm_name("Drop Location")}  # {'droplocation'}

        # 유효 위성 ID 집합(히트맵과 일치)
        valid_sat_ids = set()
        for sat in self.satellites:
            valid_sat_ids.add(str(sat.node_id))
            if isinstance(sat.node_id, (int, np.integer)):
                valid_sat_ids.add(str(int(sat.node_id)))

        # 출력 경로(집계 파일만)
        out_dir = os.path.dirname(csv_path) or "."
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        counts_path = os.path.join(out_dir, f"{stem}__drop_counts.csv")

        # 진행률 총행수(없어도 진행)
        total_rows = None
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1
        except Exception:
            total_rows = None

        drop_counts = Counter()
        seen_rows = drop_rows = mapped_rows = 0

        # ------- pandas 청크 모드 -------

        first_chunk = True
        status_col = drop_col = None
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            seen_rows += len(chunk)

            if first_chunk:
                headers = list(chunk.columns)
                norm = {_norm_name(h): h for h in headers}
                status_col = next((norm[k] for k in status_keys if k in norm), None)
                drop_col   = next((norm[k] for k in drop_keys   if k in norm), None)
                if verbose:
                    print(f"[drop-heatmap] columns: status={status_col}, drop={drop_col}")
                if not status_col or not drop_col:
                    if verbose: print("[drop-heatmap] 필요한 컬럼을 찾지 못했습니다.")
                    break
                first_chunk = False

            svals = chunk[status_col].astype(str).str.strip().str.lower()
            mask_drop = svals.eq("drop") | svals.str.contains("drop", na=False)
            drops = chunk.loc[mask_drop, drop_col].dropna()
            drop_rows += len(drops)

            for raw in drops.astype(str):
                raw = raw.strip()
                m = re.search(id_regex, raw) if id_regex else None
                key = m.group(1) if m else raw
                # 3.0 → 3 같은 숫자 문자열 정리
                try:
                    vf = float(key)
                    if math.isfinite(vf) and vf.is_integer():
                        key = str(int(vf))
                except Exception:
                    pass

                # if key in valid_sat_ids:
                drop_counts[key] += 1
                mapped_rows += 1

            if verbose:
                if total_rows:
                    pct = seen_rows / total_rows * 100
                    print(f"[drop-heatmap] processed {seen_rows:,}/{total_rows:,} rows ({pct:5.1f}%) "
                          f"| drops {drop_rows:,} | mapped {mapped_rows:,}")
                else:
                    print(f"[drop-heatmap] processed {seen_rows:,} rows | drops {drop_rows:,} | mapped {mapped_rows:,}")


        # 요약 로그
        if verbose:
            top = drop_counts.most_common(10)
            print(f"[drop-heatmap] unique drop sats: {len(drop_counts)}")
            if top: print("[drop-heatmap] top 10:", top)

        # === 집계 CSV 저장 (간단 버전) ===
        with open(counts_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "drop_count"])
            for node_id, cnt in sorted(drop_counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
                w.writerow([node_id, cnt])

        if verbose:
            print(f"[drop-heatmap] saved drop counts → {counts_path}")

        return dict(drop_counts)



    # ==========================================
    # 2) 파랑→빨강 히트맵 적용 (CSV 경로 포함)
    # ==========================================
    def apply_drop_heatmap_blue_red(
        self,
        csv_path: str = "results/tmc data rate rollback/result_360.csv",
        zero_color: VBase4 = VBase4(0.60, 0.60, 0.60, 1.0),  # 드롭 0인 위성은 회색
        chunksize: int = 200_000,
        verbose: bool = True,
        id_regex: str = r"(\d+)",
    ):
        """
        - 드롭 0: zero_color(회색)
        - 드롭 ≥1: 파랑(낮음) → 빨강(높음)로 정규화 색상
        - 모두 0이면 전체 회색 처리
        """
        drop_counts = self.load_drop_counts_from_csv(
            csv_path=csv_path,
            chunksize=chunksize,
            verbose=verbose,
            id_regex=id_regex,
        )

        # 위성별 카운트 수집
        per_sat = {}
        vals = []
        for sat in self.satellites:
            k1 = str(sat.node_id)
            k2 = str(int(sat.node_id)) if isinstance(sat.node_id, (int, np.integer)) else None
            if k1 in drop_counts:
                cnt = drop_counts[k1]
            elif k2 is not None and k2 in drop_counts:
                cnt = drop_counts[k2]
            else:
                cnt = 0
            per_sat[sat.node_id] = cnt
            vals.append(cnt)

        # 0 제외한 양수들만으로 max 산정 (0은 회색 고정이라 스케일에 영향 X)
        positives = [v for v in vals if v > 0]
        max_c = max(positives) if positives else 0

        def lerp_blue_red(t: float) -> VBase4:
            """t=0 → 파랑(0,0,1), t=1 → 빨강(1,0,0)"""
            t = max(0.0, min(1.0, t))
            return VBase4(t, 0.0, 1.0 - t, 1.0)

        # 모두 0인 경우: 전체 회색
        if max_c == 0:
            if verbose:
                print("[drop-heatmap] 모든 위성의 드롭 카운트가 0 → 전체 회색으로 표시합니다.")
            for sat in self.satellites:
                m = self.sat_markers.get(sat.node_id)
                if m:
                    m.setColor(zero_color)
            return

        # 일부라도 1 이상 존재: 0은 회색, 나머지는 파→빨로 정규화
        for sat in self.satellites:
            cnt = per_sat[sat.node_id]
            if cnt <= 0:
                color = zero_color
            else:
                t = cnt / max_c
                color = lerp_blue_red(t)
            m = self.sat_markers.get(sat.node_id)
            if m:
                m.setColor(color)
