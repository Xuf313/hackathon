"""
sar_mission.py - full Search & Rescue mission controller (TurtleBot3 Burger, PNU TECH WEEK kit)

Pipeline (Perception -> Planning -> Action, as in the course):
  odometry (encoders + gyro) -> occupancy grid (LiDAR) -> coverage exploration
  -> target detection (colour or YOLO) -> approach -> return to start (0, 0)

Decision making = Finite State Machine:
  INIT      spin 360 deg to build the first map
  EXPLORE   go where the CAMERA has not looked yet (and to map frontiers)
  APPROACH  target seen -> plan a safe path to it, final metres by camera
  REACHED   stop briefly, remember target; more targets -> EXPLORE, else RETURN
  RETURN    plan back to the start over known free space
  DONE      stopped at start
Path following = look-ahead (pure pursuit): kappa = 2y/(x^2+y^2), w = v*kappa

MODE = "tune": drive with W/A/S/D and tune the detector live with sliders
               (use this on competition day as soon as the targets are announced).

Everything you normally change is in the CONFIG block.
"""

import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from controller import Robot, Keyboard

# =============================================================================
# CONFIG  - change these on competition day
# =============================================================================
MODE = "mission"            # "mission" = autonomous run, "tune" = keyboard + detector sliders

# ---- WHAT is the target?  (describe it from the organisers' announcement) ----
# Each target type = colour(s) + shape + real size. Several types allowed.
#   colors : keys of COLOR_PRESETS (get new values with MODE = "tune")
#   shape  : "round"  ball / apple / sphere (circle-like blob)
#            "square" cube / box seen from the side (4 corners, width ~ height)
#            "tall"   bottle / can / cylinder standing up (height > 1.4 x width)
#            "wide"   lying can / flat box (width > 1.4 x height)
#            "triangle", "any"
#   size   : real WIDTH in metres (used to reject too big / too small objects of the same colour)
#   yolo   : optional COCO class id -> use YOLO instead of colour for this type
#            (32 sports ball, 47 apple, 49 orange, 39 bottle, 41 cup, 15 cat, 16 dog, 19 cow)
TARGETS = [
    dict(name="apple", colors=["red"], shape="round", size=0.10),
    # dict(name="blue box", colors=["blue"], shape="square", size=0.20),
    # dict(name="cat", yolo=15, shape="any", size=0.40),
]
FLOOR_ONLY = True           # target stands on the floor (its bottom edge is below the horizon).
                            # Rejects things on tables/shelves.
SIZE_CHECK = True           # reject blobs whose real size (from ground geometry) does not match `size`
SIZE_TOL = 1.8              # accepted if size/SIZE_TOL < measured < size*SIZE_TOL
CAM_HEIGHT = 0.10           # [m] camera height above floor (calibrate in tune mode, see README)
YOLO_MODEL = "../../models/YOLO/yolo11n.pt"
YOLO_CONF = 0.35

# ---- HOW MANY / WHEN ----
N_TARGETS = None            # None = target count is unknown: search the whole area, reach every
                            # target seen, then return when everything is explored or time runs out.
TIME_LIMIT = None           # [s] simulation-time limit, e.g. 600. None = no limit
RETURN_MARGIN = 90          # [s] start returning this long before TIME_LIMIT
REACH_DIST = 0.30           # [m] "reached" when camera distance estimate < this

SHOW_MAP = True             # OpenCV window with the map
SHOW_CAMERA = True          # OpenCV window with camera + detection mask

# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255). Get new values with MODE = "tune".
COLOR_PRESETS = {
    "red":    [((0, 150, 70), (8, 255, 255)), ((170, 150, 70), (179, 255, 255))],
    "green":  [((40, 100, 40), (85, 255, 255))],
    "orange": [((10, 170, 120), (22, 255, 255))],
    "purple": [((125, 80, 40), (160, 255, 255))],
    "yellow": [((22, 150, 120), (35, 255, 255))],
    "blue":   [((95, 150, 50), (125, 255, 255))],
    "custom": [((0, 0, 0), (179, 255, 255))],     # paste the line printed by tune mode here
}
DETECT_MIN_AREA = 12        # [px, on the half-size image]
DETECT_CONFIRM = 3          # consecutive detections needed
HORIZON_MARGIN = 0.02       # tolerance for the horizon line (fraction of image height)

# ---- robot (kit notebook) ----
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
ROBOT_RADIUS = 0.105
MAX_WHEEL = 6.6             # [rad/s] (motor limit 6.67)

# ---- motion ----
V_MAX = 0.22                # [m/s] TB3 top speed
V_APPROACH = 0.12
W_MAX = 1.8                 # [rad/s]
LOOKAHEAD = 0.40            # [m] look-ahead distance
STOP_DIST = 0.23            # [m] from robot centre: stop if obstacle this close ahead
SLOW_DIST = 0.45            # [m] start slowing down
SERVO_DIST = 1.0            # [m] closer than this: steer directly by camera

# ---- map / planning / exploration ----
RES = 0.05                  # [m] grid cell
MAP_SIZE = 40.0             # [m] square map centred on start
R_HARD = ROBOT_RADIUS + 0.05   # [m] never plan closer than this to an obstacle
R_SOFT = 0.40               # [m] prefer to stay this far from obstacles
W_SOFT = 6.0
REPLAN_PERIOD = 2.0         # [s]
CAM_RANGE = 2.5             # [m] floor counts as "looked at" up to this distance in camera view
MIN_GOAL_CLUSTER = 8        # [cells] ignore tiny unexplored patches
GOAL_TIMEOUT = 30.0         # [s]
DETECT_EVERY = 1            # run detector every N steps (auto 3 for yolo)


# =============================================================================
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Grid:
    """Log-odds occupancy grid centred on the start pose + 'seen by camera' layer."""
    L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 0.9, -4.0, 4.0

    def __init__(self):
        self.n = int(MAP_SIZE / RES)
        self.half = MAP_SIZE / 2
        self.L = np.zeros((self.n, self.n), np.float32)
        self.known = np.zeros((self.n, self.n), bool)
        self.cam_seen = np.zeros((self.n, self.n), bool)
        self.t = np.arange(RES / 2, 3.5, RES * 0.7).astype(np.float32)

    def cell(self, x, y):
        return int((y + self.half) / RES), int((x + self.half) / RES)

    def world(self, r, c):
        return (c + 0.5) * RES - self.half, (r + 0.5) * RES - self.half

    def _idx(self, xs, ys):
        c = ((xs + self.half) / RES).astype(int)
        r = ((ys + self.half) / RES).astype(int)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        return r[ok], c[ok]

    def update(self, x, y, th, r, angles, rmax, cam_half_fov):
        finite = np.isfinite(r) & (r > 0.12)
        hit = finite & (r < rmax * 0.97)
        r_free = np.minimum(np.where(hit, r, rmax * 0.97), 3.45)
        a = th + angles
        ca, sa = np.cos(a), np.sin(a)
        tx = x + self.t[None, :] * ca[:, None]
        ty = y + self.t[None, :] * sa[:, None]
        free = self.t[None, :] < (r_free[:, None] - RES)
        fr, fc = self._idx(tx[free], ty[free])
        self.L[fr, fc] += self.L_FREE
        self.known[fr, fc] = True
        # camera coverage: free floor inside the camera cone and range
        cam = free & (np.abs(angles)[:, None] < cam_half_fov) & (self.t[None, :] < CAM_RANGE)
        cr, cc = self._idx(tx[cam], ty[cam])
        self.cam_seen[cr, cc] = True
        hr, hc = self._idx(x + r[hit] * ca[hit], y + r[hit] * sa[hit])
        self.L[hr, hc] += self.L_OCC - self.L_FREE
        self.known[hr, hc] = True
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)

    def bbox(self, extra_cells=()):
        rows = np.where(np.any(self.known, axis=1))[0]
        cols = np.where(np.any(self.known, axis=0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        for (r, c) in extra_cells:
            r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
        m = 3
        return (max(r0 - m, 0), min(r1 + m + 1, self.n),
                max(c0 - m, 0), min(c1 + m + 1, self.n))


class Planner:
    """Dijkstra (skimage MCP) on a cost map that keeps the robot away from walls."""

    def __init__(self, grid):
        self.g = grid

    def build(self, robot_cell):
        g = self.g
        r0, r1, c0, c1 = g.bbox([robot_cell, g.cell(0, 0)])
        self.crop = (r0, c0)
        L = g.L[r0:r1, c0:c1]
        known = g.known[r0:r1, c0:c1]
        occ = (L > 0.6).astype(np.uint8)
        dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * RES
        free = known & (L < -0.2)
        trav = free & (dist > R_HARD)
        rr, rc = robot_cell[0] - r0, robot_cell[1] - c0
        yy, xx = np.ogrid[:L.shape[0], :L.shape[1]]
        near = (yy - rr) ** 2 + (xx - rc) ** 2 <= 16
        trav |= near & (occ == 0) & (dist > ROBOT_RADIUS * 0.8)
        cost = 1.0 + W_SOFT * np.clip((R_SOFT - dist) / (R_SOFT - R_HARD), 0, 1) ** 2
        cost[~trav] = np.inf
        self.free, self.unknown = free, ~known
        self.cam_seen = g.cam_seen[r0:r1, c0:c1]
        self.mcp = MCP_Geometric(cost, fully_connected=True)
        start = (min(max(rr, 0), L.shape[0] - 1), min(max(rc, 0), L.shape[1] - 1))
        self.cum, _ = self.mcp.find_costs([start])

    def _local(self, cell):
        return cell[0] - self.crop[0], cell[1] - self.crop[1]

    def _inside(self, r, c):
        return 0 <= r < self.cum.shape[0] and 0 <= c < self.cum.shape[1]

    def reachable(self, cell):
        r, c = self._local(cell)
        return self._inside(r, c) and np.isfinite(self.cum[r, c])

    def path_to(self, cell):
        r, c = self._local(cell)
        if not self._inside(r, c) or not np.isfinite(self.cum[r, c]):
            return None
        pts = self.mcp.traceback((r, c))
        out = [self.g.world(p[0] + self.crop[0], p[1] + self.crop[1]) for p in pts]
        return out[::2] + [out[-1]] if len(out) > 2 else out

    def nearest_reachable(self, cell, radius_m):
        r, c = self._local(cell)
        k = int(radius_m / RES)
        H, W = self.cum.shape
        r0, r1, c0, c1 = max(r - k, 0), min(r + k + 1, H), max(c - k, 0), min(c + k + 1, W)
        if r0 >= r1 or c0 >= c1:
            return None
        win = self.cum[r0:r1, c0:c1].copy()
        yy, xx = np.ogrid[r0:r1, c0:c1]
        win[(yy - r) ** 2 + (xx - c) ** 2 > k * k] = np.inf
        if not np.isfinite(win).any():
            return None
        i = np.unravel_index(np.argmin(win), win.shape)
        return i[0] + r0 + self.crop[0], i[1] + c0 + self.crop[1]

    def closest_reachable(self, cell):
        r, c = self._local(cell)
        ys, xs = np.where(np.isfinite(self.cum))
        if len(ys) == 0:
            return None
        j = int(np.argmin((ys - r) ** 2 + (xs - c) ** 2 + 1e-3 * self.cum[ys, xs]))
        return ys[j] + self.crop[0], xs[j] + self.crop[1]

    def best_goal(self, blacklist):
        """Where to explore next: reachable floor the camera has not looked at yet,
        plus map frontiers (free cells next to unknown). Returns (goal_cell, centroid_xy)."""
        reach = np.isfinite(self.cum)
        near_unk = cv2.dilate(self.unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cand = reach & ((self.free & near_unk) | ~self.cam_seen)
        if blacklist:     # remove only the blacklisted circles, keep the rest of each area
            H, W = cand.shape
            yy, xx = np.ogrid[:H, :W]
            for bx, by, br in blacklist:
                r, c = self._local(self.g.cell(bx, by))
                k = br / RES
                if -k <= r < H + k and -k <= c < W + k:
                    cand &= (yy - r) ** 2 + (xx - c) ** 2 > k * k
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        best, best_score = None, np.inf
        for i in range(1, n):
            size = stats[i, cv2.CC_STAT_AREA]
            if size < MIN_GOAL_CLUSTER:
                continue
            ys, xs = np.where(labels == i)
            costs = self.cum[ys, xs]
            j = int(np.argmin(costs))
            cell = (ys[j] + self.crop[0], xs[j] + self.crop[1])
            score = costs[j] * RES - 0.015 * min(size, 400)
            if score < best_score:
                cx, cy = cents[i]
                best_score = score
                best = (cell, self.g.world(cy + self.crop[0], cx + self.crop[1]))
        return best


def classify_shape(area, perim, bw, bh, approx_n, solidity):
    """Rough 2-D shape class of a blob."""
    aspect = bh / max(bw, 1)
    circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0
    if aspect >= 1.4:
        return "tall"
    if aspect <= 0.7:
        return "wide"
    if approx_n == 3:
        return "triangle"
    if approx_n == 4 and solidity > 0.9 and circ < 0.82:
        return "square"
    if circ >= 0.68:
        return "round"
    return "irregular"


def shape_ok(want, got):
    if want == "any":
        return True
    if want == "round":
        return got == "round"
    if want == "square":
        return got in ("square", "round")      # small cubes often look rounded after blur
    return got == want


class Detector:
    """Finds targets described in TARGETS. detect() -> (name, bearing[rad,+left], distance[m]) or None."""

    def __init__(self, camera):
        self.cam = camera
        self.W, self.H = camera.getWidth(), camera.getHeight()
        self.w, self.h = self.W // 2, self.H // 2          # half resolution: faster
        self.f = (self.w / 2) / math.tan(camera.getFov() / 2)
        self.horizon = self.h / 2
        self.streak = 0
        self.kernel = np.ones((3, 3), np.uint8)
        self.last_vis = None
        self.model = None
        if any("yolo" in t for t in TARGETS):
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            self.model.to("cpu")

    def frame(self):
        raw = self.cam.getImage()
        if raw is None:
            return None
        img = np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 4)[:, :, :3]
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def color_mask(self, img, ranges):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.zeros((self.h, self.w), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def geometry(self, x, y, bw, bh):
        """Distance from ground geometry (bottom edge) and from apparent size."""
        below = (y + bh) - self.horizon
        d_ground = CAM_HEIGHT * self.f / below if below > 1 else None
        return d_ground

    def judge(self, spec, x, y, bw, bh, shape):
        """Returns (reason_if_rejected, distance)."""
        d_size = self.f * spec["size"] / max(bw, 1)
        if FLOOR_ONLY and (y + bh) < self.horizon - HORIZON_MARGIN * self.h:
            return "on table", d_size
        if not shape_ok(spec.get("shape", "any"), shape):
            return f"shape:{shape}", d_size
        d_ground = self.geometry(x, y, bw, bh)
        if SIZE_CHECK and d_ground is not None and d_ground < 6.0:
            real_w = bw * d_ground / self.f
            if not (spec["size"] / SIZE_TOL < real_w < spec["size"] * SIZE_TOL):
                return f"size {real_w:.2f}m", d_size
            return None, 0.5 * (d_size + d_ground)
        return None, d_size

    def candidates(self, img, specs=None, ranges_override=None):
        """List of (name, x, y, bw, bh, area, shape, reason, dist) for every blob / box."""
        specs = specs or TARGETS
        out, masks = [], []
        for spec in specs:
            if "yolo" in spec:
                res = self.model.predict(img, conf=YOLO_CONF, classes=[spec["yolo"]],
                                         verbose=False, imgsz=320)[0]
                for bx in res.boxes.xyxy.cpu().numpy():
                    x, y = int(bx[0]), int(bx[1])
                    bw, bh = int(bx[2] - bx[0]), int(bx[3] - bx[1])
                    why, d = self.judge(dict(spec, shape="any"), x, y, bw, bh, "any")
                    out.append((spec["name"], x, y, bw, bh, bw * bh, "yolo", why, d))
                continue
            ranges = ranges_override or [r for col in spec["colors"] for r in COLOR_PRESETS[col]]
            mask = self.color_mask(img, ranges)
            masks.append(mask)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < DETECT_MIN_AREA:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                perim = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * perim, True)
                hull = cv2.contourArea(cv2.convexHull(c))
                shape = classify_shape(area, perim, bw, bh, len(approx), area / max(hull, 1))
                why, d = self.judge(spec, x, y, bw, bh, shape)
                out.append((spec["name"], x, y, bw, bh, area, shape, why, d))
        mask = None
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
        return mask, out

    def detect(self, show=True):
        img = self.frame()
        if img is None:
            return None
        mask, cands = self.candidates(img)
        good = sorted((c for c in cands if c[7] is None), key=lambda c: c[5], reverse=True)
        if show:
            self.draw(img, mask, cands)
        if not good:
            self.streak = 0
            return None
        self.streak += 1
        if self.streak < DETECT_CONFIRM:
            return None
        detections = []
        for candidate in good:
            name, x, y, bw, bh = candidate[:5]
            bearing = math.atan((self.w / 2 - (x + bw / 2)) / self.f)
            detections.append((name, bearing, candidate[8]))
        return detections

    def draw(self, img, mask, cands):
        vis = img.copy()
        hy = int(self.horizon)
        cv2.line(vis, (0, hy), (self.w, hy), (0, 255, 255), 1)
        for (name, x, y, bw, bh, area, shape, why, d) in cands:
            ok = why is None
            col = (255, 0, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2 if ok else 1)
            txt = f"{name} {d:.1f}m" if ok else why
            cv2.putText(vis, txt, (x, max(y - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        if mask is not None:
            vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        self.last_vis = vis


# =============================================================================
class Mission:
    def __init__(self):
        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        self.dt = self.ts / 1000.0
        R = self.robot
        self.lm = R.getDevice("left wheel motor")
        self.rm = R.getDevice("right wheel motor")
        for m in (self.lm, self.rm):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.le = self.lm.getPositionSensor()
        self.re = self.rm.getPositionSensor()
        self.le.enable(self.ts)
        self.re.enable(self.ts)
        self.lidar = R.getDevice("LDS-01")
        self.lidar.enable(self.ts)
        self.nrays = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        # kit convention: index 180 = front, 90 = left, 0 = back
        self.angles = (fov / 2 - np.arange(self.nrays) * fov / self.nrays).astype(np.float32)
        self.rmax = self.lidar.getMaxRange()
        self.gyro = None
        try:
            self.gyro = R.getDevice("gyro")
            self.gyro.enable(self.ts)
        except Exception:
            self.gyro = None
        self.camera = R.getDevice("camera")
        self.camera.enable(self.ts)
        self.cam_half_fov = 0.9 * self.camera.getFov() / 2

        self.grid = Grid()
        self.planner = Planner(self.grid)
        self.det = Detector(self.camera)
        self.detect_every = 3 if any("yolo" in t for t in TARGETS) else DETECT_EVERY

        self.t = 0.0
        self.x = self.y = self.th = 0.0
        self.prev_enc = None
        self.state = "INIT"
        self.state_t = 0.0
        self.spun = 0.0
        self.path = None
        self.path_i = 0
        self.goal = None
        self.goal_t = 0.0
        self.look_at = None
        self.look_until = -1.0
        self.last_plan = -1e9
        self.blacklist = []
        self.found = []
        self.target_name = "?"
        self.target_est = None
        self.target_seen_t = -1e9
        self.target_dist = 99.0
        self.target_bearing = 0.0
        self.blocked_t = 0.0
        self.prog_pose = (0.0, 0.0)
        self.prog_t = 0.0
        self.recover_until = -1.0
        self.scan_xy = np.zeros((0, 2), np.float32)
        self.step_n = 0

    # ------------------------------------------------------------- sensing
    def update_odometry(self):
        l, r = self.le.getValue(), self.re.getValue()
        if math.isnan(l) or math.isnan(r):
            return 0.0
        if self.prev_enc is None:
            self.prev_enc = (l, r)
            return 0.0
        dl = (l - self.prev_enc[0]) * WHEEL_RADIUS
        dr = (r - self.prev_enc[1]) * WHEEL_RADIUS
        self.prev_enc = (l, r)
        ds = (dl + dr) / 2
        dth = (dr - dl) / WHEEL_SEPARATION
        if self.gyro is not None:
            gz = self.gyro.getValues()[2]
            if not math.isnan(gz):
                dth = gz * self.dt
        mid = self.th + dth / 2
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.th = wrap(self.th + dth)
        return dth

    def update_scan(self):
        r = np.asarray(self.lidar.getRangeImage(), np.float32)
        if r.size != self.nrays:
            return
        self.grid.update(self.x, self.y, self.th, r, self.angles, self.rmax, self.cam_half_fov)
        ok = np.isfinite(r) & (r > 0.1) & (r < self.rmax)
        self.scan_xy = np.stack([r[ok] * np.cos(self.angles[ok]),
                                 r[ok] * np.sin(self.angles[ok])], 1)

    def front_clearance(self):
        p = self.scan_xy
        if len(p) == 0:
            return 99.0
        corr = (p[:, 0] > 0) & (np.abs(p[:, 1]) < ROBOT_RADIUS + 0.05)
        return float(p[corr, 0].min()) if corr.any() else 99.0

    def rear_clear(self):
        p = self.scan_xy
        corr = (p[:, 0] < 0) & (np.abs(p[:, 1]) < ROBOT_RADIUS + 0.05)
        return (not corr.any()) or (-p[corr, 0].max() > 0.30)

    # ------------------------------------------------------------- motion
    def drive(self, v, w):
        w = max(-W_MAX, min(W_MAX, w))
        if v > 0:
            front = self.front_clearance()
            if front < STOP_DIST:
                v = 0.0
                self.blocked_t += self.dt
            else:
                self.blocked_t = 0.0
                v *= min(1.0, max(0.45, (front - STOP_DIST) / (SLOW_DIST - STOP_DIST)))
        wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        s = max(1.0, abs(wl) / MAX_WHEEL, abs(wr) / MAX_WHEEL)
        self.lm.setVelocity(wl / s)
        self.rm.setVelocity(wr / s)

    def stop(self):
        self.lm.setVelocity(0.0)
        self.rm.setVelocity(0.0)

    def turn_to(self, x, y):
        """Rotate towards a point. Returns True when facing it."""
        err = wrap(math.atan2(y - self.y, x - self.x) - self.th)
        if abs(err) < 0.15:
            self.stop()
            return True
        self.drive(0.0, max(-1.5, min(1.5, 2.5 * err)))
        return False

    def follow_path(self, v_max=V_MAX):
        """Look-ahead path following. Returns True when the end is reached."""
        path = self.path
        if not path:
            self.stop()
            return True
        best_i, best_d = self.path_i, 1e9
        for i in range(self.path_i, min(len(path), self.path_i + 40)):
            d = math.hypot(path[i][0] - self.x, path[i][1] - self.y)
            if d < best_d:
                best_i, best_d = i, d
        self.path_i = best_i
        end = path[-1]
        if math.hypot(end[0] - self.x, end[1] - self.y) < 0.15:
            self.stop()
            return True
        la = end
        for i in range(best_i, len(path)):
            if math.hypot(path[i][0] - self.x, path[i][1] - self.y) >= LOOKAHEAD:
                la = path[i]
                break
        dx, dy = la[0] - self.x, la[1] - self.y
        xr = math.cos(self.th) * dx + math.sin(self.th) * dy
        yr = -math.sin(self.th) * dx + math.cos(self.th) * dy
        alpha = math.atan2(yr, xr)
        if abs(alpha) > 0.9:
            self.drive(0.0, math.copysign(1.5, alpha))
        else:
            k = 2 * yr / max(xr * xr + yr * yr, 1e-6)
            v = v_max * (1.0 - 0.5 * min(1.0, abs(alpha) / 0.9))
            self.drive(v, v * k)
        return False

    def plan_to(self, x, y, radius=0.0, allow_partial=False):
        rc = self.grid.cell(self.x, self.y)
        self.planner.build(rc)
        cell = self.grid.cell(x, y)
        self.last_plan = self.t
        if radius > 0 and not self.planner.reachable(cell):
            near = self.planner.nearest_reachable(cell, radius)
            if near is None and allow_partial:
                near = self.planner.closest_reachable(cell)
                if near is not None:
                    nx, ny = self.grid.world(*near)
                    if math.hypot(nx - self.x, ny - self.y) < 0.25:
                        near = None
            if near is None:
                return False
            cell = near
        p = self.planner.path_to(cell)
        if not p:
            self.path = None
            return False
        self.path, self.path_i = p, 0
        return True

    def set_state(self, s):
        print(f"[{self.t:7.1f}s] {self.state} -> {s}   pose=({self.x:.2f}, {self.y:.2f}, "
              f"{math.degrees(self.th):.0f}deg)  found={len(self.found)}/{N_TARGETS or 'all'}")
        self.state = s
        self.state_t = self.t
        self.path = None
        self.goal = None
        self.look_until = -1.0
        self.prog_pose, self.prog_t = (self.x, self.y), self.t
        self.blocked_t = 0.0

    # ------------------------------------------------------------- stuck handling
    def check_stuck(self):
        if self.t < self.recover_until:
            return True
        if self.t < self.look_until:
            self.prog_t = self.t
            return False
        if math.hypot(self.x - self.prog_pose[0], self.y - self.prog_pose[1]) > 0.08:
            self.prog_pose, self.prog_t = (self.x, self.y), self.t
        if (self.t - self.prog_t > 8.0) or (self.blocked_t > 5.0):
            print(f"[{self.t:7.1f}s] stuck -> recovery")
            if self.goal is not None:
                self.blacklist.append((self.goal[0], self.goal[1], 0.25))
            self.recover_until = self.t + 2.0
            self.prog_pose, self.prog_t = (self.x, self.y), self.t + 2.0
            self.blocked_t = 0.0
            self.path = None
            self.goal = None
            return True
        return False

    def recovery_motion(self):
        if self.recover_until - self.t > 1.2 and self.rear_clear():
            self.drive(-0.08, 0.0)
        else:
            self.drive(0.0, 1.2)

    # ------------------------------------------------------------- detection
    def handle_detection(self):
        if self.step_n % self.detect_every:
            return self.t - self.target_seen_t < 0.2
        detections = self.det.detect(show=SHOW_CAMERA)
        if not detections:
            return False
        for name, bearing, dist in detections:
            a = self.th + bearing
            tx, ty = self.x + dist * math.cos(a), self.y + dist * math.sin(a)
            if any(math.hypot(tx - fx, ty - fy) < 0.35 for fx, fy, _ in self.found):
                continue
            if any(math.hypot(tx - bx, ty - by) < br for bx, by, br in self.blacklist if br >= 0.6):
                continue
            self.target_name = name
            self.target_bearing = bearing
            self.target_dist = dist
            break
        else:
            return False
        if self.target_est is None or math.hypot(tx - self.target_est[0], ty - self.target_est[1]) > 1.0:
            self.target_est = (tx, ty)
        else:
            self.target_est = (0.7 * self.target_est[0] + 0.3 * tx, 0.7 * self.target_est[1] + 0.3 * ty)
        self.target_seen_t = self.t
        return True

    # ------------------------------------------------------------- states
    def s_init(self, dth):
        self.spun += abs(dth)
        self.drive(0.0, 1.0)
        if self.spun > 2 * math.pi + 0.3:
            self.set_state("EXPLORE")

    def s_explore(self, seen):
        if seen:
            self.set_state("APPROACH")
            return
        if self.t < self.look_until:           # arrived: look into the unexplored area
            if self.look_at is None or self.turn_to(*self.look_at):
                self.look_until = -1.0
            return
        if self.goal is not None and self.t - self.goal_t > GOAL_TIMEOUT:
            self.blacklist.append((self.goal[0], self.goal[1], 0.6))
            self.goal = None
        if self.goal is None or self.path is None or self.t - self.last_plan > REPLAN_PERIOD:
            self.planner.build(self.grid.cell(self.x, self.y))
            self.last_plan = self.t
            best = self.planner.best_goal(self.blacklist)
            if best is None:
                print(f"[{self.t:7.1f}s] everything explored")
                self.set_state("RETURN")
                return
            cell, centroid = best
            gx, gy = self.grid.world(*cell)
            if self.goal is None or math.hypot(gx - self.goal[0], gy - self.goal[1]) > 0.3:
                self.goal, self.goal_t = (gx, gy), self.t
            self.look_at = centroid
            self.path = self.planner.path_to(cell)
            self.path_i = 0
            if not self.path:
                self.blacklist.append((gx, gy, 0.6))
                self.goal = None
                self.drive(0.0, 1.0)
                return
        if self.follow_path():
            self.blacklist.append((self.goal[0], self.goal[1], 0.6))
            self.goal = None
            self.look_until = self.t + 4.0

    def s_approach(self, seen):
        if self.t - self.state_t > 90:
            print(f"[{self.t:7.1f}s] approach timeout, blacklisting target area")
            if self.target_est:
                self.blacklist.append((self.target_est[0], self.target_est[1], 0.8))
            self.target_est = None
            self.set_state("EXPLORE")
            return
        if seen and self.target_dist < REACH_DIST:
            self.set_state("REACHED")
            return
        if seen and self.target_dist < SERVO_DIST:
            b = self.target_bearing
            self.drive(V_APPROACH if abs(b) < 0.35 else 0.0, 2.5 * b)
            self.path = None
            return
        lost_for = self.t - self.target_seen_t
        if not seen and lost_for < 1.0 and self.target_dist < REACH_DIST + 0.2:
            self.set_state("REACHED")
            return
        if self.target_est is None:
            self.set_state("EXPLORE")
            return
        if self.path is None or self.t - self.last_plan > REPLAN_PERIOD:
            if not self.plan_to(*self.target_est, radius=0.6, allow_partial=True):
                print(f"[{self.t:7.1f}s] cannot plan to target estimate")
                self.blacklist.append((self.target_est[0], self.target_est[1], 0.8))
                self.target_est = None
                self.set_state("EXPLORE")
                return
        if self.follow_path(V_MAX * 0.8):
            if lost_for > 3.0:
                self.drive(0.0, 1.0)
                if lost_for > 12.0:
                    self.blacklist.append((self.target_est[0], self.target_est[1], 0.8))
                    self.target_est = None
                    self.set_state("EXPLORE")

    def s_reached(self):
        self.stop()
        if self.t - self.state_t > 1.5:
            est = self.target_est or (self.x, self.y)
            self.found.append((est[0], est[1], self.target_name))
            print(f"[{self.t:7.1f}s] *** TARGET {len(self.found)} ({self.target_name}) reached at "
                  f"({est[0]:.2f}, {est[1]:.2f})")
            self.target_est = None
            done = N_TARGETS is not None and len(self.found) >= N_TARGETS
            self.set_state("RETURN" if done else "EXPLORE")

    def s_return(self):
        d_home = math.hypot(self.x, self.y)
        if 0.12 <= d_home < 0.40 and self.front_clearance() > STOP_DIST:
            if self.turn_to(0.0, 0.0):            # last few cm: drive straight to the start
                self.drive(0.16, 0.0)
            return
        if d_home < 0.12:
            err = wrap(0.0 - self.th)
            if abs(err) < 0.05:
                self.stop()
                self.set_state("DONE")
            else:
                self.drive(0.0, max(-0.8, min(0.8, 2.0 * err)))
            return
        if self.path is None or self.t - self.last_plan > REPLAN_PERIOD:
            if not self.plan_to(0.0, 0.0, radius=0.3):
                self.drive(0.0, 0.6)
                return
        self.follow_path()

    # ------------------------------------------------------------- display
    def draw_map(self):
        g = self.grid
        r0, r1, c0, c1 = g.bbox([g.cell(self.x, self.y)])
        L = g.L[r0:r1, c0:c1]
        img = np.full(L.shape + (3,), 110, np.uint8)
        free = g.known[r0:r1, c0:c1] & (L < -0.2)
        img[free] = (200, 200, 200)
        img[free & g.cam_seen[r0:r1, c0:c1]] = (255, 255, 255)   # looked at by camera
        img[L > 0.6] = (0, 0, 0)

        def px(x, y):
            r, c = g.cell(x, y)
            return int(c - c0), int(r - r0)
        if self.path:
            for p in self.path:
                cv2.circle(img, px(*p), 0, (255, 128, 0), -1)
        if self.goal:
            cv2.circle(img, px(*self.goal), 2, (255, 0, 255), -1)
        for f in self.found:
            cv2.circle(img, px(f[0], f[1]), 3, (0, 200, 0), -1)
        if self.target_est:
            cv2.circle(img, px(*self.target_est), 3, (0, 0, 255), 1)
        cv2.circle(img, px(0, 0), 3, (0, 180, 255), 1)
        p = px(self.x, self.y)
        cv2.circle(img, p, 2, (0, 0, 255), -1)
        cv2.line(img, p, (int(p[0] + 6 * math.cos(self.th)), int(p[1] + 6 * math.sin(self.th))),
                 (0, 0, 255), 1)
        img = cv2.flip(img, 0)
        scale = max(1, int(500 / max(img.shape[:2])))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{self.state} t={self.t:.0f}s found {len(self.found)}/{N_TARGETS or 'all'}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow("map", img)

    # ------------------------------------------------------------- main loop
    def run(self):
        if MODE == "tune":
            return self.run_tune()
        print(f"SAR mission start. targets={[t['name'] for t in TARGETS]} count={N_TARGETS or 'unknown'}")
        while self.robot.step(self.ts) != -1:
            self.t = self.robot.getTime()
            self.step_n += 1
            dth = self.update_odometry()
            self.update_scan()
            if self.step_n < 3:
                continue
            if TIME_LIMIT and self.state in ("EXPLORE", "APPROACH") and \
                    self.t > TIME_LIMIT - RETURN_MARGIN:
                print(f"[{self.t:7.1f}s] time is running out -> RETURN")
                self.set_state("RETURN")

            seen = False
            if self.state in ("INIT", "EXPLORE", "APPROACH"):
                seen = self.handle_detection()

            near_home = self.state == "RETURN" and math.hypot(self.x, self.y) < 0.40
            if self.state in ("EXPLORE", "APPROACH", "RETURN") and not near_home and self.check_stuck():
                self.recovery_motion()
            elif self.state == "INIT":
                self.s_init(dth)
            elif self.state == "EXPLORE":
                self.s_explore(seen)
            elif self.state == "APPROACH":
                self.s_approach(seen)
            elif self.state == "REACHED":
                self.s_reached()
            elif self.state == "RETURN":
                self.s_return()
            elif self.state == "DONE":
                self.stop()

            if SHOW_MAP and self.step_n % 8 == 0:
                self.draw_map()
            if SHOW_CAMERA and self.det.last_vis is not None and self.step_n % 3 == 0:
                cv2.imshow("camera | mask", self.det.last_vis)
            if SHOW_MAP or SHOW_CAMERA:
                cv2.waitKey(1)

    # ------------------------------------------------------------- tune mode
    def run_tune(self):
        """Drive with W/A/S/D (click the 3D view first). Sliders change the colour range of the
        FIRST target type. Press 'p' in the 'tune' window to print a COLOR_PRESETS line.
        Console shows, for each blob: shape class, size check, and the camera height implied by a
        blob of the target's size (put the robot in front of a real target -> use that CAM_HEIGHT)."""
        kb = Keyboard()
        kb.enable(self.ts)
        win = "tune"
        cv2.namedWindow(win)
        spec = next((t for t in TARGETS if "colors" in t), None)
        lo, hi = COLOR_PRESETS[spec["colors"][0]][0] if spec else ((0, 0, 0), (179, 255, 255))
        for name, val, mx in (("H lo", lo[0], 179), ("H hi", hi[0], 179), ("S lo", lo[1], 255),
                              ("S hi", hi[1], 255), ("V lo", lo[2], 255), ("V hi", hi[2], 255)):
            cv2.createTrackbar(name, win, val, mx, lambda v: None)
        print("TUNE MODE: W/A/S/D to drive (click the 3D view), sliders in 'tune' window, "
              "'p' prints the colour preset.")
        last_print = 0.0
        while self.robot.step(self.ts) != -1:
            self.t = self.robot.getTime()
            key = kb.getKey()
            v = w = 0.0
            if key in (ord("W"), ord("w")):
                v = 0.15
            elif key in (ord("S"), ord("s")):
                v = -0.15
            elif key in (ord("A"), ord("a")):
                w = 1.2
            elif key in (ord("D"), ord("d")):
                w = -1.2
            wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
            wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
            self.lm.setVelocity(max(-MAX_WHEEL, min(MAX_WHEEL, wl)))
            self.rm.setVelocity(max(-MAX_WHEEL, min(MAX_WHEEL, wr)))

            g = lambda n: cv2.getTrackbarPos(n, win)
            rng = [((g("H lo"), g("S lo"), g("V lo")), (g("H hi"), g("S hi"), g("V hi")))]
            img = self.det.frame()
            if img is None:
                continue
            specs = [spec] if spec else TARGETS
            mask, cands = self.det.candidates(img, specs, ranges_override=rng if spec else None)
            self.det.draw(img, mask, cands)
            cv2.imshow(win, self.det.last_vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("p"):
                print(f'    "custom": {rng},')
            if self.t - last_print > 1.0 and cands:
                last_print = self.t
                print("-" * 60)
                for (name, x, y, bw, bh, area, shape, why, d) in sorted(cands, key=lambda c: -c[5])[:4]:
                    below = (y + bh) - self.det.horizon
                    d_size = self.det.f * specs[0]["size"] / max(bw, 1)
                    implied_h = d_size * below / self.det.f if below > 0 else float("nan")
                    print(f"{name}: shape={shape:9s} w={bw}px h={bh}px aspect={bh / max(bw, 1):.2f} "
                          f"dist~{d:.2f}m  implied CAM_HEIGHT={implied_h:.3f}  -> {why or 'ACCEPTED'}")


if __name__ == "__main__":
    Mission().run()
