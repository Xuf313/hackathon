"""TurtleBot3 autonomous search & rescue controller for Webots.

The robot starts at a known pose in an unknown apartment, searches for red apples lying on the floor,
confirms each one from close range and drives back to the start once enough apples are found.
It only uses its own sensors: wheel encoders, compass, accelerometer, 2D LiDAR and camera.

Mission state machine:
  SCAN      spin 360 deg to map the surroundings and look around with the camera
  EXPLORE   drive to the most promising unexplored spot (frontier) on the map
  APPROACH  an apple was spotted: drive to it and confirm it up close
  TO_DEST   optional: visit a destination given with --dest=x,y
  HOME      drive back to the start pose
  DONE      mission complete, stand still

On every step a safety layer stops the robot when something is too close in front; recovery
(backing off, or turning toward open space) gets it out when it is stuck.
"""
import heapq
import math
import os
import sys
from collections import deque

import cv2
import numpy as np
from controller import Display, Node, Robot

import semantic


def _arg(name):
    """Value of a '--name=value' controllerArgs entry in the world file, or None."""
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


# ---------------- mission config (world frame, metres) ----------------
START = (-0.3, -7.5, math.pi)       # start pose (x, y, heading), given by the organizers
# optional run-time settings from the world's controllerArgs: "--dest=x,y" and "--count=N"
DEST = tuple(float(v) for v in _arg("dest").split(",")) if _arg("dest") else None
TARGET_COUNT = int(_arg("count")) if _arg("count") else 2      # go home after this many apples (None: search everywhere)
TARGET_NAME = "red apple"
LOOK_SPACING = 1.5                  # do a 360 deg camera look-around every time we reach a new area this far away
VERIFY_DIST = 2.0                   # an apple only counts once it was seen clearly (round, right size) from this close
REACH_DIST = 0.35                   # target reached when this close
FALSE_FORGET = 90.0                 # a false alarm or unreachable target is ignored this long (s), then retried
GOAL_TOL = 0.25                     # dest/home reached when this close
NEAR_GOAL = 0.6                     # ... or this close when walls / obstacles stop the robot getting closer
SHOW_DEBUG = True                   # camera / map / costmap views
SAVE_DEBUG_FILES = False            # also save snapshots (rej_*.jpg, spotted.jpg, *_live.jpg, sem_objects.json)
SHOW_ALL_OBJECTS = False            # also draw rejected red blobs (grey) on the camera view, for debugging
YOLO_SHOW_CONF = 0.4                # YOLO boxes shown on the camera view only from this confidence
YOLO_EVERY = 5                      # run YOLO on every 5th frame (it only feeds the semantic map)
YOLO_WEIGHTS = "yolo11n.pt"         # COCO model, 80 classes
SEMANTIC = True                     # let recognised objects (table, fridge, ...) guide where to search
YOLO_CONF = 0.2                     # low threshold: simulated images get low confidence scores

# ---------------- TurtleBot3 Burger ----------------
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
ROBOT_RADIUS = 0.105
MAX_WHEEL = 6.67
V_MAX = 0.22                         # m/s (TB3 Burger max)
W_MAX = 2.5                          # rad/s
CAM_HEIGHT = 0.073                   # camera height above floor
APPLE_D = 0.10                       # apple diameter (m), used for size-vs-distance checks
CAM_X = 0.02                        # camera forward offset from base

# ---------------- map ----------------
RES = 0.05
MAP_HALF = 20.0                      # the building size is unknown: map 20 m around the start in every direction
X0, Y0 = START[0] - MAP_HALF, START[1] - MAP_HALF
N = int(2 * MAP_HALF / RES)
INFLATE = ROBOT_RADIUS + 0.09        # planned paths keep the robot centre this far from obstacles
SAFE_FRONT = 0.20                    # emergency stop when something is this close in front (from the LiDAR)
SAFE_REAR = 0.25                     # reverse only when nothing is this close behind
LOW_OBJECTS = True                   # use the camera to spot objects too low for the LiDAR to see
LOW_MAX_H = 0.13                     # objects lower than this (m) pass under the LiDAR scan plane
LOW_MIN_SAT = 130                    # strongly coloured pixels (any hue) can belong to an object on the floor
TILT_MAX = math.radians(6.0)         # tilting more than this while driving means the wheels are on something
                                     # too low for the LiDAR to see (carpet edge, step): back off and avoid it

robot = Robot()
dt_ms = int(robot.getBasicTimeStep())
DT = dt_ms / 1000.0

lm = robot.getDevice("left wheel motor")
rm = robot.getDevice("right wheel motor")
for m in (lm, rm):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
MAX_WHEEL = min(lm.getMaxVelocity(), rm.getMaxVelocity())   # read the real limit from the motors
le = lm.getPositionSensor(); le.enable(dt_ms)
re_ = rm.getPositionSensor(); re_.enable(dt_ms)
compass = robot.getDevice("compass"); compass.enable(dt_ms)
accel = robot.getDevice("accelerometer"); accel.enable(dt_ms)
lidar = robot.getDevice("LDS-01"); lidar.enable(dt_ms)
camera = robot.getDevice("camera"); camera.enable(dt_ms)
CW, CH = camera.getWidth(), camera.getHeight()
FOCAL = (CW / 2) / math.tan(camera.getFov() / 2)
CAM_HALF_FOV = camera.getFov() / 2
NBEAM = lidar.getHorizontalResolution()
LMAX = lidar.getMaxRange()
BEAM_ANG = math.pi - np.arange(NBEAM) * 2 * math.pi / NBEAM   # idx 180 = front, 90 = left


HERE = os.path.dirname(os.path.abspath(__file__))   # log / debug files go next to this controller
_LOG = open(os.path.join(HERE, "sar.log"), "w")
_print = print


def print(*a):
    """print() that also writes every line to sar.log."""
    _print(*a)
    _LOG.write(" ".join(map(str, a)) + "\n"); _LOG.flush()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------- state ----------------
x, y, th = START
logodds = np.zeros((N, N), np.float32)
known = np.zeros((N, N), bool)
cam_seen = np.zeros((N, N), bool)     # floor cells the camera has actually looked at
hazard = np.zeros((N, N), bool)       # low obstacles found by tilting (the LiDAR can't see them)
_grav = {"ref": None, "f": None, "n": 0}  # gravity on flat floor, filtered reading, tilted steps in a row
last_cmd = [0.0, 0.0]                # last (v, w) sent to the wheels
CAM_RANGE = 3.0                       # floor counts as "searched" only this close (detection itself works to 6 m)
NEAR_OBST = 0.30                      # floor this close to walls/furniture (where objects get tucked away) ...
NEAR_OBST_RANGE = 1.8                 # ... only counts as searched when seen from closer than this
_wall_dist = [None, -1]               # cached distance-to-obstacle map [array, step computed]
compass_off = None
compass_sign = None
prev_l = prev_r = None
spin_acc = 0.0
raw_prev = None


def compass_raw():
    c = compass.getValues()
    return math.atan2(c[0], c[1])


def to_cell(px, py):
    return int((py - Y0) / RES), int((px - X0) / RES)


def to_world(r, c):
    return X0 + (c + 0.5) * RES, Y0 + (r + 0.5) * RES


def update_odometry():
    """Dead reckoning: distance from the wheel encoders, heading from the compass."""
    global x, y, th, prev_l, prev_r, compass_off, compass_sign, spin_acc, raw_prev
    l, r = le.getValue(), re_.getValue()
    if prev_l is None:
        prev_l, prev_r = l, r
        raw_prev = compass_raw()
        return
    dl, dr = (l - prev_l) * WHEEL_RADIUS, (r - prev_r) * WHEEL_RADIUS
    prev_l, prev_r = l, r
    ds, dth_enc = (dl + dr) / 2, (dr - dl) / WHEEL_SEPARATION
    raw = compass_raw()
    if compass_sign is None:
        # the compass sign and offset are unknown: calibrate them against the encoders during the first spin
        spin_acc += dth_enc
        if abs(spin_acc) > 0.3:
            d = wrap(raw - raw_prev)
            compass_sign = 1.0 if d * spin_acc > 0 else -1.0
            compass_off = wrap(START[2] + spin_acc - compass_sign * raw)
            print(f"[sar] compass calibrated sign={compass_sign:+.0f}")
        th_new = wrap(th + dth_enc)
    else:
        th_new = wrap(compass_sign * raw + compass_off)
        raw_prev = raw
    mid = th + wrap(th_new - th) / 2
    x += ds * math.cos(mid)
    y += ds * math.sin(mid)
    th = th_new
    if compass_sign is None:
        return


def update_map(ranges):
    """Occupancy grid update: cells along each LiDAR beam become freer, the cell the beam hits more occupied."""
    idx = np.arange(0, NBEAM, 2)
    r = ranges[idx]
    hit = np.isfinite(r) & (r < LMAX - 0.05) & (r > 0.12)
    r = np.where(np.isfinite(r), np.clip(r, 0, LMAX), LMAX)
    ang = th + BEAM_ANG[idx]
    S = np.arange(0, LMAX, RES)[None, :]
    free_mask = S < (r[:, None] - RES)
    px = x + S * np.cos(ang)[:, None]
    py = y + S * np.sin(ang)[:, None]
    fr = ((py[free_mask] - Y0) / RES).astype(int)
    fc = ((px[free_mask] - X0) / RES).astype(int)
    ok = (fr >= 0) & (fr < N) & (fc >= 0) & (fc < N)
    flat = np.unique(fr[ok] * N + fc[ok])
    logodds.flat[flat] -= 0.35
    known.flat[flat] = True
    hx = x + r[hit] * np.cos(ang[hit])
    hy = y + r[hit] * np.sin(ang[hit])
    hr = ((hy - Y0) / RES).astype(int)
    hc = ((hx - X0) / RES).astype(int)
    ok = (hr >= 0) & (hr < N) & (hc >= 0) & (hc < N)
    flat = np.unique(hr[ok] * N + hc[ok])
    logodds.flat[flat] += 0.9
    known.flat[flat] = True
    np.clip(logodds, -3, 4, out=logodds)


# ---------------- localization correction: LiDAR scan matching against the map ----------------
MATCH_EVERY = 5                       # control steps between corrections
MATCH_SHIFTS = np.arange(-0.15, 0.151, 0.025)
_match = {"dist": None, "step": -1, "n": 0, "total": 0.0}


def scan_match(ranges):
    """Odometry drifts (wheel slip). Try small x/y shifts of the pose and keep the one where the
    current scan lands best on walls already in the map (truncated distance field). Heading comes
    from the compass, so only position is corrected."""
    global x, y
    occ = logodds > 1.5                               # well-established walls only
    if occ.sum() < 200:
        return
    step_i = int(robot.getTime() / DT)
    if _match["dist"] is None or step_i - _match["step"] >= 25:
        _match["dist"] = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * RES
        _match["step"] = step_i
    idx = np.arange(0, NBEAM, 3)
    r = ranges[idx]
    ok = np.isfinite(r) & (r > 0.15) & (r < 3.3)
    if ok.sum() < 30:
        return
    a = th + BEAM_ANG[idx][ok]
    bx, by = r[ok] * np.cos(a), r[ok] * np.sin(a)
    D = _match["dist"]

    def score(dx, dy):
        c = ((x + dx + bx - X0) / RES).astype(int); rr = ((y + dy + by - Y0) / RES).astype(int)
        inside = (rr >= 0) & (rr < N) & (c >= 0) & (c < N)
        d = np.full(bx.shape, 0.3)
        d[inside] = np.minimum(D[rr[inside], c[inside]], 0.3)    # truncated: people / new objects don't dominate
        return float(d.mean())

    base = score(0.0, 0.0)
    best = (base, 0.0, 0.0)
    for dx in MATCH_SHIFTS:
        for dy in MATCH_SHIFTS:
            sc = score(dx, dy)
            if sc < best[0]:
                best = (sc, dx, dy)
    if best[0] < base - 0.004:                        # clearly better alignment -> move (damped)
        x += 0.5 * best[1]; y += 0.5 * best[2]
        _match["n"] += 1; _match["total"] += 0.5 * math.hypot(best[1], best[2])


def update_cam_coverage(ranges):
    """Mark floor cells inside the camera's view cone (occluded by LiDAR hits) as searched."""
    k = int(math.degrees(CAM_HALF_FOV) * NBEAM / 360) - 2
    idx = np.arange(NBEAM // 2 - k, NBEAM // 2 + k + 1)
    r = ranges[idx]
    r = np.where(np.isfinite(r), np.minimum(r, CAM_RANGE), CAM_RANGE)
    ang = th + BEAM_ANG[idx]
    S = np.arange(0.15, CAM_RANGE, RES)[None, :]
    msk = S < r[:, None]
    px = (x + S * np.cos(ang)[:, None])[msk]
    py = (y + S * np.sin(ang)[:, None])[msk]
    sd = np.broadcast_to(S, msk.shape)[msk]                   # viewing distance of each cell
    rr = ((py - Y0) / RES).astype(int); cc = ((px - X0) / RES).astype(int)
    ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
    rr, cc, sd = rr[ok], cc[ok], sd[ok]
    # corners next to furniture are occluded from afar (LiDAR beams slip past cabinet edges higher up
    # than a floor object): those cells need a close look
    step_i = int(robot.getTime() / DT)
    if _wall_dist[0] is None or step_i - _wall_dist[1] >= 10:
        occ = (logodds > 0.6).astype(np.uint8)
        _wall_dist[0] = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * RES
        _wall_dist[1] = step_i
    close_ok = (_wall_dist[0][rr, cc] > NEAR_OBST) | (sd < NEAR_OBST_RANGE)
    cam_seen[rr[close_ok], cc[close_ok]] = True


def mark_seen_around(px, py, rad=0.4):
    """Treat the area around a spot as searched, so it isn't picked as a goal again."""
    r0, c0 = to_cell(px, py); k = int(rad / RES)
    cam_seen[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1] = True


def cost_maps():
    """Planning costs: cells too close to walls or hazards are blocked, cells near them cost extra."""
    occ = ((logodds > 0.6) | hazard).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * RES
    blocked = dist < INFLATE
    penalty = np.clip(0.55 - dist, 0, None) * 14.0      # shortest path, but kept away from walls
    return blocked, penalty


NB8 = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
       (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


def free_start(blocked, s):
    """If the robot's own cell is inside inflation, hop to the nearest free cell."""
    if not blocked[s]:
        return s
    q = deque([s]); seen = {s}
    while q:
        c = q.popleft()
        if not blocked[c]:
            return c
        for dr, dc, _ in NB8[:4]:
            n = (c[0] + dr, c[1] + dc)
            if 0 <= n[0] < N and 0 <= n[1] < N and n not in seen and abs(n[0]-s[0]) < 8 and abs(n[1]-s[1]) < 8:
                seen.add(n); q.append(n)
    return s


def astar(goal_xy, blocked, penalty):
    """A* on the grid from the robot to goal_xy; returns a list of world points, or None if unreachable."""
    s = free_start(blocked, to_cell(x, y))
    g = to_cell(*goal_xy)
    if not (0 <= g[0] < N and 0 <= g[1] < N):
        return None
    if blocked[g]:  # goal inside inflation (e.g. target next to furniture): accept nearest free cell
        g = free_start(blocked, g)
    gs = {s: 0.0}; parent = {s: None}
    pq = [(0.0, s)]
    while pq:
        _, c = heapq.heappop(pq)
        if c == g:
            break
        for dr, dc, w in NB8:
            n = (c[0] + dr, c[1] + dc)
            if not (0 <= n[0] < N and 0 <= n[1] < N) or blocked[n]:
                continue
            ng = gs[c] + w * (1 + penalty[n]) + (0.3 if not known[n] else 0)   # prefer known floor
            if ng < gs.get(n, 1e18):
                gs[n] = ng; parent[n] = c
                h = math.hypot(n[0] - g[0], n[1] - g[1])
                heapq.heappush(pq, (ng + h, n))
    if g not in parent:
        return None
    path = []
    c = g
    while c is not None:
        path.append(to_world(*c)); c = parent[c]
    return path[::-1]


def nearest_frontier(blocked, banned):
    """Next place to search. Candidates are the edge of the known map plus free floor the camera hasn't
    looked at yet, found by BFS so their path distance is known; the semantic score picks one."""
    free = known & (logodds < -0.5)
    unk = (~known).astype(np.uint8)
    front = free & (cv2.dilate(unk, np.ones((3, 3), np.uint8)) > 0)
    unseen = cv2.erode((free & ~cam_seen).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    front = front | unseen
    s = free_start(blocked, to_cell(x, y))
    q = deque([(s, 0)]); seen = np.zeros((N, N), bool); seen[s] = True
    cands, buckets = [], set()
    while q:
        c, d = q.popleft()
        if front[c]:
            wx, wy = to_world(*c)
            bk = (int(wx / 0.5), int(wy / 0.5))       # one candidate per 0.5 m bucket (the closest one)
            if bk not in buckets and math.hypot(wx - x, wy - y) > 0.5 and \
                    all(math.hypot(wx - bx, wy - by) > 0.6 for bx, by in banned):
                buckets.add(bk); cands.append((wx, wy, d * RES))
                if len(cands) >= 80:
                    break
        for dr, dc, _ in NB8[:4]:
            n = (c[0] + dr, c[1] + dc)
            if 0 <= n[0] < N and 0 <= n[1] < N and not seen[n] and not blocked[n] and known[n]:
                seen[n] = True; q.append((n, d + 1))
    if not cands:
        return None
    if not SEMANTIC:
        return cands[0][:2]                            # plain nearest-frontier exploration
    g = semantic.pick(cands, sem)
    if g != cands[0][:2]:
        print(f"[sar] semantic: chose ({g[0]:.1f},{g[1]:.1f}) over nearest ({cands[0][0]:.1f},{cands[0][1]:.1f})"
              f" likelihood={sem.likelihood(*g):.2f}")
    return g

# ---------------- YOLO object detector (feeds the semantic map) ----------------
yolo = None
yolo_boxes = []                      # cached [(x1, y1, x2, y2, name, conf)]
if SHOW_DEBUG or SEMANTIC:
    try:
        import torch
        from ultralytics import YOLO
        YOLO_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
        _wdir = os.path.join(HERE, "../../models/YOLO")
        yolo = YOLO(os.path.join(_wdir, YOLO_WEIGHTS))
        yolo.to(YOLO_DEV)
        print(f"[sar] YOLO loaded on {YOLO_DEV}: {YOLO_WEIGHTS} ({len(yolo.names)} classes)")
    except Exception as e:           # mission still works without YOLO
        print(f"[sar] YOLO disabled: {e}")


def run_yolo(frame):
    res = yolo.predict(source=frame, conf=YOLO_CONF, iou=0.5, device=YOLO_DEV, verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        if (x2 - x1) * (y2 - y1) > 0.35 * CW * CH and c < 0.5:
            continue   # huge low-confidence boxes are floor/wall hallucinations ("bed" on the floor)
        out.append((int(x1), int(y1), int(x2), int(y2), res.names[int(k)], c))
    return out


def yolo_color(name):
    h = sum(map(ord, name)) * 37 % 180
    b, g, r = cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _overlaps(a, b):
    """True if two (x1, y1, x2, y2) boxes overlap by more than 30 % of the smaller one."""
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h > 0.3 * max(small, 1)


def draw_yolo(frame, taken):
    """Label the objects YOLO is reasonably sure about; boxes in `taken` already have a colour-based label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    shown = [b for b in yolo_boxes if b[5] >= YOLO_SHOW_CONF and not any(_overlaps(b[:4], t) for t in taken)]
    for id_, (x1, y1, x2, y2, name, c) in enumerate(shown, 1):
        col = yolo_color(name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"id:{id_} {name} {c:.2f}"
        (tw, th_), _ = cv2.getTextSize(label, font, 0.45, 1)
        ty = y1 - 4 if y1 - th_ - 8 >= 0 else y1 + th_ + 6
        cv2.rectangle(frame, (x1, ty - th_ - 4), (x1 + tw + 6, ty + 3), col, -1)
        cv2.putText(frame, label, (x1 + 3, ty), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


sem = semantic.SemanticMap()
if SEMANTIC:
    print("[sar] semantic exploration ON")

rejects = []
_last_rej = [-1e9]


def detect_target(frame):
    """Find a red apple on the floor -> (bearing, distance, bbox, confidence) or None.
    Red blobs must look round, sit on the floor, have the right size for their distance and not be part
    of a taller red object; rejected blobs are kept in `rejects` for the log."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 130, 50), (8, 255, 255)) | cv2.inRange(hsv, (172, 130, 50), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    rejects.clear()
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 15:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * a / (per * per + 1e-6)
        bottom = by + bh
        why = None
        if circ < 0.5 or not (0.6 < bw / bh < 1.6):
            why = f"shape circ={circ:.2f} ar={bw/bh:.2f}"
        elif bottom < CH / 2 + 3:
            why = "above floor"
        else:
            # apple is ~0.10 m wide: its pixel size must match the ground-plane distance estimate
            dist = CAM_HEIGHT / math.tan(math.atan2(bottom - CH / 2, FOCAL)) + CAM_X
            expect = FOCAL * APPLE_D / dist
            top = max(0, int(by - 3 * expect))
            above = m[top:max(top, by - 2), bx:bx + bw]
            if not (0.55 * expect < max(bw, bh) < 1.7 * expect) or by < CH / 2 - 1.0 * expect:
                why = f"size {max(bw, bh)} vs {expect:.0f}"
            elif bx <= 2 or bx + bw >= CW - 2:
                why = "edge"   # cut off at the image edge: can't judge shape
            elif above.size and (above > 0).mean() > 0.08:
                why = "tall"   # red continues upward -> part of a tall object (fire extinguisher, cabinet)
        if why:
            rejects.append((why, (bx, by, bw, bh), a))
            continue
        if best is None or a > best[0]:
            # heuristic confidence: roundness x how well the pixel size matches an apple at that distance
            size_fit = max(0.0, 1 - abs(math.log(max(bw, bh) / expect)))
            conf = min(1.0, circ) * (0.5 + 0.5 * size_fit)
            best = (a, bx, by, bw, bh, conf)
    if best is None:
        return None
    _, bx, by, bw, bh, conf = best
    cx, bottom = bx + bw / 2, by + bh
    bearing = math.atan2(CW / 2 - cx, FOCAL)
    depress = math.atan2(bottom - CH / 2, FOCAL)
    dist = CAM_HEIGHT / math.tan(depress) + CAM_X + 0.03     # from where the apple touches the floor
    return bearing, dist, (bx, by, bw, bh), conf


def drive(v, w):
    """Send (v, w) as wheel speeds; both wheels are scaled down together if one would exceed its limit."""
    v = max(-V_MAX, min(V_MAX, v)); w = max(-W_MAX, min(W_MAX, w))
    wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    lim = 0.99 * MAX_WHEEL           # stay just under the motor limit (float rounding triggers Webots warnings)
    k = max(1.0, abs(wl) / lim, abs(wr) / lim)
    lm.setVelocity(max(-lim, min(lim, wl / k))); rm.setVelocity(max(-lim, min(lim, wr / k)))
    last_cmd[0], last_cmd[1] = v, w


# ---------------- local costmap (rolling window, robot frame) + DWA local planner ----------------
LOCAL_SIZE = 3.0                     # 3 m x 3 m window centred on the robot
LN = int(LOCAL_SIZE / RES)
LOOKAHEAD = 0.6
DWA_T, DWA_DT = 1.5, 0.1             # simulate each candidate command 1.5 s ahead
DWA_V = np.linspace(0.0, V_MAX, 6)
DWA_W = np.linspace(-W_MAX, W_MAX, 15)
COLLIDE = ROBOT_RADIUS + 0.045                      # trajectories closer than this to an obstacle are rejected
local_occ = np.zeros((LN, LN), np.uint8)
local_dist = np.full((LN, LN), LOCAL_SIZE, np.float32)
local_blocked = [False]
dwa_viz = {"cands": None, "best": None, "ok": None}

# precompute candidate trajectories (robot frame, start at origin, heading +x)
_V, _W = np.meshgrid(DWA_V, DWA_W, indexing="ij")
_V, _W = _V.ravel(), _W.ravel()
_steps = int(DWA_T / DWA_DT)
TRAJ = np.zeros((len(_V), _steps, 2), np.float32)
_px = np.zeros(len(_V)); _py = np.zeros(len(_V)); _pth = np.zeros(len(_V))
for k in range(_steps):
    _pth = _pth + _W * DWA_DT
    _px = _px + _V * np.cos(_pth) * DWA_DT
    _py = _py + _V * np.sin(_pth) * DWA_DT
    TRAJ[:, k, 0], TRAJ[:, k, 1] = _px, _py


def update_local_costmap(ranges):
    """Local costmap around the robot, rebuilt from the current scan only (so a walking person leaves no
    ghost obstacles), plus the remembered low-obstacle hazards."""
    local_occ[:] = 0
    ok = np.isfinite(ranges) & (ranges > 0.12) & (ranges < LOCAL_SIZE)
    px = ranges[ok] * np.cos(BEAM_ANG[ok]); py = ranges[ok] * np.sin(BEAM_ANG[ok])
    c = ((px + LOCAL_SIZE / 2) / RES).astype(int); r = ((py + LOCAL_SIZE / 2) / RES).astype(int)
    m = (r >= 0) & (r < LN) & (c >= 0) & (c < LN)
    local_occ[r[m], c[m]] = 1
    k = int(LOCAL_SIZE / RES); r0, c0 = to_cell(x, y)
    ra, ca = max(0, r0 - k), max(0, c0 - k)
    hr, hc = np.nonzero(hazard[ra:r0 + k + 1, ca:c0 + k + 1])
    if hr.size:
        wx = X0 + (hc + ca + 0.5) * RES - x; wy = Y0 + (hr + ra + 0.5) * RES - y
        lx = wx * math.cos(th) + wy * math.sin(th); ly = -wx * math.sin(th) + wy * math.cos(th)
        c = ((lx + LOCAL_SIZE / 2) / RES).astype(int); r = ((ly + LOCAL_SIZE / 2) / RES).astype(int)
        # patches the robot is already standing on can't be avoided: leave them out so it can drive away
        m = (r >= 0) & (r < LN) & (c >= 0) & (c < LN) & (np.hypot(lx, ly) > 0.20)
        local_occ[r[m], c[m]] = 1
    local_dist[:] = cv2.distanceTransform(1 - local_occ, cv2.DIST_L2, 5) * RES


def dwa(lx, ly):
    """Pick (v, w) whose 1.5 s trajectory is collision-free and best tracks the look-ahead point."""
    c = ((TRAJ[..., 0] + LOCAL_SIZE / 2) / RES).astype(int)
    r = ((TRAJ[..., 1] + LOCAL_SIZE / 2) / RES).astype(int)
    inside = (r >= 0) & (r < LN) & (c >= 0) & (c < LN)
    d = np.where(inside, local_dist[np.clip(r, 0, LN - 1), np.clip(c, 0, LN - 1)], LOCAL_SIZE)
    clear = d.min(axis=1)
    ok = clear > COLLIDE
    end = TRAJ[:, -1, :]
    goal_cost = np.hypot(end[:, 0] - lx, end[:, 1] - ly)
    # end close to the look-ahead point, stay away from obstacles, prefer driving faster
    cost = 1.0 * goal_cost + 1.2 * (1 - np.minimum(clear, 0.6) / 0.6) - 0.25 * _V / V_MAX
    cost[~ok] = np.inf
    dwa_viz["cands"], dwa_viz["ok"] = TRAJ, ok
    if not ok.any():
        dwa_viz["best"] = None
        return None
    i = int(np.argmin(cost))
    dwa_viz["best"] = i
    return float(_V[i]), float(_W[i])


COSTMAP_VIEW = 6.0                   # side of the costmap view window (m), centred on the robot


def costmap_view(path, size=300):
    """Global + local costmap around the robot (north up).
    Global layer: walls, the no-go margin around them, extra cost near them, low obstacles found by camera/tilt.
    Local layer: obstacles in the current LiDAR scan (e.g. the pedestrian), DWA trajectories and the chosen one."""
    k = int(COSTMAP_VIEW / RES / 2); pad = int(0.5 / RES)
    r0, c0 = to_cell(x, y)
    ra, rb = max(0, r0 - k - pad), min(N, r0 + k + pad + 1)
    ca, cb = max(0, c0 - k - pad), min(N, c0 + k + pad + 1)
    occ = ((logodds[ra:rb, ca:cb] > 0.6) | hazard[ra:rb, ca:cb]).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * RES
    win = np.full((2 * k + 1, 2 * k + 1, 3), UI_BG, np.float32)
    sr, sc = slice(r0 - k - ra, r0 - k - ra + 2 * k + 1), slice(c0 - k - ca, c0 - k - ca + 2 * k + 1)
    d = dist[sr, sc]; kn = known[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
    lo = logodds[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]; hz = hazard[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
    free = kn & (lo < 0)
    win[free] = UI_FREE
    near = free & (d < 0.45)                                        # extra path cost near obstacles
    w = np.clip((0.45 - d) / 0.45, 0, 1)[..., None]
    win[near] = (win * (1 - 0.6 * w) + UI_COST * 0.6 * w)[near]
    win[d < INFLATE] = UI_INFL                                      # no-go margin for the robot centre
    win[lo > 0.6] = UI_WALL
    win[hz] = UI_HAZARD
    img = cv2.resize(cv2.flip(win, 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    s_ = size / (2 * k + 1)

    def px(wx, wy):                                                 # world -> view pixel (north up)
        return int((wx - x) / RES * s_ + size / 2), int(-(wy - y) / RES * s_ + size / 2)

    ct, st = math.cos(th), math.sin(th)

    def rob(lx, ly):                                                # robot frame -> world
        return x + lx * ct - ly * st, y + lx * st + ly * ct

    if path and len(path) > 1:
        cv2.polylines(img, [np.int32([px(*p) for p in path])], False, UI_PATH, 2, cv2.LINE_AA)
    if dwa_viz["cands"] is not None:
        for i, tr in enumerate(dwa_viz["cands"][::3]):
            if dwa_viz["ok"][i * 3]:
                cv2.polylines(img, [np.int32([px(*rob(*p)) for p in tr[::3]])], False, UI_TRAJ, 1, cv2.LINE_AA)
        if dwa_viz["best"] is not None:
            tr = dwa_viz["cands"][dwa_viz["best"]]
            cv2.polylines(img, [np.int32([px(*rob(*p)) for p in tr[::2]])], False, UI_BEST, 3, cv2.LINE_AA)
    rr, cc = np.nonzero(local_occ)                                  # current scan, robot frame
    for ly_, lx_ in zip((rr + 0.5) * RES - LOCAL_SIZE / 2, (cc + 0.5) * RES - LOCAL_SIZE / 2):
        cv2.circle(img, px(*rob(lx_, ly_)), 1, UI_SCAN, -1)
    p = np.array(px(x, y), np.float32)
    fwd = np.array((ct, -st), np.float32); left = np.array((-fwd[1], fwd[0]), np.float32)
    r_ = max(5.0, ROBOT_RADIUS / RES * s_)
    tri = np.int32([p + fwd * r_ * 1.4, p - fwd * r_ * 0.8 + left * r_, p - fwd * r_ * 0.8 - left * r_])
    cv2.fillPoly(img, [tri], UI_ROBOT, cv2.LINE_AA)
    _badge(img, f"costmap {COSTMAP_VIEW:.0f} m  global + local")
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), UI_FRAME, 2)
    return img


def front_clearance(ranges, half_deg=35):
    """Closest LiDAR range within +-half_deg of straight ahead."""
    i0 = NBEAM // 2
    k = int(half_deg * NBEAM / 360)
    seg = ranges[i0 - k:i0 + k + 1]
    seg = seg[np.isfinite(seg) & (seg > 0.11)]
    return float(seg.min()) if seg.size else LMAX


def tilt():
    """Angle (rad) between the current gravity direction and the one measured standing flat at the start."""
    a = np.array(accel.getValues(), dtype=float)
    if not np.all(np.isfinite(a)) or np.linalg.norm(a) < 5.0:
        return 0.0                                            # sensor not ready yet
    _grav["raw"] = a
    # wheel speeds change instantly in Webots, so starting / braking gives one-step jolts: low-pass filter
    _grav["f"] = a if _grav["f"] is None else 0.8 * _grav["f"] + 0.2 * a
    g = _grav["f"] / np.linalg.norm(_grav["f"])
    if _grav["ref"] is None:
        _grav["ref"] = g
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(g @ _grav["ref"]))))


def tipping():
    """True once the robot has been tilted past TILT_MAX for 5 steps (~0.3 s) in a row: a real tip, not the
    body rocking on its caster when it brakes or starts."""
    _grav["a"] = tilt()
    _grav["n"] = _grav["n"] + 1 if _grav["a"] > TILT_MAX else 0
    return _grav["n"] >= 5


def mark_hazard_disc(px, py, rad):
    """Mark a disc of floor as an obstacle for both planners."""
    r0, c0 = to_cell(px, py); k = int(math.ceil(rad / RES))
    rr, cc = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
    m = ((rr - r0) ** 2 + (cc - c0) ** 2 <= k * k) & (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
    hazard[rr[m], cc[m]] = True


low_cands = []                       # [x, y, radius, sightings, colour] of low objects not yet trusted
low_boxes = []                       # this frame's low objects for the camera view: (x1, y1, x2, y2, label, colour)
HUE_NAMES = [(8, "red"), (20, "orange"), (33, "yellow"), (85, "green"), (128, "blue"), (170, "purple"), (180, "red")]
LOW_BGR = {"red": (60, 60, 230), "orange": (40, 170, 255), "yellow": (60, 230, 230), "green": (80, 200, 80),
           "blue": (230, 150, 60), "purple": (220, 80, 170)}


def hue_name(h):
    """Plain colour name for an OpenCV hue (0-180), used only to label boxes."""
    return next(name for top, name in HUE_NAMES if h < top)


def update_low_obstacles(frame, det):
    """Spot small, strongly coloured objects lying on the floor with the camera and mark them as obstacles
    before the robot touches them, since the LiDAR scans above them. Any colour counts except the floor's own.
    A blob counts when its bottom sits on the floor, its top is below LOW_MAX_H and it is 3-40 cm wide;
    position comes from where it touches the floor. Each object must be seen 3 times before it is marked.
    Round red blobs are skipped: those may be the apples we're looking for."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_boxes.clear()
    m = ((hsv[..., 1] >= LOW_MIN_SAT) & (hsv[..., 2] >= 50)).astype(np.uint8) * 255
    floor = hsv[CH - 30:, CW // 3:2 * CW // 3].reshape(-1, 3)        # the floor right in front of the robot
    if floor[:, 1].mean() >= LOW_MIN_SAT * 0.7:                        # a strongly coloured floor: ignore its colour
        fh = float(np.median(floor[:, 0]))
        dh = np.abs(hsv[..., 0].astype(np.int16) - fh)
        m[np.minimum(dh, 180 - dh) < 10] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 20:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        bottom = by + bh
        if bottom < CH / 2 + 3 or bottom >= CH - 2 or bx <= 2 or bx + bw >= CW - 2:
            continue                                  # not standing on the floor, or cut off by the image edge
        blob = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(blob, [c - (bx, by)], -1, 255, -1)
        hues = hsv[by:by + bh, bx:bx + bw, 0][blob > 0].astype(np.float32) * (np.pi / 90)
        colour = hue_name(float(np.degrees(np.arctan2(np.sin(hues).mean(), np.cos(hues).mean())) % 360) / 2)
        per = cv2.arcLength(c, True)
        round_ = 4 * math.pi * a / (per * per + 1e-6) >= 0.5 and 0.6 < bw / bh < 1.6
        if colour == "red" and round_:
            continue                                  # round red blobs may be the apples we're looking for
        dist = CAM_HEIGHT / math.tan(math.atan2(bottom - CH / 2, FOCAL)) + CAM_X
        if dist > 2.5:
            continue                                  # far away the floor-contact distance is too rough
        d_cam = dist - CAM_X
        top_h = CAM_HEIGHT - d_cam * (by - CH / 2) / FOCAL
        width = bw * d_cam / FOCAL
        if top_h > LOW_MAX_H or not (0.03 < width < 0.40):
            continue                                  # tall things are on the LiDAR map already
        label = f"{colour} apple" if round_ and 0.06 < width < 0.15 else f"{colour} object"
        low_boxes.append((bx, by, bx + bw, by + bh, f"{label} {dist:.1f}m", colour))
        rad = min(0.15, max(0.05, width / 2))
        b = math.atan2(CW / 2 - (bx + bw / 2), FOCAL)
        ox, oy = x + (dist + rad) * math.cos(th + b), y + (dist + rad) * math.sin(th + b)
        for cand in low_cands:
            if math.hypot(cand[0] - ox, cand[1] - oy) < 0.25:
                cand[0] += 0.3 * (ox - cand[0]); cand[1] += 0.3 * (oy - cand[1])
                cand[2] = max(cand[2], rad); cand[3] += 1
                if cand[3] == 3:
                    print(f"[sar] low obstacle ({colour}) at ({cand[0]:.2f},{cand[1]:.2f}) -> avoiding it")
                if cand[3] >= 3:
                    mark_hazard_disc(cand[0], cand[1], cand[2])
                break
        else:
            low_cands.append([ox, oy, rad, 1, colour])


def draw_low(frame):
    """Boxes for the low floor objects (decoy apples, cans) the colour detector found in this frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x1, y1, x2, y2, label, colour in low_boxes:
        col = LOW_BGR[colour]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        (tw, th_), _ = cv2.getTextSize(label, font, 0.45, 1)
        ty = y1 - 4 if y1 - th_ - 8 >= 0 else y2 + th_ + 6
        cv2.rectangle(frame, (x1, ty - th_ - 4), (x1 + tw + 6, ty + 3), col, -1)
        cv2.putText(frame, label, (x1 + 3, ty), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


BUMP_ACC = 4.0                       # sideways/forward jolt (m/s^2) that means the robot hit something
cmd_hist = []                        # recent (v, w) commands, to tell a collision from our own speed change


def bumped():
    """True when the accelerometer shows a sharp horizontal jolt while the speed command was steady:
    the robot has run into something (possibly closer than the LiDAR's 12 cm minimum range)."""
    a, ref = _grav.get("raw"), _grav.get("ref")
    if a is None or ref is None or len(cmd_hist) < 3 or abs(cmd_hist[-1][0]) < 0.03:
        return False
    if max(abs(c[0] - cmd_hist[-1][0]) for c in cmd_hist) > 0.01 or max(abs(c[1] - cmd_hist[-1][1]) for c in cmd_hist) > 0.05:
        return False                                  # we changed speed ourselves: that jolt is expected
    horiz = a - (a @ ref) * ref
    return float(np.linalg.norm(horiz)) > BUMP_ACC


def wall_ahead_on_map():
    """True if the map has a wall or obstacle right in front of the robot. The LiDAR can't see anything
    closer than 12 cm, so once the robot is almost touching a wall only the map still knows it is there."""
    for d in (0.13, 0.17, 0.21):
        for lat in (-0.09, 0.0, 0.09):
            r, c = to_cell(x + d * math.cos(th) - lat * math.sin(th), y + d * math.sin(th) + lat * math.cos(th))
            if 0 <= r < N and 0 <= c < N and (logodds[r, c] > 0.6 or hazard[r, c]):
                return True
    return False


def mark_hazard(forward):
    """Mark a small patch just ahead of (or behind) the robot as an obstacle for both planners."""
    d = 0.20 if forward else -0.20
    r0, c0 = to_cell(x + d * math.cos(th), y + d * math.sin(th))
    k = int(0.08 / RES)
    hazard[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1] = True


def rear_clearance(ranges, half_deg=40):
    """Closest LiDAR range within +-half_deg of straight behind."""
    k = int(half_deg * NBEAM / 360)
    seg = np.concatenate((ranges[:k + 1], ranges[NBEAM - k:]))   # idx 0 = straight back
    seg = seg[np.isfinite(seg) & (seg > 0.11)]
    return float(seg.min()) if seg.size else LMAX


def follow(path, ranges):
    """Pure pursuit with look-ahead; returns True when the path end is reached."""
    if not path:
        return True
    # nearest waypoint, then look-ahead point
    d = [math.hypot(px - x, py - y) for px, py in path]
    i = int(np.argmin(d))
    del path[:i]
    gx, gy = path[-1]
    if math.hypot(gx - x, gy - y) < GOAL_TOL:
        drive(0, 0); return True
    look = path[-1]
    for p in path:
        if math.hypot(p[0] - x, p[1] - y) > LOOKAHEAD:
            look = p; break
    err = wrap(math.atan2(look[1] - y, look[0] - x) - th)
    if abs(err) > 1.2:
        drive(0, 2.0 * err)          # path is behind us: rotate in place first
        return False
    # look-ahead point in the robot frame -> local planner (DWA on the local costmap)
    dx, dy = look[0] - x, look[1] - y
    lx, ly = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
    cmd = dwa(lx, ly)
    if cmd is None:
        drive(0, 0)                  # every trajectory collides: main loop's blocked logic takes over
        local_blocked[0] = True
    else:
        local_blocked[0] = False
        drive(*cmd)
    return False


def draw_bbox(frame, det):
    """Slide-style BBox: blue box + label card (id / class / confidence) + box formats."""
    b, dist, (bx, by, bw, bh), conf = det
    pad = 4
    x1, y1, x2, y2 = bx - pad, by - pad, bx + bw + pad, by + bh + pad
    blue = (255, 110, 30)
    cv2.rectangle(frame, (x1, y1), (x2, y2), blue, 2)
    lines = [f"id: {len(found_targets) + 1}{count_suffix()}", f"class: {TARGET_NAME}", f"confidence: {conf:.2f}",
             f"dist: {dist:.2f} m  bearing: {math.degrees(b):+.0f} deg"]
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    lw = max(cv2.getTextSize(l, font, fs, 1)[0][0] for l in lines) + 12
    lh = 18 * len(lines) + 8
    cx0 = min(max(0, x1), CW - lw)
    cy0 = y1 - lh - 4 if y1 - lh - 4 >= 0 else y2 + 4
    cv2.rectangle(frame, (cx0, cy0), (cx0 + lw, cy0 + lh), (255, 255, 255), -1)
    cv2.rectangle(frame, (cx0, cy0), (cx0 + lw, cy0 + lh), blue, 2)
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (cx0 + 6, cy0 + 20 + 18 * i), font, fs, blue, 1, cv2.LINE_AA)
    # the four BBox representations from the lecture
    cx, cy = bx + bw / 2, by + bh / 2
    fmts = [f"(x1,y1,x2,y2) = ({bx},{by},{bx + bw},{by + bh})",
            f"(x,y,w,h)     = ({bx},{by},{bw},{bh})",
            f"(cx,cy,w,h)   = ({cx:.0f},{cy:.0f},{bw},{bh})",
            f"norm          = ({cx / CW:.3f},{cy / CH:.3f},{bw / CW:.3f},{bh / CH:.3f})"]
    for i, l in enumerate(fmts):
        yy = CH - 12 - 20 * (len(fmts) - 1 - i)
        cv2.putText(frame, l, (10, yy), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, l, (10, yy), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------- on-screen views: Webots Display overlays in the 3D view ----------------
# The world gives the robot three Display devices ("camera_view", "map", "local_costmap"); their
# overlays are placed in the 3D view (drag / resize them there, Webots remembers it in the .wbproj).
# Worlds without them fall back to OpenCV windows.
UI_BG = (40, 30, 24)                                # BGR: deep navy (unknown space)
UI_FREE = (84, 66, 54)                                # explored floor
UI_SEARCHED = (98, 92, 60)                            # floor the camera has checked (muted teal)
UI_WALL = (236, 230, 224)                             # walls / furniture
UI_NEAR = np.array((120, 84, 64), np.float32)         # local costmap: close to an obstacle
UI_INFL = (130, 80, 150)                              # inflation (no-go for the robot centre)
UI_COST = np.array((110, 90, 150), np.float32)        # global costmap tint (toggle 'c')
UI_GRID = (62, 50, 42)
UI_PATH = (255, 205, 80)                              # cyan
UI_TRAJ = (110, 100, 90)
UI_ROBOT = (90, 200, 255)                             # amber
UI_START = (150, 210, 60)                             # green
UI_APPLE = (80, 80, 235)                              # red
UI_LEAF = (90, 190, 90)
UI_TEXT = (240, 236, 232)
UI_FRAME = (90, 76, 66)
UI_HAZARD = (40, 150, 255)                            # low obstacles (orange)
UI_SCAN = (80, 255, 255)                              # current LiDAR hits (yellow)
UI_BEST = (120, 230, 120)                             # chosen DWA trajectory (green)
SHOW_COSTMAP = [False]                # global costmap tint on the map (toggle: press 'c' in the 3D view)


def _find_display(name):
    for i in range(robot.getNumberOfDevices()):
        d = robot.getDeviceByIndex(i)
        if d.getName() == name and d.getNodeType() == Node.DISPLAY:
            return robot.getDevice(name)
    return None


disp_cam, disp_map, disp_local = (_find_display(n) for n in ("camera_view", "map", "local_costmap"))
keyboard = robot.getKeyboard(); keyboard.enable(dt_ms)


def _badge(img, txt, org=(10, 10), scale=0.45):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th_), _ = cv2.getTextSize(txt, font, scale, 1)
    x0, y0 = org
    over = img.copy()
    cv2.rectangle(over, (x0, y0), (x0 + tw + 16, y0 + th_ + 12), (30, 22, 18), -1)
    cv2.addWeighted(over, 0.75, img, 0.25, 0, img)
    cv2.putText(img, txt, (x0 + 8, y0 + th_ + 5), font, scale, UI_TEXT, 1, cv2.LINE_AA)


def _apple(img, p, r, ghost=False):
    """Little apple icon: red body, white rim, green leaf."""
    if ghost:                                          # seen, not yet confirmed: hollow
        cv2.circle(img, p, r, UI_APPLE, 2, cv2.LINE_AA)
        return
    cv2.circle(img, p, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p, r, UI_APPLE, -1, cv2.LINE_AA)
    cv2.ellipse(img, (p[0] + r // 3, p[1] - r), (max(2, r // 2), max(1, r // 4)), -30, 0, 360, UI_LEAF, -1, cv2.LINE_AA)
    cv2.circle(img, (p[0] - r // 3, p[1] - r // 3), max(1, r // 4), (200, 200, 255), -1, cv2.LINE_AA)   # shine


def _start(img, p, r):
    """Start / home: green ring with a dot."""
    cv2.circle(img, p, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p, r, UI_START, -1, cv2.LINE_AA)
    cv2.circle(img, p, max(2, r // 3), (255, 255, 255), -1, cv2.LINE_AA)


# map palette (BGR)
M_BG_IN, M_BG_OUT = (46, 34, 26), (22, 16, 12)          # vignette: centre / corners
M_DOT = (70, 56, 46)                                    # dot grid over unknown space
M_FLOOR = (86, 70, 58)                                  # explored floor
M_SEARCHED = (122, 112, 58)                             # floor the camera has checked (teal)
M_WALL = (244, 240, 236)
M_PATH = (255, 196, 84)                                 # planned path (sky blue)
M_ROBOT = (64, 186, 255)                                # amber
M_START = (120, 214, 96)                                # green
M_HAZARD = (52, 140, 255)                               # low obstacles (orange)
M_PANEL = (30, 22, 17)                                  # header / legend bars
trail = []                                              # robot positions, for the faded trail on the map
_vignette = {}                                          # cached map background per canvas size


def _blend(img, mask, colour, alpha=1.0):
    """Alpha-blend a solid colour into img where mask (0..1 float) is set."""
    a = (np.clip(mask, 0, 1) * alpha)[..., None]
    img[:] = img * (1 - a) + np.array(colour, np.float32) * a


def _glow(img, centre, radius, colour, strength=0.5):
    """Soft round glow (blurred disc) under an icon."""
    m = np.zeros(img.shape[:2], np.float32)
    cv2.circle(m, centre, radius, 1.0, -1, cv2.LINE_AA)
    _blend(img, cv2.GaussianBlur(m, (0, 0), radius * 0.6), colour, strength)


def _chip(img, org, text, dot=None, scale=0.42):
    """Rounded label chip with an optional coloured dot; returns its right edge."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th_), _ = cv2.getTextSize(text, font, scale, 1)
    x0, y0 = org; pad = 8; dw = 14 if dot else 0
    x1, y1 = x0 + tw + 2 * pad + dw, y0 + th_ + 10
    over = img.copy()
    r = (y1 - y0) // 2
    cv2.rectangle(over, (x0 + r, y0), (x1 - r, y1), (58, 46, 38), -1, cv2.LINE_AA)
    cv2.circle(over, (x0 + r, y0 + r), r, (58, 46, 38), -1, cv2.LINE_AA)
    cv2.circle(over, (x1 - r, y0 + r), r, (58, 46, 38), -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.85, img, 0.15, 0, img)
    if dot:
        cv2.circle(img, (x0 + pad + 4, y0 + r), 4, dot, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad + dw, y1 - 6), font, scale, UI_TEXT, 1, cv2.LINE_AA)
    return x1


def render_map(path, size=480):
    """The overall map: explored and searched floor, walls, path, trail, start, apples and the robot.
    Drawn at twice the size and scaled down for smooth edges; zooms to the explored area."""
    rows, cols = np.nonzero(known)
    if rows.size:
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    else:
        r0 = r1 = c0 = c1 = N // 2
    for px_, py_ in (START[:2], (x, y)):
        r, c = to_cell(px_, py_); r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
    side = max(r1 - r0, c1 - c0, 60) + 24
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0, c0 = max(0, rc - side // 2), max(0, cc - side // 2)
    r0, c0 = min(r0, N - side), min(c0, N - side)
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    kn, lo, seen_, hz = known[sl], logodds[sl], cam_seen[sl], hazard[sl]

    H = size * 2                                        # supersampled canvas
    k = H / side

    # background: vignette + dot grid every metre
    if _vignette.get(H) is None:
        yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
        v = np.clip(np.hypot(xx - H / 2, yy - H / 2) / (H * 0.72), 0, 1)[..., None]
        _vignette[H] = np.array(M_BG_IN, np.float32) * (1 - v) + np.array(M_BG_OUT, np.float32) * v
    img = _vignette[H].copy()
    step = 1.0 / RES * k
    ox = (-((X0 + c0 * RES) % 1.0)) / RES * k
    oy = (((Y0 + (r0 + side) * RES) % 1.0)) / RES * k
    for gx in np.arange(ox, H, step):
        for gy in np.arange(oy - step, H, step):
            cv2.circle(img, (int(gx), int(gy)), 2, M_DOT, -1, cv2.LINE_AA)

    def up(mask, blur):                                 # grid mask -> smooth canvas mask
        m = cv2.resize(cv2.flip(mask.astype(np.float32), 0), (H, H), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), max(0.8, k * blur))

    free = kn & (lo < 0)
    _blend(img, np.clip(up(free, 0.45) * 1.6, 0, 1), M_FLOOR)
    _blend(img, np.clip(up(free & seen_, 0.6) * 1.4, 0, 1), M_SEARCHED, 0.9)
    if SHOW_COSTMAP[0]:
        blocked, _ = cost_maps()
        _blend(img, up(free & blocked[sl], 0.3), UI_INFL, 0.55)

    # walls with a soft drop shadow
    wall = np.clip(up(lo > 0.6, 0.3) * 2.2, 0, 1)
    shadow = np.roll(cv2.GaussianBlur(wall, (0, 0), k * 0.9), (int(k * 0.5), int(k * 0.5)), (0, 1))
    _blend(img, shadow, (8, 6, 4), 0.55)
    _blend(img, wall, M_WALL)

    def px_of(wx, wy):                                  # world -> canvas pixel (+y up)
        r, c = to_cell(wx, wy)
        return int((c - c0 + 0.5) * k), int((side - 1 - (r - r0) + 0.5) * k)

    # low obstacles: soft orange patches
    if hz.any():
        _blend(img, np.clip(up(hz, 0.5) * 2.0, 0, 1), M_HAZARD, 0.9)

    # robot trail, fading out with age
    if len(trail) > 1:
        pts = [px_of(*p) for p in trail]
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = tuple(float(M_BG_IN[j] * (1 - a * 0.7) + M_ROBOT[j] * a * 0.7) for j in range(3))
            cv2.line(img, pts[i - 1], pts[i], col, max(2, int(k * 0.35)), cv2.LINE_AA)

    # planned path: glow + core, ring at the goal
    pts = [px_of(*p) for p in (path or [])]
    if len(pts) > 1:
        g = np.zeros((H, H), np.float32)
        cv2.polylines(g, [np.int32(pts)], False, 1.0, int(k * 1.6), cv2.LINE_AA)
        _blend(img, cv2.GaussianBlur(g, (0, 0), k * 0.8), M_PATH, 0.45)
        cv2.polylines(img, [np.int32(pts)], False, M_PATH, max(3, int(k * 0.4)), cv2.LINE_AA)
        cv2.circle(img, pts[-1], int(k * 1.6), M_PATH, max(2, int(k * 0.25)), cv2.LINE_AA)

    mr = max(10, int(H / 40))
    # start / home: green ring with a house glyph
    sp = px_of(*START[:2])
    _glow(img, sp, int(mr * 1.6), M_START, 0.45)
    cv2.circle(img, sp, mr, M_START, -1, cv2.LINE_AA)
    cv2.circle(img, sp, mr, (255, 255, 255), 2, cv2.LINE_AA)
    hw = mr * 0.5
    roof = np.int32([(sp[0] - hw * 1.2, sp[1]), (sp[0], sp[1] - hw * 1.1), (sp[0] + hw * 1.2, sp[1])])
    cv2.fillPoly(img, [roof], (255, 255, 255), cv2.LINE_AA)
    cv2.rectangle(img, (int(sp[0] - hw * 0.75), int(sp[1])), (int(sp[0] + hw * 0.75), int(sp[1] + hw * 0.9)), (255, 255, 255), -1)

    # apples: glow, icon and a check badge once rescued
    for fx, fy in found_targets:
        p = px_of(fx, fy)
        _glow(img, p, int(mr * 1.7), UI_APPLE, 0.5)
        _apple(img, p, mr)
        b = (p[0] + int(mr * 0.9), p[1] + int(mr * 0.9))
        cv2.circle(img, b, int(mr * 0.55), M_START, -1, cv2.LINE_AA)
        cv2.polylines(img, [np.int32([(b[0] - mr * 0.3, b[1]), (b[0] - mr * 0.05, b[1] + mr * 0.25),
                                      (b[0] + mr * 0.32, b[1] - mr * 0.25)])], False, (255, 255, 255), 3, cv2.LINE_AA)
    if target_xy:
        _apple(img, px_of(*target_xy), mr, ghost=True)

    # robot: amber halo + arrow
    p = np.array(px_of(x, y), np.float32)
    _glow(img, (int(p[0]), int(p[1])), int(mr * 2.0), M_ROBOT, 0.45)
    fwd = np.array((math.cos(th), -math.sin(th)), np.float32)
    left = np.array((-fwd[1], fwd[0]), np.float32)
    s_ = mr * 1.5
    tri = np.int32([p + fwd * s_, p - fwd * s_ * 0.6 + left * s_ * 0.75, p - fwd * s_ * 0.2, p - fwd * s_ * 0.6 - left * s_ * 0.75])
    cv2.fillPoly(img, [tri], M_ROBOT, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (255, 255, 255), 3, cv2.LINE_AA)

    view = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)

    # header: title + status chips; footer: legend
    over = view.copy()
    cv2.rectangle(over, (0, 0), (size, 34), M_PANEL, -1)
    cv2.rectangle(over, (0, size - 26), (size, size), M_PANEL, -1)
    cv2.addWeighted(over, 0.78, view, 0.22, 0, view)
    cv2.putText(view, "MAP", (12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.55, UI_TEXT, 1, cv2.LINE_AA)
    xe = _chip(view, (60, 7), f"apples {len(found_targets)}{count_suffix()}", UI_APPLE)
    xe = _chip(view, (xe + 6, 7), state.lower(), M_ROBOT)
    _chip(view, (xe + 6, 7), f"{robot.getTime():.0f} s")
    lx = 10
    for name, col in (("searched", M_SEARCHED), ("path", M_PATH), ("obstacle", M_HAZARD), ("start", M_START)):
        cv2.circle(view, (lx + 5, size - 13), 5, col, -1, cv2.LINE_AA)
        cv2.putText(view, name, (lx + 15, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_TEXT, 1, cv2.LINE_AA)
        lx += 30 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]

    # rounded frame
    mask = np.zeros((size, size), np.uint8)
    cv2.rectangle(mask, (14, 0), (size - 15, size - 1), 255, -1)
    cv2.rectangle(mask, (0, 14), (size - 1, size - 15), 255, -1)
    for cx_, cy_ in ((14, 14), (size - 15, 14), (14, size - 15), (size - 15, size - 15)):
        cv2.circle(mask, (cx_, cy_), 14, 255, -1, cv2.LINE_AA)
    view[mask == 0] = M_BG_OUT
    return view


def _paste(disp, bgr):
    h, w = disp.getHeight(), disp.getWidth()
    if bgr.shape[:2] != (h, w):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    ref = disp.imageNew(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)).tobytes(), Display.BGRA, w, h)
    disp.imagePaste(ref, 0, 0, False)
    disp.imageDelete(ref)


def show(frame, path, det):
    key = keyboard.getKey()
    while key != -1:
        if key in (ord("C"), ord("c")):
            SHOW_COSTMAP[0] = not SHOW_COSTMAP[0]
        key = keyboard.getKey()
    if not SHOW_DEBUG:
        return
    if SHOW_ALL_OBJECTS:
        for why, (rx, ry, rw, rh), _ in rejects:   # rejected red blobs: thin grey boxes
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (160, 160, 160), 1)
    taken = [b[:4] for b in low_boxes]
    if det:
        bx_, by_, bw_, bh_ = det[2]
        taken.append((bx_, by_, bx_ + bw_, by_ + bh_))
    if yolo is not None:
        draw_yolo(frame, taken)
    draw_low(frame)
    if det:
        draw_bbox(frame, det)
    status = f"{state}  x={x:.2f} y={y:.2f}"
    (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (CW - sw - 16, 4), (CW - 4, sh + 14), (0, 0, 0), -1)
    cv2.putText(frame, status, (CW - sw - 10, sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    step_i = int(robot.getTime() / DT)
    if SAVE_DEBUG_FILES and yolo_boxes and step_i % 50 == 0:   # snapshot for slides/debugging
        cv2.imwrite(os.path.join(HERE, "cam_live.jpg"), frame)
    m = render_map(path, disp_map.getWidth() if disp_map else 600) if step_i % 2 == 0 or not disp_map else None
    lv = costmap_view(path, disp_local.getWidth() if disp_local else 300)
    if disp_cam:
        _paste(disp_cam, frame)
    else:
        cv2.imshow("camera", frame)
    if disp_local:
        _paste(disp_local, lv)
    else:
        cv2.imshow("costmap", lv)
    if m is not None:
        if disp_map:
            _paste(disp_map, m)
        else:
            cv2.imshow("map", m)
    if SAVE_DEBUG_FILES and step_i % 50 == 0:
        if m is not None:
            cv2.imwrite(os.path.join(HERE, "map_live.jpg"), m)
        import json   # object map dump (for checking label accuracy against the real scene)
        with open(os.path.join(HERE, "sem_objects.json"), "w") as fh:
            json.dump([{**o, "label": semantic.SemanticMap.label(o)[0], "share": semantic.SemanticMap.label(o)[1],
                        "reliable": o["n"] >= semantic.MIN_SEEN and semantic.SemanticMap.label(o)[1] >= semantic.MIN_SHARE}
                       for o in sem.objects], fh)
    if not (disp_cam and disp_map and disp_local):
        if (cv2.waitKey(1) & 0xFF) == ord("c"):
            SHOW_COSTMAP[0] = not SHOW_COSTMAP[0]


# ---------------- mission helpers ----------------
def count_suffix():
    return f"/{TARGET_COUNT}" if TARGET_COUNT else ""


def at_goal(goal, ranges):
    """Destination / home reached: within GOAL_TOL, or within NEAR_GOAL when a wall or obstacle stops the
    robot getting any closer (e.g. the start pose right next to a wall)."""
    d = math.hypot(goal[0] - x, goal[1] - y)
    if d < GOAL_TOL:
        return True
    if d >= NEAR_GOAL:
        return False
    if front_clearance(ranges) < SAFE_FRONT + 0.05 or local_blocked[0] or blocked_since is not None:
        return True
    blocked, _ = cost_maps()
    return bool(blocked[to_cell(*goal)])


def start_backoff(t, dur, reason):
    """Back off for `dur` s, unless that keeps happening (3 times in 15 s): then turn toward open space
    and give up the current goal instead, so the robot can't reverse itself into a corner."""
    global backoff_until
    recent_backoffs[:] = [b for b in recent_backoffs if t - b < 15.0] + [t]
    if len(recent_backoffs) < 3:
        backoff_until = t + dur
        return
    recent_backoffs.clear()
    start_escape(t, f"{reason}: 3 back-offs in 15 s")


def start_escape(t, why):
    """Turn toward open space for up to 3 s and drop the current exploration goal."""
    global backoff_until, escape_until, goal, path
    print(f"[sar] {why} -> turning toward open space and dropping the current goal")
    backoff_until = 0.0; escape_until = t + 3.0; path = None
    if state == "EXPLORE" and goal is not None:
        banned.append(goal); goal = None


def open_direction(ranges):
    """Robot-frame angle of the most open direction (LiDAR ranges smoothed over ~30 deg)."""
    r = np.where(np.isfinite(ranges), np.minimum(ranges, LMAX), LMAX)
    k = max(1, NBEAM // 12)
    sm = np.convolve(np.concatenate((r[-k:], r, r[:k])), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    return float(BEAM_ANG[int(np.argmax(sm))])


def dup_tol(d):
    """How close (m) a sighting at distance d must be to a rescued apple to count as that same apple.
    Far-away distance estimates are poor, so the tolerance grows with distance."""
    return max(1.5, 0.6 + 0.35 * d)


def after_search():
    """State after the search: visit the destination if one was given, otherwise go home."""
    return "TO_DEST" if DEST else "HOME"


# ---------------- main loop ----------------
state = "SCAN"
path = None
goal = None
target_xy = None
banned = []
last_plan = -1e9
last_tilt = -1e9
escape_until = 0.0                    # turning toward open space until then
recent_backoffs = []                  # times of recent back-offs
blocked_since = None
backoff_until = 0.0
scan_turned = 0.0
seen_count = 0
wd_t, wd_xy = 0.0, (x, y)
approach_fails = 0
false_targets = []                    # (x, y, t) of false alarms / unreachable targets, ignored for a while
found_targets = []                    # apples already reached
close_obs = []                        # close-range position estimates of the current target
verified = 0                          # close-range confirmations of the current target
arrived = False
look_spots = [(x, y)]                 # where 360 deg look-arounds were done
no_frontier_rounds = 0                # times in a row nothing was left to search

print(f"[sar] mission: find {f'{TARGET_COUNT}x' if TARGET_COUNT else 'every'} {TARGET_NAME}"
      f"{f' -> destination {DEST}' if DEST else ''} -> home {START[:2]}")

while robot.step(dt_ms) != -1:
    t = robot.getTime()
    near = arrived; arrived = False
    if not trail or math.hypot(x - trail[-1][0], y - trail[-1][1]) > 0.1:
        trail.append((x, y))                          # for the faded trail on the map
    th_before = th
    update_odometry()
    if tipping() and last_cmd[0] > 0.03 and t >= backoff_until and t >= escape_until and t - last_tilt > 2.0 \
            and state not in ("SCAN", "DONE"):
        # tipping while driving forward: something too low for the LiDAR (carpet edge, step) is under the wheels
        print(f"[sar] tilt {math.degrees(_grav['a']):.1f} deg at ({x:.2f},{y:.2f}) -> low obstacle ahead, "
              f"marking it and backing off until level")
        last_tilt = t
        mark_hazard(True)
        path = None; last_plan = -1e9
        if state == "EXPLORE":
            goal = None
        drive(0, 0)
        start_backoff(t, 3.0, "tilt")                  # ends early once the robot is level again
    cmd_hist.append(tuple(last_cmd)); del cmd_hist[:-3]
    if bumped() and t >= backoff_until and t >= escape_until and state not in ("SCAN", "DONE"):
        forward = last_cmd[0] > 0
        print(f"[sar] bump at ({x:.2f},{y:.2f}) -> marking it, backing off and re-planning")
        mark_hazard(forward)
        path = None; last_plan = -1e9
        drive(0, 0)
        if forward:
            start_backoff(t, 1.0, "bump")
    ranges = np.array(lidar.getRangeImage(), dtype=np.float32)
    if int(t / DT) % MATCH_EVERY == 0 and t > 8.0:
        scan_match(ranges)
    update_map(ranges)
    update_cam_coverage(ranges)
    update_local_costmap(ranges)
    frame = cv2.cvtColor(np.frombuffer(camera.getImage(), np.uint8).reshape(CH, CW, 4), cv2.COLOR_BGRA2BGR)
    if yolo is not None and int(t / DT) % YOLO_EVERY == 0:
        yolo_boxes = run_yolo(frame)
        if SEMANTIC and state in ("SCAN", "EXPLORE"):
            sem.add(yolo_boxes, ranges, (x, y, th), CW, FOCAL, None, NBEAM)
    det = detect_target(frame)
    if LOW_OBJECTS and int(t / DT) % 2 == 0:
        update_low_obstacles(frame, det)
    if rejects and t - _last_rej[0] > 3 and max(r[2] for r in rejects) > 80:
        _last_rej[0] = t
        print(f"[rej] t={t:.1f} {rejects[:3]}")
        if SAVE_DEBUG_FILES:
            cv2.imwrite(os.path.join(HERE, f"rej_{int(t)}.jpg"), frame)

    # ---- perception -> target estimate (needs a few consistent detections) ----
    if det and state in ("SCAN", "EXPLORE", "APPROACH") and det[1] < 6.0:
        b, dist = det[0], det[1]
        tx, ty = x + dist * math.cos(th + b), y + dist * math.sin(th + b)
        if any(math.hypot(tx - fx, ty - fy) < 1.0 and t - ft < FALSE_FORGET for fx, fy, ft in false_targets) or \
                any(math.hypot(tx - fx, ty - fy) < dup_tol(dist) for fx, fy in found_targets):   # already rescued
            det = None
    if det and state in ("SCAN", "EXPLORE", "APPROACH") and det[1] < 6.0:
        if target_xy is None or math.hypot(tx - target_xy[0], ty - target_xy[1]) > 1.0:
            if state != "APPROACH":
                target_xy = (tx, ty); seen_count = 1; verified = 0; close_obs = []
        else:
            a = 0.3
            target_xy = ((1 - a) * target_xy[0] + a * tx, (1 - a) * target_xy[1] + a * ty)
            seen_count += 1
            # close-up confirmation: far away a few red pixels look round (e.g. a red can on its side)
            _, _, (_, _, bw_, bh_), conf_ = det
            if det[1] < VERIFY_DIST and 0.8 < bw_ / max(bh_, 1) < 1.25 and conf_ >= 0.6:
                verified += 1
                close_obs.append((tx, ty))           # close-range sightings give the most accurate position
        if seen_count >= 3 and state != "APPROACH":
            print(f"[sar] {TARGET_NAME} spotted at ({target_xy[0]:.2f}, {target_xy[1]:.2f})")
            state = "APPROACH"; path = None; last_plan = -1e9
            if SAVE_DEBUG_FILES:
                cv2.imwrite(os.path.join(HERE, "spotted.jpg"), frame)

    if state == "DONE":
        drive(0, 0); show(frame, path, det); continue

    # ---- safety layer / recovery ----
    if t < escape_until:
        err = wrap(open_direction(ranges))
        if abs(err) < 0.25:
            escape_until = 0.0                    # facing open space: let the planner take over again
        else:
            drive(0, 1.5 if err > 0 else -1.5)
            show(frame, path, det); continue
    if t < backoff_until and t - last_tilt > 1.0 and _grav.get("a", 0.0) < TILT_MAX / 2 and \
            last_tilt > backoff_until - 3.5:
        backoff_until = t                         # tilt back-off: level again, off the edge -> stop reversing
    if t < backoff_until:
        # never reverse blind: back off only while the LiDAR shows nothing close behind,
        # and stop if reversing tips the robot onto something behind it
        if _grav.get("a", 0.0) > TILT_MAX and t - last_tilt > 3.5:
            backoff_until = 0.0
            drive(0, 0)
        elif rear_clearance(ranges) > SAFE_REAR:
            drive(-0.08, 0.0)
        else:
            start_escape(t, "no room to back off")
            drive(0, 0)
        show(frame, path, det); continue

    if state == "SCAN":
        scan_turned += abs(wrap(th - th_before))
        drive(0, 1.6)
        if scan_turned > 2 * math.pi:
            state = "EXPLORE"; path = None
        show(frame, path, det); continue

    # choose goal for the current state
    if state == "APPROACH" and any(math.hypot(target_xy[0] - fx, target_xy[1] - fy) < 1.8 for fx, fy in found_targets):
        # far-away distance estimates are poor: the "new" target turned out to be one we already reached
        print(f"[sar] target ({target_xy[0]:.2f},{target_xy[1]:.2f}) is an already-found {TARGET_NAME} -> keep exploring")
        target_xy = None; seen_count = 0; verified = 0
        state = "EXPLORE"; goal = None; path = None; last_plan = -1e9; drive(0, 0); continue
    if state == "APPROACH":
        goal = target_xy
        if (near or math.hypot(goal[0] - x, goal[1] - y) < REACH_DIST) and verified < 2:
            print(f"[sar] {TARGET_NAME} at ({goal[0]:.2f},{goal[1]:.2f}) NOT confirmed up close -> false alarm, keep exploring")
            false_targets.append((*target_xy, t)); target_xy = None; seen_count = 0; verified = 0
            state = "EXPLORE"; goal = None; path = None; last_plan = -1e9; drive(0, 0); continue
        if near or math.hypot(goal[0] - x, goal[1] - y) < REACH_DIST:
            print(f"[sar] {TARGET_NAME} confirmed up close ({verified} close views)")
            pos = tuple(np.median(np.array(close_obs), axis=0)) if close_obs else target_xy
            found_targets.append((float(pos[0]), float(pos[1])))
            sem.forget_near(pos[0], pos[1], 1.5)       # a rescued apple must not pull the search back to it
            mark_hazard_disc(pos[0], pos[1], 0.06)     # and must not be driven over later
            target_xy = None
            n = len(found_targets)
            if TARGET_COUNT and n >= TARGET_COUNT:
                print(f"[sar] reached {TARGET_NAME} #{n} at t={t:.1f}s -> all {TARGET_COUNT} found, search over")
                state = after_search()
            else:
                print(f"[sar] reached {TARGET_NAME} #{n} at t={t:.1f}s -> searching for the next one")
                state = "EXPLORE"; goal = None; target_xy = None; seen_count = 0; approach_fails = 0
            path = None; last_plan = -1e9; drive(0, 0); continue
    elif state == "TO_DEST":
        goal = DEST
        if near or at_goal(goal, ranges):
            print(f"[sar] destination reached at t={t:.1f}s -> returning home")
            state = "HOME"; path = None; last_plan = -1e9; backoff_until = 0.0; drive(0, 0); continue
    elif state == "HOME":
        goal = START[:2]
        if near or at_goal(goal, ranges):
            print(f"[sar] MISSION COMPLETE at t={t:.1f}s")
            state = "DONE"; backoff_until = 0.0; drive(0, 0); continue

    # (re)plan periodically so the map/pedestrian changes are taken into account
    if path is None or t - last_plan > 1.5:
        blocked, penalty = cost_maps()
        if state == "EXPLORE":
            if goal is not None and math.hypot(goal[0] - x, goal[1] - y) < 0.4:
                mark_seen_around(*goal, rad=0.15)   # arrived: don't pick this exact spot again
            if goal is None or path is None or cam_seen[to_cell(*goal)] or math.hypot(goal[0] - x, goal[1] - y) < 0.4 or t - last_plan > 6:
                goal = nearest_frontier(blocked, banned)
                if goal is None:
                    no_frontier_rounds += 1
                    if no_frontier_rounds >= 2:
                        # nothing reachable is left unseen, even after retrying skipped goals: search is over
                        print(f"[sar] search complete at t={t:.1f}s: {len(found_targets)} {TARGET_NAME}(s) found")
                        state = after_search(); goal = None; path = None; last_plan = -1e9; drive(0, 0); continue
                    print("[sar] no frontier left; clearing banned list and spinning")
                    banned.clear(); state = "SCAN"; scan_turned = 0.0; continue
        new = astar(goal, blocked, penalty)
        last_plan = t
        if new is None:
            if state == "EXPLORE":
                banned.append(goal); goal = None
            path = []
        else:
            path = new

    # progress watchdog: no real motion for 3.5 s -> back off, forget this goal, replan
    if t - wd_t > 3.5:
        if math.hypot(x - wd_xy[0], y - wd_xy[1]) < 0.08 and path:
            print(f"[sar] stuck at ({x:.2f},{y:.2f}) -> recovery")
            start_backoff(t, 1.0, "stuck"); path = None
            if state == "EXPLORE" and goal:
                banned.append(goal); goal = None       # skipped for now, retried before the search ends
            if state == "APPROACH":
                approach_fails += 1
                if approach_fails >= 3:
                    print("[sar] target unreachable/false -> back to EXPLORE")
                    false_targets.append((*target_xy, t)); target_xy = None
                    state = "EXPLORE"; goal = None; approach_fails = 0
        wd_t, wd_xy = t, (x, y)

    clear = front_clearance(ranges)
    if clear < SAFE_FRONT or local_blocked[0] or wall_ahead_on_map():
        # something right in front / DWA found no safe trajectory (possibly the pedestrian): stop, re-plan
        # around it at once, and back off if that doesn't help within 2.5 s
        local_blocked[0] = False          # re-run DWA next step
        if blocked_since is None:
            blocked_since = t
            last_plan = -1e9
        if t - blocked_since > 2.5:
            start_backoff(t, 0.8, "blocked"); blocked_since = None; path = None
            if state == "EXPLORE" and goal:
                banned.append(goal); goal = None
        # still allowed to rotate toward the path so we can turn away from walls
        if path:
            look = path[min(len(path) - 1, 6)]
            err = wrap(math.atan2(look[1] - y, look[0] - x) - th)
            if abs(err) > 0.5:
                drive(0, 1.5 * err)
            else:
                # path points into the obstacle: turn toward the more open side instead of freezing
                left = np.nanmin(np.where(np.isfinite(ranges[60:150]), ranges[60:150], LMAX))
                right = np.nanmin(np.where(np.isfinite(ranges[210:300]), ranges[210:300], LMAX))
                drive(0, 1.2 if left > right else -1.2)
        else:
            drive(0, 0)
    else:
        done = follow(path, ranges)
        if not local_blocked[0]:
            blocked_since = None
        if done:
            if state == "EXPLORE":
                if goal is not None:
                    no_frontier_rounds = 0        # reached a frontier: the search is still making progress
                goal = None; path = None
                # new area reached: 360 deg camera look-around (camera sees only 60 deg ahead,
                # objects tucked beside furniture are otherwise missed)
                if all(math.hypot(x - lx, y - ly) > LOOK_SPACING for lx, ly in look_spots):
                    look_spots.append((x, y)); state = "SCAN"; scan_turned = 0.0
            elif path and goal and math.hypot(goal[0] - x, goal[1] - y) < 0.6:
                # goal sits inside obstacle inflation (e.g. start pose next to a wall): closest safe spot is good enough
                path = None; arrived = True

    if int(t / DT) % 30 == 0:
        print(f"[dbg] t={t:.1f} {state} scanmatch={_match['n']}x/{_match['total']:.2f}m pose=({x:.2f},{y:.2f},{th:.2f}) goal={goal} clear={front_clearance(ranges):.2f} tilt={math.degrees(_grav.get('a', 0.0)):.1f} det={det[:2] if det else None}")
    show(frame, path, det)
