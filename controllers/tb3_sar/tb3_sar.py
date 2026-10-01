"""TB3 Autonomous Search & Rescue (sensors only: encoders + compass + LiDAR + camera).

Mission (FSM):
  SCAN      spin 360 deg at start to build the first map and look around
  EXPLORE   frontier exploration (BFS to nearest unknown border, A* + pure pursuit)
  APPROACH  target (red apple) seen -> plan to it until within REACH_DIST
  TO_DEST   deliver: plan to the destination (safe zone)
  HOME      plan back to the start pose
  DONE

Safety layer runs every step: stop if anything is too close in the driving direction
(walls, furniture, the moving pedestrian); wait, then back off and replan.
"""
import heapq
import math
import os
from collections import deque

import cv2
import numpy as np
from controller import Display, Node, Robot

import semantic

# ---------------- mission config (world frame, metres) ----------------
START = (-0.3, -7.5, math.pi)       # known start pose of the robot
DEST = (-4.94, -7.33)               # destination / "safe zone"
TARGET_NAME = "red apple"           # "red apple" (colour detection) or "football" (YOLO COCO "sports ball")
TARGETS = {
    "red apple": dict(kind="color", diameter=0.10, count=2),
    "football": dict(kind="yolo", yolo_class="soccer ball", diameter=0.22, min_conf=0.25, count=1,
                     max_colorful=0.25),            # football is black/white: reject strongly coloured boxes
}
TARGET = TARGETS[TARGET_NAME]
TARGET_COUNT = TARGET["count"]      # how many targets must be found before delivering
LOOK_SPACING = 1.5                  # do a 360 deg camera look-around every time we reach a new area this far away
VERIFY_DIST = 2.0                   # a target only counts if it was confirmed (round, right size) closer than this
REACH_DIST = 0.35                   # target reached when this close
GOAL_TOL = 0.25                     # dest/home reached when this close
SHOW_DEBUG = True                   # camera / map / local costmap views
SAVE_DEBUG_FILES = False            # True: also save debug snapshots here (rej_*.jpg, spotted.jpg, *_live.jpg, sem_objects.json)
SHOW_ALL_OBJECTS = False            # False: camera shows boxes on the target only (YOLO still feeds the semantic map)
YOLO_EVERY = 1 if TARGET["kind"] == "yolo" else 5  # YOLO-based targets need detections every frame
YOLO_WEIGHTS = "yolo_world_apartment.pt"  # open-vocabulary YOLO-World (make_world_model.py); fallback yolo11n.pt
SEMANTIC = True                     # semantic frontier exploration (YOLO objects bias where to search)
YOLO_CONF = 0.2                     # sim renders score low; lecture used 0.1

# ---------------- robot constants (from lecture notebook) ----------------
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
ROBOT_RADIUS = 0.105
MAX_WHEEL = 6.67
V_MAX = 0.22                         # m/s (TB3 Burger max)
W_MAX = 2.5                          # rad/s
CAM_HEIGHT = 0.073                   # camera height above floor
APPLE_D = TARGET["diameter"]         # target diameter (m), used for size-vs-distance checks
CAM_X = 0.02                        # camera forward offset from base

# ---------------- map ----------------
RES = 0.05
X0, Y0 = -15.0, -15.0
N = int(18.0 / RES)                  # covers [-15, 3] x [-15, 3]
INFLATE = ROBOT_RADIUS + 0.09        # obstacle inflation radius
SAFE_FRONT = 0.20                    # emergency stop distance (from lidar centre)

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
    _print(*a)
    _LOG.write(" ".join(map(str, a)) + "\n"); _LOG.flush()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------- state ----------------
x, y, th = START
logodds = np.zeros((N, N), np.float32)
known = np.zeros((N, N), bool)
cam_seen = np.zeros((N, N), bool)     # floor cells the camera has actually looked at
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
        # auto-calibrate compass direction against encoders during the first spin
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
    r0, c0 = to_cell(px, py); k = int(rad / RES)
    cam_seen[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1] = True


def cost_maps():
    occ = (logodds > 0.6).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * RES
    blocked = dist < INFLATE
    penalty = np.clip(0.45 - dist, 0, None) * 8.0
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
            ng = gs[c] + w * (1 + penalty[n]) + (0.3 if not known[n] else 0)
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
    free = known & (logodds < -0.5)
    unk = (~known).astype(np.uint8)
    front = free & (cv2.dilate(unk, np.ones((3, 3), np.uint8)) > 0)
    # search goal = LiDAR frontier OR free floor the camera has not looked at yet (specks removed)
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
    jev.request(cands, sem, TARGET_NAME)
    r = jev.take()
    if r:
        print(f"[sar] Jev picked frontier ({r[0]:.2f},{r[1]:.2f}) conf={r[2]:.2f}")
        return r[:2]
    g = semantic.pick(cands, sem)
    if g != cands[0][:2]:
        print(f"[sar] semantic: chose ({g[0]:.1f},{g[1]:.1f}) over nearest ({cands[0][0]:.1f},{cands[0][1]:.1f})"
              f" likelihood={sem.likelihood(*g):.2f}")
    return g

# ---------------- YOLO (display only: labels every object the camera sees) ----------------
yolo = None
yolo_boxes = []                      # cached [(x1, y1, x2, y2, name, conf)]
if SHOW_DEBUG or SEMANTIC:
    try:
        import torch
        from ultralytics import YOLO
        YOLO_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
        _wdir = os.path.join(HERE, "../../models/YOLO")
        if not os.path.exists(os.path.join(_wdir, YOLO_WEIGHTS)):
            print(f"[sar] {YOLO_WEIGHTS} missing (run models/YOLO/make_world_model.py) -> using yolo11n.pt (COCO)")
            YOLO_WEIGHTS = "yolo11n.pt"
            TARGETS["football"]["yolo_class"] = "sports ball"
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


def draw_yolo(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for id_, (x1, y1, x2, y2, name, c) in enumerate(yolo_boxes, 1):
        col = yolo_color(name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"id:{id_} {name} {c:.2f}"
        (tw, th_), _ = cv2.getTextSize(label, font, 0.45, 1)
        ty = y1 - 4 if y1 - th_ - 8 >= 0 else y1 + th_ + 6
        cv2.rectangle(frame, (x1, ty - th_ - 4), (x1 + tw + 6, ty + 3), col, -1)
        cv2.putText(frame, label, (x1 + 3, ty), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


sem = semantic.SemanticMap(semantic.PRIORS.get(TARGET_NAME))
jev = semantic.JevChooser()
if SEMANTIC:
    print(f"[sar] semantic exploration ON (chooser: {'Jev' if jev.enabled else 'local prior'})")

rejects = []
_last_rej = [-1e9]


def detect_yolo_target(frame):
    """Target from YOLO boxes (e.g. football = COCO 'sports ball'), with the same floor/size checks."""
    best = None
    rejects.clear()
    for x1, y1, x2, y2, name, conf in yolo_boxes:
        if name != TARGET["yolo_class"]:
            continue
        bw, bh = x2 - x1, y2 - y1
        why = None
        if conf < TARGET["min_conf"]:
            why = f"conf {conf:.2f}"
        elif y2 < CH / 2 + 3:
            why = "above floor"
        elif x1 <= 2 or x2 >= CW - 2:
            why = "edge"
        else:
            # two independent distance estimates must roughly agree: from box size and from floor contact
            d_size = FOCAL * APPLE_D / max(bw, bh)
            d_floor = CAM_HEIGHT / math.tan(math.atan2(y2 - CH / 2, FOCAL)) + CAM_X
            roi = cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2HSV)
            sat = float((roi[..., 1] > 120).mean()) if roi.size else 0.0
            if not (0.35 < d_size / d_floor < 3.0):
                why = f"dist size {d_size:.1f} vs floor {d_floor:.1f}"
            elif sat > TARGET.get("max_colorful", 1.0):
                why = f"too colourful ({sat:.2f})"               # a red apple labelled "sports ball"
        if why:
            rejects.append((why, (x1, y1, bw, bh), bw * bh))
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, y1, bw, bh)
    if best is None:
        return None
    conf, bx, by, bw, bh = best
    bearing = math.atan2(CW / 2 - (bx + bw / 2), FOCAL)
    dist = FOCAL * APPLE_D / max(bw, bh) + CAM_X + APPLE_D / 2    # size-based: robust for big objects
    return bearing, dist, (bx, by, bw, bh), conf


def detect_target(frame):
    """Red apple on the floor -> (bearing, distance, bbox) or None."""
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
    dist = CAM_HEIGHT / math.tan(depress) + CAM_X + 0.03
    return bearing, dist, (bx, by, bw, bh), conf


def drive(v, w):
    v = max(-V_MAX, min(V_MAX, v)); w = max(-W_MAX, min(W_MAX, w))
    wl = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    wr = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    lim = 0.99 * MAX_WHEEL           # stay just under the motor limit (float rounding triggers Webots warnings)
    k = max(1.0, abs(wl) / lim, abs(wr) / lim)
    lm.setVelocity(max(-lim, min(lim, wl / k))); rm.setVelocity(max(-lim, min(lim, wr / k)))


# ---------------- local costmap (rolling window, robot frame) + DWA local planner ----------------
LOCAL_SIZE = 3.0                     # 3 m x 3 m window centred on the robot
LN = int(LOCAL_SIZE / RES)
LOOKAHEAD = 0.6
DWA_T, DWA_DT = 1.5, 0.1             # simulate each candidate command 1.5 s ahead
DWA_V = np.linspace(0.0, V_MAX, 6)
DWA_W = np.linspace(-W_MAX, W_MAX, 15)
COLLIDE = ROBOT_RADIUS + 0.03
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
    """Rebuild the local costmap from the CURRENT scan only (moving people leave no ghosts)."""
    local_occ[:] = 0
    ok = np.isfinite(ranges) & (ranges > 0.12) & (ranges < LOCAL_SIZE)
    px = ranges[ok] * np.cos(BEAM_ANG[ok]); py = ranges[ok] * np.sin(BEAM_ANG[ok])
    c = ((px + LOCAL_SIZE / 2) / RES).astype(int); r = ((py + LOCAL_SIZE / 2) / RES).astype(int)
    m = (r >= 0) & (r < LN) & (c >= 0) & (c < LN)
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
    cost = 1.0 * goal_cost + 0.8 * (1 - np.minimum(clear, 0.5) / 0.5) - 0.25 * _V / V_MAX
    cost[~ok] = np.inf
    dwa_viz["cands"], dwa_viz["ok"] = TRAJ, ok
    if not ok.any():
        dwa_viz["best"] = None
        return None
    i = int(np.argmin(cost))
    dwa_viz["best"] = i
    return float(_V[i]), float(_W[i])


def local_view(size=300):
    """Local costmap (robot frame, forward = right): obstacles, inflation, DWA candidates + chosen one."""
    heat = np.clip(local_dist / 0.8, 0, 1)[..., None]
    img = (np.float32(UI_BG) * heat + UI_NEAR * (1 - heat)).astype(np.uint8)        # darker = farther from obstacles
    img[local_dist < COLLIDE] = UI_INFL                                   # lethal for the robot centre
    img[local_occ > 0] = UI_WALL
    k = size / LN
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    if dwa_viz["cands"] is not None:
        for i, tr in enumerate(dwa_viz["cands"][::3]):
            if dwa_viz["ok"][i * 3]:
                pts = (((tr + LOCAL_SIZE / 2) / RES) * k).astype(np.int32)
                cv2.polylines(img, [pts], False, UI_TRAJ, 1, cv2.LINE_AA)
        if dwa_viz["best"] is not None:
            pts = (((dwa_viz["cands"][dwa_viz["best"]] + LOCAL_SIZE / 2) / RES) * k).astype(np.int32)
            cv2.polylines(img, [pts], False, UI_PATH, 3, cv2.LINE_AA)
    img = cv2.flip(img, 0)                              # robot frame: forward = right, left = up
    cc = size // 2
    rr = max(4, int(ROBOT_RADIUS / RES * k))
    cv2.circle(img, (cc, cc), rr, UI_ROBOT, -1, cv2.LINE_AA)
    cv2.line(img, (cc, cc), (cc + rr, cc), UI_BG, 2, cv2.LINE_AA)
    _badge(img, "local costmap  3 m")
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), UI_FRAME, 2)
    return img


def front_clearance(ranges, half_deg=35):
    i0 = NBEAM // 2
    k = int(half_deg * NBEAM / 360)
    seg = ranges[i0 - k:i0 + k + 1]
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
    lines = [f"id: {len(found_targets) + 1}/{TARGET_COUNT}", f"class: {TARGET_NAME}", f"confidence: {conf:.2f}",
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


def render_map(path, size=480):
    """Clean global map: walls, explored floor, robot, path, start and found apples. Zooms to the explored area."""
    rows, cols = np.nonzero(known)
    if rows.size:
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    else:
        r0 = r1 = c0 = c1 = N // 2
    for px, py in (START[:2], (x, y)):
        r, c = to_cell(px, py); r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
    side = max(r1 - r0, c1 - c0, 60) + 16
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0, c0 = max(0, rc - side // 2), max(0, cc - side // 2)
    r0, c0 = min(r0, N - side), min(c0, N - side)
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    kn, lo, seen_ = known[sl], logodds[sl], cam_seen[sl]

    cell = np.empty((side, side, 3), np.float32)
    cell[:] = UI_BG
    free = kn & (lo < 0)
    cell[free] = UI_FREE
    cell[free & seen_] = UI_SEARCHED
    if SHOW_COSTMAP[0]:
        blocked, _ = cost_maps()
        b = free & blocked[sl]
        cell[b] = cell[b] * 0.4 + UI_COST * 0.6
    view = cv2.resize(cv2.flip(cell, 0), (size, size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    k = size / side

    # faint 1 m grid
    step = 1.0 / RES * k
    off_x = ((X0 + c0 * RES) % 1.0) / RES * k
    off_y = ((Y0 + (r0 + side) * RES) % 1.0) / RES * k
    for gx in np.arange(-off_x, size, step):
        cv2.line(view, (int(gx), 0), (int(gx), size), UI_GRID, 1)
    for gy in np.arange(off_y, size, step):
        cv2.line(view, (0, int(gy)), (size, int(gy)), UI_GRID, 1)

    # walls on top, slightly thickened and anti-aliased
    wall = cv2.resize(cv2.flip((lo > 0.6).astype(np.float32), 0), (size, size), interpolation=cv2.INTER_LINEAR)
    wall = np.clip(cv2.GaussianBlur(wall, (0, 0), max(0.6, k * 0.25)) * 1.8, 0, 1)[..., None]
    view = (view * (1 - wall) + np.array(UI_WALL, np.float32) * wall).astype(np.uint8)

    def px_of(wx, wy):                 # world -> display pixel (map is flipped so +y is up)
        r, c = to_cell(wx, wy)
        return int((c - c0 + 0.5) * k), int((side - 1 - (r - r0) + 0.5) * k)

    pts = [px_of(px, py) for px, py in (path or [])]
    if len(pts) > 1:
        cv2.polylines(view, [np.int32(pts)], False, UI_PATH, 2, cv2.LINE_AA)

    mr = max(6, int(size / 50))
    _start(view, px_of(*START[:2]), mr)
    for fx, fy in found_targets:
        _apple(view, px_of(fx, fy), mr)
    if target_xy:
        _apple(view, px_of(*target_xy), mr, ghost=True)

    # robot: amber arrowhead pointing along the heading
    p = np.array(px_of(x, y), np.float32)
    fwd = np.array((math.cos(th), -math.sin(th)), np.float32)
    left = np.array((-fwd[1], fwd[0]), np.float32)
    s = mr * 1.6
    tri = np.int32([p + fwd * s, p - fwd * s * 0.6 + left * s * 0.75, p - fwd * s * 0.25, p - fwd * s * 0.6 - left * s * 0.75])
    cv2.fillPoly(view, [tri], UI_ROBOT, cv2.LINE_AA)
    cv2.polylines(view, [tri], True, (30, 22, 18), 1, cv2.LINE_AA)

    _badge(view, f"{TARGET_NAME}  {len(found_targets)}/{TARGET_COUNT}")
    cv2.rectangle(view, (0, 0), (size - 1, size - 1), UI_FRAME, 2)
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
        if yolo is not None:
            draw_yolo(frame)
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
    lv = local_view(disp_local.getWidth() if disp_local else 300)
    if disp_cam:
        _paste(disp_cam, frame)
    else:
        cv2.imshow("camera", frame)
    if disp_local:
        _paste(disp_local, lv)
    else:
        cv2.imshow("local costmap", lv)
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


# ---------------- main loop ----------------
state = "SCAN"
path = None
goal = None
target_xy = None
banned = []
last_plan = -1e9
blocked_since = None
backoff_until = 0.0
scan_turned = 0.0
seen_count = 0
wd_t, wd_xy = 0.0, (x, y)
approach_fails = 0
false_targets = []
found_targets = []                    # apples already reached
verified = 0                          # close-range confirmations of the current target
arrived = False
look_spots = [(x, y)]                 # where 360 deg look-arounds were done

print(f"[sar] mission: find {TARGET_COUNT}x {TARGET_NAME} -> destination {DEST} -> home {START[:2]}")

while robot.step(dt_ms) != -1:
    t = robot.getTime()
    near = arrived; arrived = False
    th_before = th
    update_odometry()
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
    det = detect_target(frame) if TARGET["kind"] == "color" else detect_yolo_target(frame)
    if rejects and t - _last_rej[0] > 3 and max(r[2] for r in rejects) > 80:
        _last_rej[0] = t
        print(f"[rej] t={t:.1f} {rejects[:3]}")
        if SAVE_DEBUG_FILES:
            cv2.imwrite(os.path.join(HERE, f"rej_{int(t)}.jpg"), frame)

    # ---- perception -> target estimate (needs a few consistent detections) ----
    if det and state in ("SCAN", "EXPLORE", "APPROACH") and det[1] < 6.0:
        b, dist = det[0], det[1]
        tx, ty = x + dist * math.cos(th + b), y + dist * math.sin(th + b)
        if any(math.hypot(tx - fx, ty - fy) < 1.0 for fx, fy in false_targets) or \
                any(math.hypot(tx - fx, ty - fy) < 1.5 for fx, fy in found_targets):   # already rescued
            det = None
    if det and state in ("SCAN", "EXPLORE", "APPROACH") and det[1] < 6.0:
        if target_xy is None or math.hypot(tx - target_xy[0], ty - target_xy[1]) > 1.0:
            if state != "APPROACH":
                target_xy = (tx, ty); seen_count = 1; verified = 0
        else:
            a = 0.3
            target_xy = ((1 - a) * target_xy[0] + a * tx, (1 - a) * target_xy[1] + a * ty)
            seen_count += 1
            # close-up confirmation: far away a few red pixels look round (e.g. a red can on its side)
            _, _, (_, _, bw_, bh_), conf_ = det
            if det[1] < VERIFY_DIST and 0.8 < bw_ / max(bh_, 1) < 1.25 and conf_ >= 0.6:
                verified += 1
        if seen_count >= 3 and state != "APPROACH":
            print(f"[sar] {TARGET_NAME} spotted at ({target_xy[0]:.2f}, {target_xy[1]:.2f})")
            state = "APPROACH"; path = None; last_plan = -1e9
            if SAVE_DEBUG_FILES:
                cv2.imwrite(os.path.join(HERE, "spotted.jpg"), frame)

    # ---- safety layer / recovery ----
    if t < backoff_until:
        drive(-0.08, 0.0)
        show(frame, path, det); continue

    if state == "SCAN":
        scan_turned += abs(wrap(th - th_before))
        drive(0, 1.6)
        if scan_turned > 2 * math.pi:
            state = "EXPLORE"; path = None
        show(frame, path, det); continue

    if state == "DONE":
        drive(0, 0); show(frame, path, det); continue

    # choose goal for the current state
    if state == "APPROACH" and any(math.hypot(target_xy[0] - fx, target_xy[1] - fy) < 1.5 for fx, fy in found_targets):
        # far-away distance estimates are poor: the "new" target turned out to be one we already reached
        print(f"[sar] target ({target_xy[0]:.2f},{target_xy[1]:.2f}) is an already-found {TARGET_NAME} -> keep exploring")
        target_xy = None; seen_count = 0; verified = 0
        state = "EXPLORE"; goal = None; path = None; last_plan = -1e9; drive(0, 0); continue
    if state == "APPROACH":
        goal = target_xy
        if (near or math.hypot(goal[0] - x, goal[1] - y) < REACH_DIST) and verified < 2:
            print(f"[sar] {TARGET_NAME} at ({goal[0]:.2f},{goal[1]:.2f}) NOT confirmed up close -> false alarm, keep exploring")
            false_targets.append(target_xy); target_xy = None; seen_count = 0; verified = 0
            state = "EXPLORE"; goal = None; path = None; last_plan = -1e9; drive(0, 0); continue
        if near or math.hypot(goal[0] - x, goal[1] - y) < REACH_DIST:
            print(f"[sar] {TARGET_NAME} confirmed up close ({verified} close views)")
            found_targets.append(target_xy)
            target_xy = None
            n = len(found_targets)
            if n >= TARGET_COUNT:
                print(f"[sar] reached {TARGET_NAME} #{n} at t={t:.1f}s -> all {TARGET_COUNT} found, heading to destination")
                state = "TO_DEST"
            else:
                print(f"[sar] reached {TARGET_NAME} #{n} at t={t:.1f}s -> searching for the next one")
                state = "EXPLORE"; goal = None; target_xy = None; seen_count = 0; approach_fails = 0
            path = None; last_plan = -1e9; drive(0, 0); continue
    elif state == "TO_DEST":
        goal = DEST
        if near or math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOL:
            print(f"[sar] destination reached at t={t:.1f}s -> returning home")
            state = "HOME"; path = None; last_plan = -1e9; drive(0, 0); continue
    elif state == "HOME":
        goal = START[:2]
        if near or math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOL:
            print(f"[sar] MISSION COMPLETE at t={t:.1f}s")
            state = "DONE"; drive(0, 0); continue

    # (re)plan periodically so the map/pedestrian changes are taken into account
    if path is None or t - last_plan > 1.5:
        blocked, penalty = cost_maps()
        if state == "EXPLORE":
            if goal is not None and math.hypot(goal[0] - x, goal[1] - y) < 0.4:
                mark_seen_around(*goal, rad=0.15)   # arrived: don't pick this exact spot again
            if goal is None or path is None or cam_seen[to_cell(*goal)] or math.hypot(goal[0] - x, goal[1] - y) < 0.4 or t - last_plan > 6:
                goal = nearest_frontier(blocked, banned)
                if goal is None:
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

    # progress watchdog: no real motion for 5 s -> back off, forget this goal, replan
    if t - wd_t > 3.5:
        if math.hypot(x - wd_xy[0], y - wd_xy[1]) < 0.08 and path:
            print(f"[sar] stuck at ({x:.2f},{y:.2f}) -> recovery")
            backoff_until = t + 1.0; path = None
            if state == "EXPLORE" and goal:
                banned.append(goal); mark_seen_around(*goal); goal = None
            if state == "APPROACH":
                approach_fails += 1
                if approach_fails >= 3:
                    print("[sar] target unreachable/false -> back to EXPLORE")
                    false_targets.append(target_xy); target_xy = None
                    state = "EXPLORE"; goal = None; approach_fails = 0
        wd_t, wd_xy = t, (x, y)

    clear = front_clearance(ranges)
    if clear < SAFE_FRONT or local_blocked[0]:
        # something right in front / DWA found no safe trajectory (possibly the pedestrian): stop, wait, back off
        local_blocked[0] = False          # re-run DWA next step
        if blocked_since is None:
            blocked_since = t
        if t - blocked_since > 2.5:
            backoff_until = t + 0.8; blocked_since = None; path = None
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
                goal = None; path = None
                # new area reached: 360 deg camera look-around (camera sees only 60 deg ahead,
                # objects tucked beside furniture are otherwise missed)
                if all(math.hypot(x - lx, y - ly) > LOOK_SPACING for lx, ly in look_spots):
                    look_spots.append((x, y)); state = "SCAN"; scan_turned = 0.0
            elif path and goal and math.hypot(goal[0] - x, goal[1] - y) < 0.6:
                # goal sits inside obstacle inflation (e.g. start pose next to a wall): closest safe spot is good enough
                path = None; arrived = True

    if int(t / DT) % 30 == 0:
        print(f"[dbg] t={t:.1f} {state} scanmatch={_match['n']}x/{_match['total']:.2f}m pose=({x:.2f},{y:.2f},{th:.2f}) goal={goal} clear={front_clearance(ranges):.2f} det={det[:2] if det else None}")
    show(frame, path, det)
