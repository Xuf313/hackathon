"""TurtleBot3 Burger - Autonomous Search & Rescue (Webots R2025a, sensors only, no Supervisor).

Mission: find TARGET_COUNT red apples in an unknown apartment -> go to DEST -> return to START.

Perception -> Planning -> Action (lecture pipeline):

  Sensors      wheel encoders (required), 2D LiDAR LDS-01 (required), IMU gyro (optional), camera
  Vision       camera BGRA -> RGB, HSV red Binary Mask (blur + opening/closing, contour, centroid,
               min-enclosing circle) + YOLO11n (COCO dataset) for confirmation and semantics
  SLAM         Mapping      : occupancy grid, Bayesian filter in log-odds form
               Localization : differential-drive odometry (encoder [+ gyro]) prediction
                              -> Monte Carlo particle filter (likelihood-field update, ESS resampling)
                              -> scan matching: ICP (SVD) + scan-to-map optimization
                                 (Gauss-Newton on a bilinear, truncated distance-field cost)
  Planning     Global costmap / local costmap (255 unknown, 254 occupied, 253 inscribed,
               1..252 inflation, 0 free), A* global planner, semantic frontier exploration
  Action       look-ahead (pure pursuit) path tracking checked against the local costmap,
               FSM decision making with recovery behaviours
"""
import heapq
import math
import os
import time

import cv2
import numpy as np
from controller import Robot

# =====================================================================================
# Mission configuration (world frame, metres). The start pose is known (lecture premise:
# no kidnapped-robot problem, initial position & orientation given).
# =====================================================================================
START = (-0.3, -7.5, math.pi)       # TurtleBot3Burger translation/rotation in worlds/apartment*.wbt
DEST = (-4.57, -5.78)               # destination (OrderSign in the apartment world)
TARGET_NAME = "red apple"
TARGET_COUNT = 2                    # red apples to find before going to DEST
REACH_DIST = 0.40                   # target counted as reached (robot centre -> apple centre)
GOAL_TOL = 0.25                     # DEST / START reached
MAX_SEARCH_TIME = 1500.0            # [s sim] fail-safe: stop searching and deliver what was found
LOOK_SPACING = 1.5                  # 360 deg camera look-around when reaching a new area this far away

SHOW_WINDOWS = os.environ.get("TB3_NO_GUI") is None
USE_YOLO = True                     # YOLO11n (COCO) for target confirmation + semantic exploration
YOLO_WEIGHTS = "yolo11n.pt"         # models/YOLO/yolo11n.pt (downloaded automatically if missing)
YOLO_CONF = 0.15
YOLO_EVERY = 5                      # run YOLO every Nth control step
USE_IMU = True                      # fuse the gyro into the heading when the device exists

# =====================================================================================
# Robot constants (lecture notebook: "핵심 파라미터")
# =====================================================================================
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
ROBOT_RADIUS = 0.105
V_MAX = 0.20                        # [m/s]
W_MAX = 1.8                         # [rad/s]
CAM_HEIGHT = 0.073                  # camera optical centre above the floor
CAM_X = 0.02                        # camera forward offset from the base centre
APPLE_D = 0.10                      # apple diameter (RedApple.proto sphere radius 0.05)

# =====================================================================================
# Map / costmap
# =====================================================================================
RES = 0.05                          # [m/cell]
X0, Y0 = -16.0, -15.0               # map origin (lower-left corner)
NX, NY = 440, 440                   # covers x in [-16, 6], y in [-15, 7]

L_OCC = math.log(0.70 / 0.30)       # inverse sensor model, log-odds form
L_FREE = math.log(0.40 / 0.60)
L_MIN, L_MAX = -4.0, 4.0            # clamping keeps the map able to forget moving people
L_OCC_TH = math.log(0.77 / 0.23)    # p > 0.77 -> occupied (>= 2 consistent hits: a passing person is not a wall)
L_FREE_TH = math.log(0.35 / 0.65)   # p < 0.35 -> free

COST_UNKNOWN, COST_LETHAL, COST_INSCRIBED, COST_FREE = 255, 254, 253, 0
INSCRIBED_R = ROBOT_RADIUS + 0.02   # robot centre closer than this to an obstacle = collision
INFLATION_R = 0.50                  # cost decays to 0 at this distance
COST_SCALE = 6.0                    # exponential decay rate of the inflation cost

LOCAL_SIZE = 3.0                    # rolling-window local costmap (robot centred, world aligned)
LN = int(LOCAL_SIZE / RES)

# Localization
N_PARTICLES = 150
PF_BEAMS = 60                       # beams used by the particle filter / scan matcher
PF_SIGMA = 0.08                     # likelihood-field std-dev [m]
PF_Z_HIT, PF_Z_RAND = 0.9, 0.1
PF_EFFECTIVE_BEAMS = 12.0           # beams are correlated: temper the likelihood
LOC_EVERY = 2                       # correction every 2 control steps (5-10 Hz), prediction every step
TRUNC = 0.30                        # scan-to-map cost truncation (robust to people / unmapped objects)
SCAN_BEAMS = 180                    # beams used by the scan-to-map optimisation
SCAN_SIGMA = 0.05                   # scan residual std-dev [m]
SCAN_N_EFF = 30.0                   # effective independent beams
PRIOR_XY = (0.005, 0.10)            # prior std-dev: base + per metre travelled
PRIOR_TH = (0.003, 0.10)            # prior std-dev: base + per radian turned

# Look-ahead controller
LOOKAHEAD = 0.45
SAFE_T = 1.2                        # forward-simulated horizon for the local costmap check [s]

# =====================================================================================
# Devices
# =====================================================================================
robot = Robot()
DT_MS = int(robot.getBasicTimeStep())
DT = DT_MS / 1000.0

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for _m in (left_motor, right_motor):
    _m.setPosition(float("inf"))
    _m.setVelocity(0.0)
MAX_WHEEL = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(DT_MS)
right_encoder.enable(DT_MS)

lidar = robot.getDevice("LDS-01")
lidar.enable(DT_MS)
NBEAM = lidar.getHorizontalResolution()
LMAX = lidar.getMaxRange()
BEAM_ANG = math.pi - np.arange(NBEAM) * 2 * math.pi / NBEAM    # idx N/2 = front, N/4 = left

camera = robot.getDevice("camera")
camera.enable(DT_MS)
CW, CH = camera.getWidth(), camera.getHeight()
FOCAL = (CW / 2) / math.tan(camera.getFov() / 2)
CAM_HALF_FOV = camera.getFov() / 2


def _optional_device(name):
    try:
        return robot.getDevice(name)
    except Exception:
        return None


gyro = _optional_device("gyro") if USE_IMU else None
if gyro is not None:
    gyro.enable(DT_MS)

_LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "autonomous.log"), "w")


def log(*a):
    msg = f"[{robot.getTime():7.1f}] " + " ".join(map(str, a))
    print(msg)
    _LOG.write(msg + "\n")
    _LOG.flush()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_cell(x, y):
    return int((y - Y0) / RES), int((x - X0) / RES)


def to_world(r, c):
    return X0 + (c + 0.5) * RES, Y0 + (r + 0.5) * RES


def in_map(r, c):
    return 0 <= r < NY and 0 <= c < NX


def world_to_rc(px, py):
    """Vectorised world -> (row, col) as float cell coordinates (cell centres at .0)."""
    return (py - Y0) / RES - 0.5, (px - X0) / RES - 0.5


# =====================================================================================
# 1. Odometry: differential drive robot, wheel encoders (+ optional gyro)
#    ds = (dr + dl) / 2,  dtheta = (dr - dl) / L
# =====================================================================================
class Odometry:
    def __init__(self):
        self.prev = None
        self.gyro_sign = None          # auto-detected during the first spin (+1 / -1 / 0 = unusable)
        self.gyro_axis = 2
        self._cal_enc = 0.0
        self._cal_gyro = np.zeros(3)

    def step(self):
        """Returns the motion increment (ds, dtheta) since the previous call."""
        l, r = left_encoder.getValue(), right_encoder.getValue()
        if self.prev is None or not (math.isfinite(l) and math.isfinite(r)):
            self.prev = (l, r)
            return 0.0, 0.0
        dl = (l - self.prev[0]) * WHEEL_RADIUS
        dr = (r - self.prev[1]) * WHEEL_RADIUS
        self.prev = (l, r)
        ds = (dr + dl) / 2.0
        dth = (dr - dl) / WHEEL_SEPARATION
        if gyro is not None:
            g = np.array(gyro.getValues()) * DT
            if self.gyro_sign is None:
                # calibrate: which gyro axis / sign matches the encoder yaw rate?
                self._cal_enc += abs(dth)
                self._cal_gyro += g * math.copysign(1.0, dth) if abs(dth) > 1e-6 else 0.0
                if self._cal_enc > 1.0:
                    self.gyro_axis = int(np.argmax(np.abs(self._cal_gyro)))
                    ratio = self._cal_gyro[self.gyro_axis] / self._cal_enc
                    self.gyro_sign = (1 if ratio > 0 else -1) if 0.7 < abs(ratio) < 1.3 else 0
                    log(f"[imu] gyro axis={self.gyro_axis} ratio={ratio:.2f} ->",
                        "fused" if self.gyro_sign else "ignored")
            elif self.gyro_sign:
                # gyro does not suffer from wheel slip: trust it for the heading increment
                dth = 0.85 * self.gyro_sign * g[self.gyro_axis] + 0.15 * dth
        return ds, dth


# =====================================================================================
# 2. SLAM - Mapping: occupancy grid, Bayesian filter in log-odds
#    l_t = l_{t-1} + log(p(occ|z)/(1-p(occ|z))) - l_0     (l_0 = 0 for prior 0.5)
#    p = 1 - 1 / (1 + exp(l))
# =====================================================================================
class OccupancyGrid:
    def __init__(self):
        self.logodds = np.zeros((NY, NX), np.float32)
        self.observed = np.zeros((NY, NX), bool)
        self._ray_s = np.arange(0.0, LMAX, RES * 0.8)[None, :]

    def update(self, pose, ranges):
        x, y, th = pose
        idx = np.arange(0, NBEAM, 2)
        r = ranges[idx]
        valid = np.isfinite(r) & (r > 0.12)
        hit = valid & (r < LMAX - 0.05)
        r = np.where(valid, np.minimum(r, LMAX), LMAX * 0.9)    # no return -> free up to near max range
        ang = th + BEAM_ANG[idx]
        free_mask = self._ray_s < (r[:, None] - RES)
        px = x + self._ray_s * np.cos(ang)[:, None]
        py = y + self._ray_s * np.sin(ang)[:, None]
        self._add(px[free_mask], py[free_mask], L_FREE)
        self._add(x + r[hit] * np.cos(ang[hit]), y + r[hit] * np.sin(ang[hit]), L_OCC)
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)

    def _add(self, px, py, l):
        rr = ((py - Y0) / RES).astype(np.int32)
        cc = ((px - X0) / RES).astype(np.int32)
        ok = (rr >= 0) & (rr < NY) & (cc >= 0) & (cc < NX)
        flat = np.unique(rr[ok] * NX + cc[ok])       # each cell updated once per scan
        self.logodds.flat[flat] += l
        self.observed.flat[flat] = True

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def occupied(self):
        return self.logodds > L_OCC_TH

    def free(self):
        return self.observed & (self.logodds < L_FREE_TH)


# =====================================================================================
# 3. SLAM - Localization
#    Distance field of the map = the "expected scan" used by both the particle filter and the
#    scan matcher. Bilinear interpolation makes the cost continuous (sub-cell accuracy).
# =====================================================================================
class DistanceField:
    def __init__(self):
        self.dist = np.full((NY, NX), 1.0, np.float32)
        self.nn_r = np.zeros((NY, NX), np.int32)       # nearest occupied cell (for ICP correspondences)
        self.nn_c = np.zeros((NY, NX), np.int32)
        self.n_occ = 0

    def rebuild(self, occ):
        self.n_occ = int(occ.sum())
        src = np.where(occ, 0, 255).astype(np.uint8)
        d, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        self.dist = np.minimum(d * RES, 1.0).astype(np.float32)
        # label -> coordinates of that zero pixel
        zr, zc = np.nonzero(src == 0)
        if zr.size:
            lut_r = np.zeros(labels.max() + 1, np.int32)
            lut_c = np.zeros(labels.max() + 1, np.int32)
            lab_at_zero = labels[zr, zc]
            lut_r[lab_at_zero], lut_c[lab_at_zero] = zr, zc
            self.nn_r, self.nn_c = lut_r[labels], lut_c[labels]

    def sample(self, px, py):
        """Bilinear distance lookup, 1.0 m outside the map."""
        fr, fc = world_to_rc(px, py)
        r0, c0 = np.floor(fr).astype(np.int32), np.floor(fc).astype(np.int32)
        ok = (r0 >= 0) & (r0 < NY - 1) & (c0 >= 0) & (c0 < NX - 1)
        r0c, c0c = np.clip(r0, 0, NY - 2), np.clip(c0, 0, NX - 2)
        ar, ac = fr - r0c, fc - c0c
        d = self.dist
        v = ((1 - ar) * (1 - ac) * d[r0c, c0c] + (1 - ar) * ac * d[r0c, c0c + 1]
             + ar * (1 - ac) * d[r0c + 1, c0c] + ar * ac * d[r0c + 1, c0c + 1])
        return np.where(ok, v, 1.0)

    def gradient(self, px, py):
        e = RES * 0.5
        gx = (self.sample(px + e, py) - self.sample(px - e, py)) / (2 * e)
        gy = (self.sample(px, py + e) - self.sample(px, py - e)) / (2 * e)
        return gx, gy


def scan_points(ranges, n=None):
    """Subsampled LiDAR hits in the robot frame: (k, 2)."""
    idx = np.linspace(0, NBEAM - 1, n or PF_BEAMS).astype(int)
    r = ranges[idx]
    ok = np.isfinite(r) & (r > 0.12) & (r < LMAX - 0.05)
    a = BEAM_ANG[idx][ok]
    return np.stack([r[ok] * np.cos(a), r[ok] * np.sin(a)], axis=1)


def transform(pts, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.stack([pose[0] + c * pts[:, 0] - s * pts[:, 1], pose[1] + s * pts[:, 0] + c * pts[:, 1]], axis=1)


class ParticleFilter:
    """Monte Carlo Localization: initialization -> prediction -> update -> resampling."""

    def __init__(self, pose):
        # 1. Initialization: known start pose -> particles around it, equal weights 1/n
        self.p = np.tile(np.array(pose, np.float64), (N_PARTICLES, 1))
        self.p[:, :2] += np.random.normal(0, 0.02, (N_PARTICLES, 2))
        self.p[:, 2] += np.random.normal(0, 0.02, N_PARTICLES)
        self.w = np.full(N_PARTICLES, 1.0 / N_PARTICLES)
        self.ess = float(N_PARTICLES)

    def predict(self, ds, dth):
        """2. Prediction: move every particle with the odometry increment + encoder noise model."""
        if abs(ds) < 1e-6 and abs(dth) < 1e-6:
            return
        n = N_PARTICLES
        s_trans = 0.05 * abs(ds) + 0.01 * abs(dth)
        s_rot = 0.05 * abs(dth) + 0.10 * abs(ds)
        nds = ds + np.random.normal(0, s_trans, n)
        ndth = dth + np.random.normal(0, s_rot, n)
        mid = self.p[:, 2] + ndth / 2
        self.p[:, 0] += nds * np.cos(mid)
        self.p[:, 1] += nds * np.sin(mid)
        self.p[:, 2] = (self.p[:, 2] + ndth + np.pi) % (2 * np.pi) - np.pi

    def update(self, pts, field):
        """3. Update: compare the real scan with the map (likelihood field) -> P(z | particle)."""
        if len(pts) < 10:
            return
        c, s = np.cos(self.p[:, 2])[:, None], np.sin(self.p[:, 2])[:, None]
        wx = self.p[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
        wy = self.p[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
        d = field.sample(wx, wy)
        lik = PF_Z_HIT * np.exp(-0.5 * (d / PF_SIGMA) ** 2) + PF_Z_RAND
        loglik = np.log(lik).mean(axis=1) * PF_EFFECTIVE_BEAMS
        logw = np.log(self.w + 1e-300) + loglik
        logw -= logw.max()
        w = np.exp(logw)
        self.w = w / w.sum()
        # 4. Resampling only when needed: ESS = 1 / sum(w^2)
        self.ess = 1.0 / np.sum(self.w ** 2)
        if self.ess < N_PARTICLES / 2:
            self.resample()

    def resample(self):
        """Systematic resampling (CDF intervals) + jittering against particle depletion."""
        n = N_PARTICLES
        positions = (np.arange(n) + np.random.uniform()) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), positions), n - 1)
        self.p = self.p[idx]
        self.p[:, :2] += np.random.normal(0, 0.01, (n, 2))
        self.p[:, 2] += np.random.normal(0, 0.005, n)
        self.w = np.full(n, 1.0 / n)

    def estimate(self):
        x = np.sum(self.w * self.p[:, 0])
        y = np.sum(self.w * self.p[:, 1])
        th = math.atan2(np.sum(self.w * np.sin(self.p[:, 2])), np.sum(self.w * np.cos(self.p[:, 2])))
        return np.array([x, y, th])

    def shift(self, delta):
        """Move the whole cloud by the scan-matching correction (keeps its shape)."""
        self.p[:, 0] += delta[0]
        self.p[:, 1] += delta[1]
        self.p[:, 2] = (self.p[:, 2] + delta[2] + np.pi) % (2 * np.pi) - np.pi


def icp(pts, pose, field, iters=15, max_corr=0.30):
    """Scan-to-map ICP: closest occupied cell correspondences + SVD (Kabsch).
        H = P'^T Q' = U S V^T,  R = V U^T,  t = q_mean - R p_mean
    Returns (pose, mean residual, #correspondences)."""
    pose = np.array(pose, np.float64)
    err, n = float("inf"), 0
    for _ in range(iters):
        P = transform(pts, pose)
        rr, cc = world_to_rc(P[:, 0], P[:, 1])
        rr, cc = np.round(rr).astype(int), np.round(cc).astype(int)
        ok = (rr >= 0) & (rr < NY) & (cc >= 0) & (cc < NX)
        rr, cc, P = rr[ok], cc[ok], P[ok]
        d = field.dist[rr, cc]
        m = d < max_corr
        if m.sum() < 15:
            return pose, float("inf"), int(m.sum())
        P = P[m]
        qx, qy = to_world(field.nn_r[rr[m], cc[m]], field.nn_c[rr[m], cc[m]])
        Q = np.stack([qx, qy], axis=1)
        # trim the worst 20 % (people, unmapped furniture)
        res = np.hypot(*(P - Q).T)
        keep = res <= np.quantile(res, 0.8)
        P, Q = P[keep], Q[keep]
        pm, qm = P.mean(0), Q.mean(0)
        H = (P - pm).T @ (Q - qm)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        dth = math.atan2(R[1, 0], R[0, 0])
        t = qm - R @ pm
        # compose: new world pose = R * old + t
        pose[:2] = R @ pose[:2] + t
        pose[2] = wrap(pose[2] + dth)
        err, n = float(res[keep].mean()), int(keep.sum())
        if abs(dth) < 1e-4 and np.hypot(*t) < 1e-4:
            break
    return pose, err, n


def scan_cost(pts, pose, field, prior=None, sig=None):
    """Accurate scan-to-map cost (MAP form):
        C(x) = N_eff / (n sigma_d^2) * sum_i min(D(T_x p_i), TRUNC)^2          (scan vs map)
             + |xy - xy_pred|^2 / sigma_xy^2 + (th - th_pred)^2 / sigma_th^2  (odometry / IMU prior)
    D = bilinearly interpolated distance field (sub-cell accurate), truncation = robust to people
    and unmapped objects, the prior stops a few misleading points from dragging the pose."""
    P = transform(pts, pose)
    d = np.minimum(field.sample(P[:, 0], P[:, 1]), TRUNC)
    c = SCAN_N_EFF * float(np.mean(d ** 2)) / SCAN_SIGMA ** 2
    if prior is not None:
        c += ((pose[0] - prior[0]) ** 2 + (pose[1] - prior[1]) ** 2) / sig[0] ** 2
        c += wrap(pose[2] - prior[2]) ** 2 / sig[1] ** 2
    return c


def scan_to_map_optimize(pts, pose, field, prior, sig, iters=10):
    """Scan-to-map optimisation: Gauss-Newton / Levenberg-Marquardt on scan_cost."""
    pose = np.array(pose, np.float64)
    cost = scan_cost(pts, pose, field, prior, sig)
    lam = 1e-3
    for _ in range(iters):
        P = transform(pts, pose)
        d = field.sample(P[:, 0], P[:, 1])
        m = d < TRUNC                                     # inliers only (outliers have zero gradient)
        if m.sum() < 15:
            break
        gx, gy = field.gradient(P[m, 0], P[m, 1])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        lx, ly = pts[m, 0], pts[m, 1]
        dpx_dth = -s * lx - c * ly
        dpy_dth = c * lx - s * ly
        k = math.sqrt(SCAN_N_EFF / len(pts)) / SCAN_SIGMA
        J = k * np.stack([gx, gy, gx * dpx_dth + gy * dpy_dth], axis=1)
        r = k * d[m]
        # prior rows
        Jp = np.diag([1 / sig[0], 1 / sig[0], 1 / sig[1]])
        rp = np.array([(pose[0] - prior[0]) / sig[0], (pose[1] - prior[1]) / sig[0], wrap(pose[2] - prior[2]) / sig[1]])
        J, r = np.vstack([J, Jp]), np.concatenate([r, rp])
        A = J.T @ J
        A += lam * np.diag(np.diag(A))
        delta = -np.linalg.solve(A, J.T @ r)
        delta[:2] = np.clip(delta[:2], -0.05, 0.05)
        delta[2] = float(np.clip(delta[2], -0.05, 0.05))
        cand = pose + delta
        cand[2] = wrap(cand[2])
        new_cost = scan_cost(pts, cand, field, prior, sig)
        if new_cost < cost:
            pose, cost, lam = cand, new_cost, lam * 0.3
            if np.abs(delta).max() < 1e-4:
                break
        else:
            lam *= 10
    return pose, cost


# =====================================================================================
# 4. Costmaps (nav2 convention)
#    255 unknown | 254 occupied (lethal) | 253 inscribed | 1..252 inflation | 0 free
# =====================================================================================
def inflate(occ, unknown=None):
    dist = cv2.distanceTransform(np.where(occ, 0, 255).astype(np.uint8), cv2.DIST_L2, 5) * RES
    cost = np.zeros(occ.shape, np.uint8)
    band = (dist > INSCRIBED_R) & (dist < INFLATION_R)
    cost[band] = np.clip(252 * np.exp(-COST_SCALE * (dist[band] - INSCRIBED_R)), 1, 252).astype(np.uint8)
    cost[dist <= INSCRIBED_R] = COST_INSCRIBED
    cost[occ] = COST_LETHAL
    if unknown is not None:
        cost[unknown & (cost < COST_INSCRIBED)] = COST_UNKNOWN
    return cost


def global_costmap(grid):
    occ = grid.occupied()
    unknown = ~grid.observed
    return inflate(occ, unknown)


class LocalCostmap:
    """Rolling window around the robot built from the CURRENT scan only, so a walking person
    shows up immediately and leaves no ghost behind."""

    def __init__(self):
        self.cost = np.zeros((LN, LN), np.uint8)
        self.dist = np.full((LN, LN), LOCAL_SIZE, np.float32)
        self.origin = (0.0, 0.0)

    def update(self, pose, ranges):
        x, y, th = pose
        self.origin = (x - LOCAL_SIZE / 2, y - LOCAL_SIZE / 2)
        ok = np.isfinite(ranges) & (ranges > 0.12) & (ranges < LOCAL_SIZE)
        a = th + BEAM_ANG[ok]
        px, py = x + ranges[ok] * np.cos(a), y + ranges[ok] * np.sin(a)
        c = ((px - self.origin[0]) / RES).astype(int)
        r = ((py - self.origin[1]) / RES).astype(int)
        m = (r >= 0) & (r < LN) & (c >= 0) & (c < LN)
        occ = np.zeros((LN, LN), bool)
        occ[r[m], c[m]] = True
        self.cost = inflate(occ)
        self.dist = cv2.distanceTransform(np.where(occ, 0, 255).astype(np.uint8), cv2.DIST_L2, 5) * RES

    def clearance(self, px, py):
        c = int((px - self.origin[0]) / RES)
        r = int((py - self.origin[1]) / RES)
        return float(self.dist[r, c]) if 0 <= r < LN and 0 <= c < LN else LOCAL_SIZE

    def at(self, px, py):
        c = ((np.asarray(px) - self.origin[0]) / RES).astype(int)
        r = ((np.asarray(py) - self.origin[1]) / RES).astype(int)
        ok = (r >= 0) & (r < LN) & (c >= 0) & (c < LN)
        return np.where(ok, self.cost[np.clip(r, 0, LN - 1), np.clip(c, 0, LN - 1)], 0)

    def stamp_into(self, gcost):
        """Overlay local obstacles (e.g. a person) onto the global costmap for replanning."""
        r0, c0 = to_cell(self.origin[0] + RES / 2, self.origin[1] + RES / 2)
        rs, cs = max(0, r0), max(0, c0)
        re_, ce = min(NY, r0 + LN), min(NX, c0 + LN)
        if rs >= re_ or cs >= ce:
            return gcost
        sub = self.cost[rs - r0:re_ - r0, cs - c0:ce - c0]
        g = gcost[rs:re_, cs:ce]
        known = g != COST_UNKNOWN
        g[known] = np.maximum(g[known], sub[known])
        return gcost


# =====================================================================================
# 5. Global planner: A* on the global costmap, f(n) = g(n) + h(n)
#    Planned on a 2x coarser grid (max-pooled cost) to keep Python fast.
# =====================================================================================
PLAN_DS = 2
NB8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


def coarse(cost):
    h, w = cost.shape[0] // PLAN_DS, cost.shape[1] // PLAN_DS
    c = cost[:h * PLAN_DS, :w * PLAN_DS].reshape(h, PLAN_DS, w, PLAN_DS)
    lethal = (c == COST_LETHAL) | (c == COST_INSCRIBED)
    known = np.where(c == COST_UNKNOWN, 0, c).max(axis=(1, 3))
    allunk = (c == COST_UNKNOWN).all(axis=(1, 3))
    out = np.where(allunk, COST_UNKNOWN, known).astype(np.uint8)
    out[lethal.any(axis=(1, 3))] = COST_INSCRIBED
    return out


def nearest_traversable(cost, cell, allow_unknown, radius=12):
    r, c = cell
    if in_grid(cost, r, c) and traversable(cost[r, c], allow_unknown):
        return cell
    best, bd = None, 1e9
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            n = (r + dr, c + dc)
            if in_grid(cost, *n) and traversable(cost[n], allow_unknown):
                d = dr * dr + dc * dc
                if d < bd:
                    best, bd = n, d
    return best


def in_grid(a, r, c):
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1]


def traversable(v, allow_unknown):
    return v < COST_INSCRIBED or (allow_unknown and v == COST_UNKNOWN)


def astar(cost_full, start_xy, goal_xy, allow_unknown=False, max_expand=60000):
    cost = coarse(cost_full)
    H, W = cost.shape

    def cell(p):
        r, c = to_cell(*p)
        return r // PLAN_DS, c // PLAN_DS

    s = nearest_traversable(cost, cell(start_xy), True, radius=6)
    g = nearest_traversable(cost, cell(goal_xy), allow_unknown, radius=8)
    if s is None or g is None:
        return None
    gscore = np.full((H, W), np.inf)
    parent = {}
    closed = np.zeros((H, W), bool)
    gscore[s] = 0.0
    pq = [(0.0, s)]
    expanded = 0
    while pq:
        _, cur = heapq.heappop(pq)
        if closed[cur]:
            continue
        if cur == g:
            break
        closed[cur] = True
        expanded += 1
        if expanded > max_expand:
            return None
        g0 = gscore[cur]
        for dr, dc, step in NB8:
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or closed[n]:
                continue
            v = cost[n]
            if not traversable(v, allow_unknown) and n != g:
                continue
            w = step * (1.0 + (3.0 if v == COST_UNKNOWN else 4.0 * v / 252.0))
            ng = g0 + w
            if ng < gscore[n]:
                gscore[n] = ng
                parent[n] = cur
                h = math.hypot(n[0] - g[0], n[1] - g[1])
                heapq.heappush(pq, (ng + h, n))
    if g != s and g not in parent:
        return None
    out = []
    c = g
    while c is not None:
        out.append(to_world(c[0] * PLAN_DS + PLAN_DS // 2, c[1] * PLAN_DS + PLAN_DS // 2))
        c = parent.get(c)
    out.reverse()
    gr, gc = to_cell(*goal_xy)
    if in_map(gr, gc) and traversable(cost_full[gr, gc], allow_unknown):
        out[-1] = tuple(goal_xy)                          # exact goal when it is itself reachable
    return smooth(out)


def smooth(path, iters=30, a=0.5, b=0.25):
    """Gradient-style path smoothing (keeps end points)."""
    if len(path) < 3:
        return path
    p = np.array(path, float)
    q = p.copy()
    for _ in range(iters):
        q[1:-1] += a * (p[1:-1] - q[1:-1]) + b * (q[:-2] + q[2:] - 2 * q[1:-1])
    return [tuple(v) for v in q]


def geodesic_distance(trav, start, max_steps=600):
    """BFS wavefront (8-connected) as repeated constrained dilation: path-length estimates to every
    reachable cell, used to rank frontiers."""
    dist = np.full(trav.shape, np.inf, np.float32)
    if not in_grid(trav, *start):
        return dist
    cur = np.zeros(trav.shape, np.uint8)
    cur[start] = 1
    visited = cur.astype(bool)
    dist[start] = 0
    k3 = np.ones((3, 3), np.uint8)
    for step in range(1, max_steps):
        nxt = (cv2.dilate(cur, k3) > 0) & trav & ~visited
        if not nxt.any():
            break
        dist[nxt] = step * RES
        visited |= nxt
        cur = nxt.astype(np.uint8)
    return dist


# =====================================================================================
# 6. Perception: RGBA -> RGB, red Binary Mask, contour / centroid / min enclosing circle,
#    YOLO11n (COCO) confirmation + semantic object map
# =====================================================================================
RED_LOWER1, RED_UPPER1 = np.array([0, 120, 50]), np.array([8, 255, 255])
RED_LOWER2, RED_UPPER2 = np.array([172, 120, 50]), np.array([180, 255, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
COCO_APPLE, COCO_ORANGE, COCO_BALL = 47, 49, 32


def camera_rgb():
    """Webots gives BGRA bytes -> numpy -> RGB (drop alpha)."""
    bgra = np.frombuffer(camera.getImage(), np.uint8).reshape((CH, CW, 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)


def red_mask(rgb):
    blur = cv2.GaussianBlur(rgb, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    m = cv2.bitwise_or(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1), cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return m


def detect_red_apple(rgb, yolo_boxes):
    """Returns dict(bearing, dist, circle, conf, bbox) for the best red-apple blob, or None."""
    mask = red_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx <= 1 or bx + bw >= CW - 1:
            continue                                       # cut by the image border
        per = cv2.arcLength(cnt, True)
        circ = 4 * math.pi * area / (per * per + 1e-9)
        if circ < 0.55 or not (0.6 < bw / bh < 1.6):
            continue
        (ccx, ccy), rad = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] else (ccx, ccy)
        # distance from the known apple size (pinhole: D_px = f * D / Z) ...
        d_size = FOCAL * APPLE_D / max(2 * rad, 1.0)
        # ... and from the ground plane (the apple touches the floor)
        bottom = by + bh
        depress = math.atan2(bottom - CH / 2, FOCAL)
        d_ground = CAM_HEIGHT / math.tan(depress) if depress > 0.005 else float("inf")
        if bottom < CH / 2:                                # above the horizon: not on the floor
            continue
        if math.isfinite(d_ground) and not (0.4 < d_size / d_ground < 2.5):
            continue                                       # size does not match an apple on the floor
        # tall red object (fire extinguisher, cabinet)? red continues above the blob
        top = max(0, int(by - 1.5 * bh))
        above = mask[top:max(top, by - 2), bx:bx + bw]
        if above.size and (above > 0).mean() > 0.15:
            continue
        # the size estimate does not depend on the camera height -> main estimate
        dist = d_size if not math.isfinite(d_ground) else 0.75 * d_size + 0.25 * d_ground
        dist += CAM_X                                      # from the robot centre
        conf = min(1.0, circ)
        for x1, y1, x2, y2, cls, name, c in yolo_boxes:
            if cls in (COCO_APPLE, COCO_ORANGE, COCO_BALL) and x1 - 5 <= cx <= x2 + 5 and y1 - 5 <= cy <= y2 + 5:
                conf = min(1.0, conf + 0.5 * c + 0.2)      # YOLO agrees: much more confident
        if best is None or area > best["area"]:
            best = dict(bearing=math.atan2(CW / 2 - cx, FOCAL), dist=dist, circle=((int(cx), int(cy)), int(rad)),
                        conf=conf, bbox=(bx, by, bw, bh), area=area)
    return best, mask


class Yolo:
    def __init__(self):
        self.model = None
        self.boxes = []
        if not USE_YOLO:
            return
        try:
            from ultralytics import YOLO
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.normpath(os.path.join(here, "../../models/YOLO", YOLO_WEIGHTS))
            self.model = YOLO(path if os.path.exists(path) else YOLO_WEIGHTS)
            self.model.to("cpu")
            log("[yolo] loaded", YOLO_WEIGHTS)
        except Exception as e:                         # mission still works with the colour mask only
            log("[yolo] disabled:", e)

    def run(self, rgb):
        if self.model is None:
            return []
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)      # ultralytics expects BGR numpy frames
        res = self.model.predict(source=bgr, conf=YOLO_CONF, iou=0.5, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            if (x2 - x1) * (y2 - y1) > 0.35 * CW * CH and c < 0.5:
                continue                                # huge low-confidence boxes: floor/wall hallucinations
            out.append((int(x1), int(y1), int(x2), int(y2), int(k), res.names[int(k)], float(c)))
        self.boxes = out
        return out


# Semantic prior: how strongly a COCO class suggests "an apple is nearby" (kitchen / dining context)
APPLE_PRIOR = {"dining table": 1.0, "bowl": 0.9, "refrigerator": 0.8, "oven": 0.7, "microwave": 0.6,
               "sink": 0.5, "cup": 0.5, "wine glass": 0.4, "bottle": 0.4, "chair": 0.5, "banana": 0.9,
               "orange": 0.9, "apple": 1.0, "couch": 0.2, "potted plant": 0.2, "tv": 0.1, "bed": 0.1}
INDOOR = set(APPLE_PRIOR) | {"toilet", "laptop", "clock", "vase", "book", "sports ball", "teddy bear",
                             "keyboard", "mouse"}


class SemanticMap:
    def __init__(self):
        self.objects = []                               # [name, x, y, hits]

    def add(self, boxes, ranges, pose):
        x, y, th = pose
        for x1, y1, x2, y2, cls, name, conf in boxes:
            if name not in INDOOR:
                continue                                # person is dynamic; outdoor classes = hallucinations
            b1, b2 = math.atan2(CW / 2 - x1, FOCAL), math.atan2(CW / 2 - x2, FOCAL)
            idx = [int(round((math.pi - b) * NBEAM / (2 * math.pi))) % NBEAM for b in np.linspace(b2, b1, 7)]
            r = ranges[idx]
            r = r[np.isfinite(r) & (r > 0.12)]
            if r.size == 0 or np.median(r) > 3.4:
                continue
            d, b = float(np.median(r)), (b1 + b2) / 2
            ox, oy = x + d * math.cos(th + b), y + d * math.sin(th + b)
            for o in self.objects:
                if o[0] == name and math.hypot(o[1] - ox, o[2] - oy) < 0.6:
                    k = o[3]
                    o[1], o[2], o[3] = (o[1] * k + ox) / (k + 1), (o[2] * k + oy) / (k + 1), min(k + 1, 20)
                    break
            else:
                self.objects.append([name, ox, oy, 1])

    def likelihood(self, wx, wy, sigma=1.5):
        return sum(APPLE_PRIOR.get(n, 0.0) * math.exp(-math.hypot(wx - ox, wy - oy) / sigma)
                   for n, ox, oy, h in self.objects if h >= 2)


# =====================================================================================
# 7. Frontier exploration with semantics
#    frontier = free cell next to unknown; also free floor the camera has not looked at yet
#    score    = path length - W_SIZE * size - W_SEM * semantic likelihood
# =====================================================================================
W_SIZE, W_SEM, W_UNSEEN = 0.02, 2.5, 0.5


def frontier_candidates(grid, gcost, cam_seen, pose, banned):
    free = grid.free()
    unknown = (~grid.observed).astype(np.uint8)
    frontier = free & (cv2.dilate(unknown, np.ones((3, 3), np.uint8)) > 0)
    unseen = cv2.erode((free & ~cam_seen).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    trav = gcost < COST_INSCRIBED
    start = nearest_traversable(gcost, to_cell(pose[0], pose[1]), False, radius=6)
    if start is None:
        return []
    dist = geodesic_distance(trav, start)
    # frontier cells themselves may lie inside inflation: measure from their reachable neighbourhood
    reach = cv2.erode(np.where(np.isfinite(dist), dist, 1e4).astype(np.float32), np.ones((5, 5), np.uint8))
    cands = []
    for kind, mask in (("frontier", frontier), ("unseen", unseen)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            size = stats[i, cv2.CC_STAT_AREA]
            if size < (6 if kind == "frontier" else 20):
                continue
            rr, cc = np.nonzero(labels == i)
            dd = reach[rr, cc]
            j = int(np.argmin(dd))
            if dd[j] >= 1e4:
                continue                                  # not reachable through known free space
            # goal = reachable cell of this cluster closest to its centroid
            mr, mc = rr.mean(), cc.mean()
            ok = dd < 1e4
            k = int(np.argmin(np.where(ok, (rr - mr) ** 2 + (cc - mc) ** 2, np.inf)))
            gx, gy = to_world(rr[k], cc[k])
            if any(math.hypot(gx - bx, gy - by) < 0.5 for bx, by in banned):
                continue
            if math.hypot(gx - pose[0], gy - pose[1]) < 0.35:
                continue
            cands.append(dict(kind=kind, xy=(gx, gy), size=int(size), dist=float(dd[k])))
    return cands


def choose_frontier(cands, sem):
    best, best_s = None, float("inf")
    for c in cands:
        s = c["dist"] - W_SIZE * min(c["size"], 150) - W_SEM * sem.likelihood(*c["xy"])
        if c["kind"] == "unseen":
            s += W_UNSEEN
        c["score"] = s
        if s < best_s:
            best, best_s = c, s
    return best


# =====================================================================================
# 8. Action: look-ahead (pure pursuit) controller + local costmap safety check
#    kappa = 2 y_LA / (x_LA^2 + y_LA^2),  w = v kappa,  v_r = v + wL/2,  v_l = v - wL/2
# =====================================================================================
def drive(v, w):
    v = float(np.clip(v, -V_MAX, V_MAX))
    w = float(np.clip(w, -W_MAX, W_MAX))
    vr = v + w * WHEEL_SEPARATION / 2
    vl = v - w * WHEEL_SEPARATION / 2
    wr, wl = vr / WHEEL_RADIUS, vl / WHEEL_RADIUS
    lim = 0.98 * MAX_WHEEL
    k = max(1.0, abs(wl) / lim, abs(wr) / lim)
    left_motor.setVelocity(wl / k)
    right_motor.setVelocity(wr / k)
    return v, w


def arc_collides(pose, v, w, local, horizon=SAFE_T):
    """Forward-simulate (v, w) on the local costmap. Entering the inscribed zone is a collision;
    if the robot is already inside it (squeezed next to furniture), only motions that get
    even closer count, so it can always turn or back away."""
    x, y, th = pose
    limit = min(INSCRIBED_R, local.clearance(x, y) - 0.01)
    for _ in range(int(horizon / 0.1)):
        th += w * 0.1
        x += v * math.cos(th) * 0.1
        y += v * math.sin(th) * 0.1
        if local.clearance(x, y) < limit:
            return True
    return False


class PathFollower:
    def __init__(self):
        self.path = []
        self.idx = 0
        self.last_la = None

    def set(self, path):
        self.path = list(path or [])
        self.idx = 0

    def remaining(self, pose):
        if not self.path:
            return 0.0
        return math.hypot(self.path[-1][0] - pose[0], self.path[-1][1] - pose[1])

    def step(self, pose, local):
        """Returns ('done' | 'blocked' | 'ok', v, w)."""
        if not self.path:
            return "done", 0.0, 0.0
        x, y, th = pose
        pts = np.array(self.path)
        # 1. nearest waypoint (Euclidean), searched forward only so we never jump back
        win = pts[self.idx:self.idx + 40]
        self.idx += int(np.argmin(np.hypot(win[:, 0] - x, win[:, 1] - y)))
        if math.hypot(pts[-1, 0] - x, pts[-1, 1] - y) < GOAL_TOL:
            return "done", 0.0, 0.0
        # 2. look-ahead point: LOOKAHEAD metres further ALONG THE PATH (not straight-line)
        acc, la = 0.0, pts[-1]
        for i in range(self.idx, len(pts) - 1):
            seg = math.hypot(*(pts[i + 1] - pts[i]))
            if acc + seg >= LOOKAHEAD:
                a = (LOOKAHEAD - acc) / max(seg, 1e-9)
                la = pts[i] + a * (pts[i + 1] - pts[i])
                break
            acc += seg
        self.last_la = tuple(la)
        # 3. into the robot frame (robot at origin, heading = +x)
        dx, dy = la[0] - x, la[1] - y
        xl = math.cos(th) * dx + math.sin(th) * dy
        yl = -math.sin(th) * dx + math.cos(th) * dy
        heading_err = math.atan2(yl, xl)
        if abs(heading_err) > 1.0:                          # look-ahead behind/side: turn in place first
            w = float(np.clip(2.0 * heading_err, -W_MAX, W_MAX))
            if arc_collides(pose, 0.0, w, local, 0.3):
                return "blocked", 0.0, 0.0
            return "ok", 0.0, w
        # 4. curvature and 5. wheel speeds
        kappa = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)
        v = V_MAX / (1.0 + 1.5 * abs(kappa))
        # slow down near obstacles (local costmap) and near the goal
        near = local.at(x + 0.25 * math.cos(th), y + 0.25 * math.sin(th))
        v *= 1.0 - 0.6 * min(float(near), 252.0) / 252.0
        v = min(v, 0.3 + math.hypot(pts[-1, 0] - x, pts[-1, 1] - y))
        for scale in (1.0, 0.5, 0.25):
            vv = v * scale
            if not arc_collides(pose, vv, vv * kappa, local):
                return "ok", vv, vv * kappa
        return "blocked", 0.0, 0.0


# =====================================================================================
# 9. Decision making (FSM) + main loop
# =====================================================================================
class Mission:
    def __init__(self):
        self.odom = Odometry()
        self.grid = OccupancyGrid()
        self.field = DistanceField()
        self.pf = ParticleFilter(START)
        self.pose = np.array(START, np.float64)
        self.moved = np.zeros(2)                   # motion since the last correction (for the prior)
        self.sure_free = np.zeros((NY, NX), bool)
        self.local = LocalCostmap()
        self.gcost = np.full((NY, NX), COST_UNKNOWN, np.uint8)
        self.follower = PathFollower()
        self.yolo = Yolo()
        self.sem = SemanticMap()
        self.cam_seen = np.zeros((NY, NX), bool)
        self.state, self.prev_state = "INIT_SCAN", None
        self.turned = 0.0
        self.goal = None
        self.banned = []
        self.found = []
        self.false_targets = []
        self.target = None
        self.target_hits = 0
        self.target_t = -1e9
        self.goal_t = -1e9
        self.last_plan = -1e9
        self.blocked_since = None
        self.recover_until = 0.0
        self.recover_w = 1.0
        self.watch_t, self.watch_xy = 0.0, (START[0], START[1])
        self.look_spots = [(START[0], START[1])]
        self.approach_fails = 0
        self.plan_fails = 0
        self.step_i = 0
        self.loc_quality = 0.0
        self.det = None
        self.rgb = None
        self.mask = None
        log(f"mission: find {TARGET_COUNT}x {TARGET_NAME} -> DEST {DEST} -> START {START[:2]}")

    # ---------------- perception / SLAM ----------------
    def localize_and_map(self, ranges, ds, dth):
        # --- pose prediction (every step): odometry / gyro ---
        self.pf.predict(ds, dth)
        mid = self.pose[2] + dth / 2
        self.pose += (ds * math.cos(mid), ds * math.sin(mid), dth)
        self.pose[2] = wrap(self.pose[2])
        self.moved += np.array([abs(ds), abs(dth)])
        if self.step_i % LOC_EVERY:
            return
        # --- pose update (5-10 Hz): particle filter -> ICP -> scan-to-map optimisation ---
        pts = scan_points(ranges)
        dense = scan_points(ranges, SCAN_BEAMS)
        pred = self.pose.copy()
        if self.field.n_occ > 200:
            pts, dense = self.static_points(pts, pred), self.static_points(dense, pred)
        if self.field.n_occ > 200 and len(pts) > 20 and len(dense) > 40:
            self.pf.update(pts, self.field)
            est = self.pf.estimate()
            icp_pose, icp_err, n = icp(pts, est, self.field)
            start = icp_pose if (n > 25 and icp_err < 0.08 and np.hypot(*(icp_pose[:2] - est[:2])) < 0.25
                                 and abs(wrap(icp_pose[2] - est[2])) < 0.2) else est
            sig = (PRIOR_XY[0] + PRIOR_XY[1] * self.moved[0], PRIOR_TH[0] + PRIOR_TH[1] * self.moved[1])
            opt, cost = scan_to_map_optimize(dense, start, self.field, pred, sig)
            if cost <= scan_cost(dense, pred, self.field, pred, sig):
                self.pose = opt
            else:
                self.pose = pred
            self.pf.shift(np.array([self.pose[0] - est[0], self.pose[1] - est[1], wrap(self.pose[2] - est[2])]))
            P = transform(dense, self.pose)
            self.loc_quality = float(np.median(self.field.sample(P[:, 0], P[:, 1])))
        else:
            self.pose = self.pf.estimate()
        self.moved[:] = 0
        # --- mapping with the corrected pose, then refresh the "expected scan" distance field ---
        self.grid.update(self.pose, ranges)
        self.field.rebuild(self.grid.occupied())
        self.sure_free = cv2.erode((self.grid.logodds < 2 * L_FREE_TH).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def static_points(self, pts, pose):
        """Dynamic obstacle filter: a beam that ends in space the map knows is free (with a
        one-cell margin) hit something that moved there - e.g. the walking person. Such points
        are ignored for localization (they are still used for mapping and the local costmap)."""
        P = transform(pts, pose)
        rr, cc = world_to_rc(P[:, 0], P[:, 1])
        rr, cc = np.round(rr).astype(int), np.round(cc).astype(int)
        ok = (rr >= 0) & (rr < NY) & (cc >= 0) & (cc < NX)
        dyn = np.zeros(len(pts), bool)
        dyn[ok] = self.sure_free[rr[ok], cc[ok]]
        return pts[~dyn]

    def update_camera_coverage(self, ranges):
        """Floor cells inside the camera cone (occluded by LiDAR hits) count as searched."""
        x, y, th = self.pose
        k = int(math.degrees(CAM_HALF_FOV) * NBEAM / 360) - 2
        idx = np.arange(NBEAM // 2 - k, NBEAM // 2 + k + 1)
        r = ranges[idx]
        r = np.where(np.isfinite(r), np.minimum(r, 3.5), 3.5)
        ang = th + BEAM_ANG[idx]
        s = np.arange(0.15, 3.5, RES)[None, :]
        m = s < r[:, None]
        px = (x + s * np.cos(ang)[:, None])[m]
        py = (y + s * np.sin(ang)[:, None])[m]
        rr, cc = ((py - Y0) / RES).astype(int), ((px - X0) / RES).astype(int)
        ok = (rr >= 0) & (rr < NY) & (cc >= 0) & (cc < NX)
        self.cam_seen[rr[ok], cc[ok]] = True

    def perceive(self, ranges):
        self.rgb = camera_rgb()
        boxes = self.yolo.boxes
        if self.yolo.model is not None and self.step_i % YOLO_EVERY == 0:
            boxes = self.yolo.run(self.rgb)
            if self.state in ("INIT_SCAN", "EXPLORE", "LOOK_AROUND"):
                self.sem.add(boxes, ranges, self.pose)
        det, self.mask = detect_red_apple(self.rgb, boxes)
        self.det = None
        if det is None or det["dist"] > 4.0:
            return
        x, y, th = self.pose
        tx = x + det["dist"] * math.cos(th + det["bearing"])
        ty = y + det["dist"] * math.sin(th + det["bearing"])
        if any(math.hypot(tx - fx, ty - fy) < 1.0 for fx, fy in self.found) or \
                any(math.hypot(tx - fx, ty - fy) < 0.5 for fx, fy in self.false_targets):
            return                                            # already rescued / known false alarm
        # an apple lies on the floor: the LiDAR plane passes above it, so the cell must not be a wall
        r, c = to_cell(tx, ty)
        if in_map(r, c) and self.grid.logodds[r, c] > 2.0 and det["dist"] > 1.0:
            return
        self.det = det
        t = robot.getTime()
        stale = self.state != "APPROACH" and t - self.target_t > 3.0
        if self.target is None or stale or math.hypot(tx - self.target[0], ty - self.target[1]) > 0.8:
            if self.state != "APPROACH":
                self.target, self.target_hits, self.target_t = (tx, ty), 1, t
            return
        self.target_t = t
        a = 0.3                                               # running average of the estimate
        self.target = ((1 - a) * self.target[0] + a * tx, (1 - a) * self.target[1] + a * ty)
        self.target_hits += 1
        if self.target_hits >= 3 and self.state in ("INIT_SCAN", "EXPLORE", "LOOK_AROUND"):
            log(f"[target] {TARGET_NAME} spotted at ({self.target[0]:.2f}, {self.target[1]:.2f})"
                f" conf={det['conf']:.2f}")
            self.set_state("APPROACH")

    # ---------------- helpers ----------------
    def set_state(self, s):
        if s != self.state:
            log(f"[fsm] {self.state} -> {s}")
        self.prev_state, self.state = self.state, s
        self.follower.set([])
        self.last_plan = -1e9
        self.turned = 0.0
        self.blocked_since = None

    def plan_to(self, goal, allow_unknown):
        cost = self.local.stamp_into(self.gcost.copy())
        path = astar(cost, self.pose[:2], goal, allow_unknown=allow_unknown)
        self.last_plan = robot.getTime()
        self.follower.set(path)
        return path is not None

    def pick_frontier(self):
        cands = frontier_candidates(self.grid, self.gcost, self.cam_seen, self.pose, self.banned)
        best = choose_frontier(cands, self.sem)
        if best:
            log(f"[explore] {len(cands)} candidates -> {best['kind']} ({best['xy'][0]:.2f},{best['xy'][1]:.2f})"
                f" dist={best['dist']:.1f} size={best['size']} sem={self.sem.likelihood(*best['xy']):.2f}")
        return best["xy"] if best else None

    def spin(self, dth, w=1.2):
        self.turned += abs(dth)
        drive(0.0, w)
        return self.turned >= 2 * math.pi

    def goal_tol(self, xy):
        """A goal inside an obstacle / its inscribed zone (e.g. DEST is a sign) cannot be reached
        by the robot centre: getting within half a metre is arriving."""
        r, c = to_cell(*xy)
        blocked = in_map(r, c) and COST_INSCRIBED <= self.gcost[r, c] <= COST_LETHAL
        return 0.5 if blocked else GOAL_TOL

    def reached(self, xy, tol):
        return math.hypot(xy[0] - self.pose[0], xy[1] - self.pose[1]) < tol

    # ---------------- one control step ----------------
    def step(self):
        t = robot.getTime()
        self.step_i += 1
        ds, dth = self.odom.step()
        ranges = np.array(lidar.getRangeImage(), dtype=np.float32)
        self.localize_and_map(ranges, ds, dth)
        self.local.update(self.pose, ranges)
        if self.step_i % 5 == 0:
            self.gcost = global_costmap(self.grid)
        self.update_camera_coverage(ranges)
        self.perceive(ranges)

        if t < self.recover_until:                            # recovery: back off / turn, only where it is safe
            for v, w in ((-0.06, self.recover_w), (0.0, self.recover_w), (0.0, -self.recover_w)):
                if not arc_collides(self.pose, v, w, self.local, 0.5):
                    drive(v, w)
                    return
            drive(0, 0)
            return
        s = self.state
        if s == "INIT_SCAN" or s == "LOOK_AROUND":
            if self.spin(dth):
                self.set_state("EXPLORE")
            return
        if s == "DONE":
            drive(0, 0)
            return
        if s == "EXPLORE" and t > MAX_SEARCH_TIME:
            log(f"[fsm] search time over ({len(self.found)}/{TARGET_COUNT} found) -> destination")
            self.set_state("TO_DEST")
            return

        # ----- goal for the current state -----
        if s == "APPROACH":
            self.goal = self.target
            if self.reached(self.goal, REACH_DIST):
                self.target_reached()
                return
        elif s == "TO_DEST":
            self.goal = DEST
            if self.reached(DEST, self.goal_tol(DEST)):
                self.dest_reached()
                return
        elif s == "HOME":
            self.goal = START[:2]
            if self.reached(START[:2], self.goal_tol(START[:2])):
                self.home_reached()
                return

        # ----- (re)plan the global path every 2 s: map changes and people are taken into account -----
        if not self.follower.path or t - self.last_plan > 2.0:
            if s == "EXPLORE" and (self.goal is None or t - self.goal_t > 15.0 or not self.goal_useful(self.goal)):
                self.goal = self.pick_frontier()
                self.goal_t = t
                if self.goal is None:
                    self.no_frontier()
                    return
            if not self.plan_to(self.goal, allow_unknown=(s != "EXPLORE")):
                log(f"[plan] no path to ({self.goal[0]:.2f}, {self.goal[1]:.2f}) in {s}")
                drive(0, 0)
                self.goal_failed(t)
                return

        # ----- follow the global path with the look-ahead controller -----
        status, v, w = self.follower.step(self.pose, self.local)
        if status == "done":
            drive(0, 0)
            self.path_end_reached()
            return
        if status == "blocked":
            drive(0, 0)
            self.watch_t, self.watch_xy = t, (self.pose[0], self.pose[1])
            if self.blocked_since is None:
                self.blocked_since = t
            elif t - self.blocked_since > 3.0:               # a person may pass: wait, then replan around it
                log("[local] path blocked -> recovery + replan around the obstacle")
                self.blocked_since = None
                self.start_recovery(t)
            return
        self.blocked_since = None
        drive(v, w)

        # ----- progress watchdog -----
        if t - self.watch_t > 6.0:
            if math.hypot(self.pose[0] - self.watch_xy[0], self.pose[1] - self.watch_xy[1]) < 0.10:
                log(f"[watchdog] no progress in {s} -> recovery")
                self.start_recovery(t)
                self.goal_failed(t)
            self.watch_t, self.watch_xy = t, (self.pose[0], self.pose[1])

    # ---------------- FSM events ----------------
    def goal_useful(self, goal):
        """An exploration goal stays useful while unknown or camera-unseen floor is around it."""
        r, c = to_cell(*goal)
        k = 4
        win = (slice(max(0, r - k), r + k + 1), slice(max(0, c - k), c + k + 1))
        return bool((~self.grid.observed[win]).any() or (self.grid.free()[win] & ~self.cam_seen[win]).any())

    def target_reached(self):
        self.found.append(self.target)
        log(f"[target] reached {TARGET_NAME} #{len(self.found)} at ({self.target[0]:.2f}, {self.target[1]:.2f})")
        self.target, self.target_hits, self.approach_fails = None, 0, 0
        drive(0, 0)
        self.set_state("TO_DEST" if len(self.found) >= TARGET_COUNT else "LOOK_AROUND")

    def dest_reached(self):
        log("[fsm] destination reached -> returning home")
        drive(0, 0)
        self.set_state("HOME")

    def home_reached(self):
        log(f"[fsm] MISSION COMPLETE: {len(self.found)} {TARGET_NAME}(s) found, back at START")
        drive(0, 0)
        self.set_state("DONE")

    def path_end_reached(self):
        s = self.state
        if s == "EXPLORE":
            self.goal = None
            if all(math.hypot(self.pose[0] - lx, self.pose[1] - ly) > LOOK_SPACING for lx, ly in self.look_spots):
                self.look_spots.append((self.pose[0], self.pose[1]))
                self.set_state("LOOK_AROUND")
            return
        # the goal lies inside the inflation (apple next to furniture, sign on the wall):
        # the closest safe spot is good enough
        if s == "APPROACH" and self.reached(self.goal, REACH_DIST + 0.3):
            self.target_reached()
        elif s == "TO_DEST" and self.reached(self.goal, 0.6):
            self.dest_reached()
        elif s == "HOME" and self.reached(self.goal, 0.6):
            self.home_reached()
        else:
            self.follower.set([])                          # replan next step

    def no_frontier(self):
        self.plan_fails += 1
        if self.plan_fails >= 3:
            log("[explore] nothing left to explore -> destination")
            self.set_state("TO_DEST")
            return
        log("[explore] no frontier left: clear bans, search the explored area again")
        self.banned.clear()
        if self.plan_fails == 2:
            self.cam_seen[:] = False                       # look at everything once more
        self.set_state("LOOK_AROUND")

    def goal_failed(self, t):
        if self.state == "EXPLORE" and self.goal is not None:
            self.banned.append(self.goal)
            self.goal = None
        elif self.state == "APPROACH":
            self.approach_fails += 1
            if self.approach_fails > 3:
                self.abandon_target()

    def abandon_target(self):
        log("[target] unreachable / false detection -> back to exploring")
        self.false_targets.append(self.target)
        self.target, self.target_hits, self.approach_fails = None, 0, 0
        self.set_state("EXPLORE")

    def start_recovery(self, t):
        x, y, th = self.pose
        left = self.local.at(x + 0.3 * math.cos(th + 1.2), y + 0.3 * math.sin(th + 1.2))
        right = self.local.at(x + 0.3 * math.cos(th - 1.2), y + 0.3 * math.sin(th - 1.2))
        self.recover_w = 0.8 if left <= right else -0.8
        self.recover_until = t + 1.2
        self.follower.set([])
        self.last_plan = -1e9

    # ---------------- visualisation ----------------
    def render(self):
        if not SHOW_WINDOWS or self.rgb is None:
            return
        try:
            cam = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
            for x1, y1, x2, y2, cls, name, c in self.yolo.boxes:
                cv2.rectangle(cam, (x1, y1), (x2, y2), (200, 200, 0), 1)
                cv2.putText(cam, f"{name} {c:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (200, 200, 0), 1)
            if self.det:
                (cx, cy), rad = self.det["circle"]
                cv2.circle(cam, (cx, cy), rad, (255, 0, 0), 2)
                cv2.circle(cam, (cx, cy), 4, (255, 0, 255), -1)
                cv2.putText(cam, f"{TARGET_NAME} {self.det['dist']:.2f}m conf {self.det['conf']:.2f}",
                            (cx - rad, max(12, cy - rad - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(cam, f"{self.state}  found {len(self.found)}/{TARGET_COUNT}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("camera", cam)
            cv2.imshow("binary mask", self.mask)
            cv2.imshow("global costmap", self.map_image())
            lc = cv2.applyColorMap(self.local.cost, cv2.COLORMAP_JET)
            cv2.imshow("local costmap", cv2.flip(cv2.resize(lc, (LN * 4, LN * 4), interpolation=cv2.INTER_NEAREST), 0))
            cv2.waitKey(1)
        except cv2.error as e:
            log("[gui] disabled:", e)
            globals()["SHOW_WINDOWS"] = False

    def map_image(self, size=640):
        c = self.gcost
        img = np.zeros((NY, NX, 3), np.uint8)
        img[c == COST_FREE] = (255, 255, 255)
        infl = (c > 0) & (c < COST_INSCRIBED)
        v = c[infl].astype(np.float32) / 252.0
        img[infl] = np.stack([255 - 55 * v, 255 - 255 * v, 255 - 55 * v], axis=1).astype(np.uint8)   # magenta ramp
        img[c == COST_INSCRIBED] = (255, 0, 255)
        img[c == COST_LETHAL] = (0, 0, 0)
        img[c == COST_UNKNOWN] = (128, 128, 128)
        seen = self.cam_seen & (c == COST_FREE)
        img[seen] = (210, 245, 210)
        rows, cols = np.nonzero(self.grid.observed)
        if rows.size:
            r0, r1, c0, c1 = rows.min() - 10, rows.max() + 10, cols.min() - 10, cols.max() + 10
        else:
            r0, r1, c0, c1 = 0, NY, 0, NX
        for px, py in (START[:2], DEST):
            r, cc = to_cell(px, py)
            r0, r1, c0, c1 = min(r0, r - 10), max(r1, r + 10), min(c0, cc - 10), max(c1, cc + 10)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(NY, r1), min(NX, c1)

        def pt(wx, wy):
            r, cc = to_cell(wx, wy)
            return cc - c0, r - r0
        crop = img[r0:r1, c0:c1].copy()
        if self.follower.path:
            cv2.polylines(crop, [np.int32([pt(*p) for p in self.follower.path])], False, (255, 0, 0), 1)
        for p in self.pf.p[::3]:
            cv2.circle(crop, pt(p[0], p[1]), 0, (0, 200, 255), -1)
        for n, ox, oy, h in self.sem.objects:
            if h >= 2:
                cv2.circle(crop, pt(ox, oy), 2, (0, 160, 255), -1)
        cv2.rectangle(crop, np.subtract(pt(*START[:2]), 3), np.add(pt(*START[:2]), 3), (200, 0, 200), -1)
        cv2.rectangle(crop, np.subtract(pt(*DEST), 3), np.add(pt(*DEST), 3), (0, 160, 0), -1)
        for fx, fy in self.found:
            cv2.circle(crop, pt(fx, fy), 4, (0, 0, 255), 2)
        if self.target:
            cv2.drawMarker(crop, pt(*self.target), (0, 0, 255), cv2.MARKER_DIAMOND, 8, 2)
        if self.goal and self.state == "EXPLORE":
            cv2.drawMarker(crop, pt(*self.goal), (0, 140, 255), cv2.MARKER_CROSS, 8, 2)
        p = pt(self.pose[0], self.pose[1])
        cv2.circle(crop, p, 3, (0, 128, 255), -1)
        cv2.line(crop, p, (int(p[0] + 8 * math.cos(self.pose[2])), int(p[1] + 8 * math.sin(self.pose[2]))),
                 (0, 128, 255), 2)
        crop = cv2.flip(crop, 0)                                  # +y up
        k = size / max(crop.shape[:2])
        return cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)), interpolation=cv2.INTER_NEAREST)


def main():
    mission = Mission()
    last_dbg = 0.0
    while robot.step(DT_MS) != -1:
        tick = time.time()
        mission.step()
        mission.render()
        t = robot.getTime()
        if t - last_dbg >= 5.0:
            last_dbg = t
            x, y, th = mission.pose
            log(f"[dbg] {mission.state} pose=({x:.2f},{y:.2f},{math.degrees(th):.0f}deg) ess={mission.pf.ess:.0f}"
                f" fit={mission.loc_quality:.3f} found={len(mission.found)} cpu={1000 * (time.time() - tick):.0f}ms")
        if mission.state == "DONE" and os.environ.get("TB3_EXIT_ON_DONE"):
            break


if __name__ == "__main__":
    main()
