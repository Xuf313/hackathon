"""Semantic exploration: COCO objects seen by YOLO -> object map -> "where is the apple likely?" prior.

Frontier choice = trade-off between travel cost and semantic likelihood:
    score = path_distance - W_SEM * sum_obj prior[class] * exp(-d(frontier, obj) / SIGMA)
Optionally a Jev (TypeSafe AI System One model) call picks among the top candidates.
"""
import math
import os
import threading

import numpy as np

# Every class name below is a COCO class, i.e. something the provided yolo11n.pt model can output.
# Nothing is taken from the competition world: the priors are general "where is fruit kept" knowledge.
APPLE_PRIOR = {   # how strongly each object suggests "an apple is nearby" (kitchen / dining context)
    "dining table": 1.0, "bowl": 0.9, "refrigerator": 0.8, "oven": 0.7, "microwave": 0.5, "sink": 0.5,
    "chair": 0.5, "wine glass": 0.4, "bottle": 0.4, "apple": 1.0, "orange": 0.9, "banana": 0.8,
    "couch": 0.2, "potted plant": 0.2,
}
SKIP = {"person"}                     # dynamic: never anchor semantics on the pedestrian
INDOOR = {                            # classes allowed on the map: COCO's indoor super-categories
    # furniture
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    # electronic
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    # appliance
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # kitchen
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    # food
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    # indoor
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
W_SEM = 3.0                           # metres of detour one fully-likely object is worth
SIGMA = 1.5                           # influence radius (m)
MERGE = 0.6                           # detections within this distance = same object
SEM_CONF = 0.35                       # ignore weak YOLO boxes for mapping
SEM_RANGE = 2.5                       # only map objects closer than this (far boxes are unreliable)
MIN_SEEN = 3                          # sightings before an object is trusted
MIN_SHARE = 0.6                       # winning class must hold >= 60% of the (confidence-weighted) votes


class SemanticMap:
    """Object map with class voting: every detection at a spot votes for its class (weighted by
    confidence); an object is only shown/used once it was seen often enough and the vote is clear."""

    def __init__(self, prior=None):
        self.objects = []             # dicts: x, y, n (sightings), votes {cls: conf_sum}
        self.prior = prior if prior is not None else APPLE_PRIOR
        self.ignore = []              # (x, y, r) zones around rescued targets: no objects mapped there

    def forget_near(self, wx, wy, rad):
        """Drop objects near (wx, wy) and ignore new ones there (e.g. a rescued apple YOLO keeps seeing)."""
        self.objects = [o for o in self.objects if math.hypot(o["x"] - wx, o["y"] - wy) >= rad]
        self.ignore.append((wx, wy, rad))

    def add(self, boxes, ranges, pose, cam_w, focal, beam_ang_of_idx, nbeam):
        """Project YOLO boxes onto the floor map using the LiDAR range along the box bearing."""
        x, y, th = pose
        for x1, y1, x2, y2, name, conf in boxes:
            if name in SKIP or name not in INDOOR or conf < SEM_CONF:
                continue
            # bearings of the box edges -> LiDAR beams covering the box
            b1 = math.atan2(cam_w / 2 - x1, focal)
            b2 = math.atan2(cam_w / 2 - x2, focal)
            idx = [int(round((math.pi - b) * nbeam / (2 * math.pi))) % nbeam
                   for b in np.linspace(b2, b1, 7)]
            r = ranges[idx]
            r = r[np.isfinite(r) & (r > 0.12)]
            if r.size == 0 or np.median(r) > SEM_RANGE:
                continue
            d = float(np.median(r))
            b = (b1 + b2) / 2
            ox, oy = x + d * math.cos(th + b), y + d * math.sin(th + b)
            if any(math.hypot(ox - ix, oy - iy) < ir for ix, iy, ir in self.ignore):
                continue
            # merge with any object at this spot (whatever its class) -> class votes compete
            near = [o for o in self.objects if math.hypot(o["x"] - ox, o["y"] - oy) < MERGE]
            if near:
                o = min(near, key=lambda o: math.hypot(o["x"] - ox, o["y"] - oy))
                k = min(o["n"], 20)
                o["x"] = (o["x"] * k + ox) / (k + 1); o["y"] = (o["y"] * k + oy) / (k + 1)
                o["n"] += 1
                o["votes"][name] = o["votes"].get(name, 0.0) + conf
            else:
                self.objects.append({"x": ox, "y": oy, "n": 1, "votes": {name: conf}})

    @staticmethod
    def label(o):
        cls = max(o["votes"], key=o["votes"].get)
        share = o["votes"][cls] / sum(o["votes"].values())
        return cls, share

    def reliable(self):
        """[(cls, x, y, n, share)] for objects that passed the voting thresholds."""
        out = []
        for o in self.objects:
            cls, share = self.label(o)
            if o["n"] >= MIN_SEEN and share >= MIN_SHARE:
                out.append((cls, o["x"], o["y"], o["n"], share))
        return out

    def likelihood(self, wx, wy):
        s = 0.0
        for cls, ox, oy, n, share in self.reliable():
            s += self.prior.get(cls, 0.0) * share * math.exp(-math.hypot(wx - ox, wy - oy) / SIGMA)
        return s

    def nearby(self, wx, wy, rad=2.5):
        out = [(c, math.hypot(wx - ox, wy - oy)) for c, ox, oy, n, sh in self.reliable()
               if math.hypot(wx - ox, wy - oy) < rad]
        return sorted(out, key=lambda t: t[1])[:5]


def pick(cands, sem):
    """cands: [(wx, wy, path_dist)] -> best (wx, wy) by distance-vs-semantics trade-off."""
    best, best_s = None, 1e18
    for wx, wy, dist in cands:
        s = dist - W_SEM * sem.likelihood(wx, wy)
        if s < best_s:
            best, best_s = (wx, wy), s
    return best


# ---------------- optional: Jev (TypeSafe AI) as the high-level chooser ----------------
class JevChooser:
    """Asks Jev to choose among the top-K frontier candidates. Runs in a background thread
    (70-500 ms latency) so the control loop never blocks; falls back to `pick` when unavailable."""

    def __init__(self):
        self.client = None
        self.pending = None
        self.result = None
        if not os.environ.get("TYPESAFE_API_KEY"):
            return
        try:
            from typesafe_sdk import Choice, TypeSafeClient
            self.Choice = Choice
            self.client = TypeSafeClient()
        except Exception as e:
            print(f"[sar] Jev disabled: {e}")

    @property
    def enabled(self):
        return self.client is not None

    def request(self, cands, sem, target_name):
        if not self.enabled or (self.pending and self.pending.is_alive()):
            return
        top = sorted(cands, key=lambda c: c[2] - W_SEM * sem.likelihood(c[0], c[1]))[:6]
        lines, criteria = [], {}
        for i, (wx, wy, dist) in enumerate(top):
            near = ", ".join(f"{n} {d:.1f}m" for n, d in sem.nearby(wx, wy)) or "nothing recognised"
            criteria[f"f{i}"] = f"frontier {i}: {dist:.1f} m away; nearby: {near}"
            lines.append(criteria[f"f{i}"])
        state = (f"A small floor robot searches an apartment for a {target_name} lying on the floor. "
                 "Pick the unexplored frontier most worth visiting next (likely location vs travel cost).\n"
                 + "\n".join(lines))

        def work():
            try:
                resp = self.client.system_one(state=state, questions={
                    "frontier": self.Choice(instructions="Which frontier should the robot explore next?",
                                            criteria=criteria)})
                i = int(resp.answers["frontier"].choice[1:])
                self.result = (top[i][0], top[i][1], resp.answers["frontier"].confidence)
            except Exception as e:
                print(f"[sar] Jev call failed: {e}")
        self.pending = threading.Thread(target=work, daemon=True)
        self.pending.start()

    def take(self):
        r, self.result = self.result, None
        return r
