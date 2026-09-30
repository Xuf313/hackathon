"""Semantic exploration: COCO objects seen by YOLO -> object map -> "where is the apple likely?" prior.

Frontier choice = trade-off between travel cost and semantic likelihood:
    score = path_distance - W_SEM * sum_obj prior[class] * exp(-d(frontier, obj) / SIGMA)
Optionally a Jev (TypeSafe AI System One model) call picks among the top candidates.
"""
import math
import os
import threading

import numpy as np

# how strongly each COCO class suggests "an apple is nearby" (kitchen / dining context)
APPLE_PRIOR = {
    "dining table": 1.0, "bowl": 0.9, "refrigerator": 0.8, "oven": 0.7, "microwave": 0.6,
    "sink": 0.5, "cup": 0.5, "wine glass": 0.4, "bottle": 0.4, "chair": 0.5, "banana": 0.9,
    "orange": 0.9, "apple": 1.0, "couch": 0.2, "potted plant": 0.2, "tv": 0.1, "bed": 0.1,
}
SKIP = {"person"}                     # dynamic: never anchor semantics on the pedestrian
# only indoor-plausible COCO classes go on the map (yolo11n hallucinates "airplane", "traffic light"...
# on simulator renders, e.g. ceiling lamps)
INDOOR = {"chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "refrigerator", "oven",
          "microwave", "sink", "bowl", "cup", "bottle", "wine glass", "potted plant", "clock", "vase",
          "book", "apple", "orange", "banana", "sports ball", "teddy bear", "keyboard", "mouse"}
W_SEM = 3.0                           # metres of detour one fully-likely object is worth
SIGMA = 1.5                           # influence radius (m)
MERGE = 0.6                           # same class within this distance = same object


class SemanticMap:
    def __init__(self):
        self.objects = []             # [cls, x, y, hits]

    def add(self, boxes, ranges, pose, cam_w, focal, beam_ang_of_idx, nbeam):
        """Project YOLO boxes onto the floor map using the LiDAR range along the box bearing."""
        x, y, th = pose
        for x1, y1, x2, y2, name, conf in boxes:
            if name in SKIP or name not in INDOOR:
                continue
            # bearings of the box edges -> LiDAR beams covering the box
            b1 = math.atan2(cam_w / 2 - x1, focal)
            b2 = math.atan2(cam_w / 2 - x2, focal)
            idx = [int(round((math.pi - b) * nbeam / (2 * math.pi))) % nbeam
                   for b in np.linspace(b2, b1, 7)]
            r = ranges[idx]
            r = r[np.isfinite(r) & (r > 0.12)]
            if r.size == 0 or np.median(r) > 3.4:
                continue
            d = float(np.median(r))
            b = (b1 + b2) / 2
            ox, oy = x + d * math.cos(th + b), y + d * math.sin(th + b)
            for o in self.objects:
                if o[0] == name and math.hypot(o[1] - ox, o[2] - oy) < MERGE:
                    k = o[3]
                    o[1] = (o[1] * k + ox) / (k + 1); o[2] = (o[2] * k + oy) / (k + 1); o[3] = min(k + 1, 20)
                    break
            else:
                self.objects.append([name, ox, oy, 1])

    def likelihood(self, wx, wy):
        s = 0.0
        for name, ox, oy, hits in self.objects:
            if hits < 2:              # need two sightings before trusting an object
                continue
            s += APPLE_PRIOR.get(name, 0.0) * math.exp(-math.hypot(wx - ox, wy - oy) / SIGMA)
        return s

    def nearby(self, wx, wy, rad=2.5):
        out = [(n, math.hypot(wx - ox, wy - oy)) for n, ox, oy, h in self.objects
               if h >= 2 and math.hypot(wx - ox, wy - oy) < rad]
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
