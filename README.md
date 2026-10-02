<br>

<h1 align="center">
  2026 부산대학교 TECH WEEK:<br>
  Autonomous Mobile Robot의 Search & Rescue Mission
</h1>

<p align="center"><b>Team Alien</b> · 쳇제이야 · 칫수뛔이 · 텟까웅산 · 손옐래나</p>

<br>

---

## Overview

A **TurtleBot3 Burger** in the **Webots** apartment world that runs a full search-and-rescue mission with no human input:

1. **Search** the whole apartment for **red apples** lying on the floor. Their number and positions are unknown.
2. **Confirm** each apple up close, so decoys (green, purple and orange apples, a red can) don't count.
3. **Return home** to the start pose at `(-0.3, -7.5)` once no reachable area is left unseen.

Only the start pose is known in advance, as the hackathon rules allow: no map, no target positions, no destination and no target count. If the organizers announce a destination or a target count, they can be passed in at runtime (see [Key settings](#key-settings-top-of-tb3_sarpy)).

The robot only uses its own sensors: **wheel encoders, compass, 360° LiDAR (LDS-01) and camera**. It has no GPS and no ground-truth pose, and a **pedestrian** keeps walking through the rooms.

The whole mission lives in one controller, [`controllers/tb3_sar/tb3_sar.py`](controllers/tb3_sar/tb3_sar.py). The semantic-exploration helper is in [`controllers/tb3_sar/semantic.py`](controllers/tb3_sar/semantic.py).

## Team

| Team | Members |
| --- | --- |
| **Alien** 👽 | 쳇제이야 · 칫수뛔이 · 텟까웅산 · 손옐래나 |

## Results (earlier run)

> ⚠️ This run used an **earlier version** that had a fixed destination `(-4.94, -7.33)` and a fixed count of 2 apples, which the rules don't allow. Both were removed afterwards; the current version still needs a new full run in Webots.

From [`controllers/tb3_sar/sar.log`](controllers/tb3_sar/sar.log):

| Event | Sim time |
| --- | --- |
| Start: 360° scan, compass calibration | 0 s |
| Red apple #1 reached and confirmed (65 close views) | 298.2 s |
| Red apple #2 reached and confirmed (173 close views) | 430.4 s |
| Fixed destination reached (removed since) | 479.5 s |
| Back home: **MISSION COMPLETE** | **502.5 s** |

- **2 / 2** red apples found, **0** false alarms
- Got stuck **2** times and recovered on its own both times
- LiDAR scan matching applied **247** pose corrections, **4.49 m** in total

> This run used `yolo11n.pt` (COCO), which is now the only model the code uses. The frontier choice came from the local semantic prior (Jev was off).

## How it works

### 1. Mission state machine

```
SCAN ──► EXPLORE ──► APPROACH        EXPLORE ──► (TO_DEST) ──► HOME ──► DONE
  ▲         │  ▲          │          search over:  only with
  └─────────┘  └──────────┘          nothing left  --dest
 new area:      false alarm, or      unseen
 look 360°      apple rescued
```

| State | What the robot does |
| --- | --- |
| `SCAN` | Spins 360° to build the first map and look around. Repeated in every new area at least 1.5 m from earlier look spots, because the camera only sees ahead. |
| `EXPLORE` | Frontier exploration: BFS to candidate frontiers, a semantic score, then A* and path following. The search is over when no reachable floor is left unseen (checked twice, retrying skipped goals in between), or when `--count` targets were found. |
| `APPROACH` | Target seen 3 times: plan to it until within 0.35 m. It needs at least 2 close-up confirmations under 2 m, or it is marked as a false alarm. |
| `TO_DEST` | Only if a destination was given with `--dest`: plan to it. Skipped otherwise. |
| `HOME` | Plan back to the start pose. |
| `DONE` | Stop. |

A **safety layer** runs on every step under all states. If anything is closer than 20 cm in the driving direction, or DWA finds no safe trajectory, the robot stops, waits, backs off and re-plans. It only reverses while the LiDAR shows nothing within 25 cm behind it. Objects too low for the LiDAR (about 15 cm up), such as a carpet edge or the first stair, are caught by the built-in **accelerometer**: if the robot tips more than 6° for 0.3 s while driving forward, it stops, reverses until level again and marks that spot as an obstacle for both planners. If it has to back off 3 times within 15 s, or there's no room behind it, it turns toward the most open direction and drops its current goal instead, so it can't reverse itself into a corner. A progress watchdog (no motion for 3.5 s) triggers the same recovery. Home (or a given destination) counts as reached within 0.25 m, or within 0.6 m when a wall or obstacle stops the robot getting closer, so it doesn't keep backing off next to the goal.

### 2. Localization and mapping

- **Odometry + compass.** Wheel encoders give the distance travelled and the compass gives heading. The compass sign and offset are **auto-calibrated against the encoders during the first spin**.
- **LiDAR scan matching.** Every 5 steps the robot tries x/y shifts of ±15 cm in 2.5 cm steps. It keeps the shift where the scan best matches the walls already in the map (a truncated distance field) and applies half of it. This removes wheel-slip drift.
- **Occupancy grid.** A 40 × 40 m log-odds grid with 5 cm cells, centred on the start pose (no prior knowledge of the building's size), updated from every second LiDAR beam. Obstacles are inflated by the robot radius plus 9 cm for planning.
- **Camera coverage map.** Floor cells inside the camera's view cone (within 3 m) are marked as *searched*. Floor next to furniture only counts when seen from closer than 1.8 m.

### 3. Semantic exploration (`semantic.py`)

YOLO labels objects in the camera image, and the LiDAR range along each box's bearing places them on the map. Each object collects class votes weighted by confidence. It is only trusted after **3 sightings** with a **≥ 60 %** vote share.

Each frontier is then scored, and the lowest score wins:

```
score = path_distance − 3.0 × Σ prior[class] × share × exp(−d / 1.5 m)
```

The priors encode general knowledge of where fruit is kept: `dining table 1.0`, `bowl 0.9`, `orange 0.9`, `refrigerator 0.8`, `oven 0.7`, `chair 0.5`, `couch 0.2`, … Every class name is one of the 80 COCO classes the provided `yolo11n.pt` model detects, and the map only keeps COCO's indoor categories. Nothing is taken from the competition world. People are never used as anchors.

When `TYPESAFE_API_KEY` is set, an optional **Jev (TypeSafe AI)** chooser picks among the top 6 frontiers. It runs in a background thread so the control loop never blocks, and falls back to the local score.

### 4. Red-apple detection

An HSV red mask, then every blob has to pass these checks:

| # | Check | Rejects |
| --- | --- | --- |
| 1 | Red hue band | other colours (green / purple / orange apples) |
| 2 | Circularity ≥ 0.5, aspect 0.6–1.6 | cans on their side, edges of furniture |
| 3 | Bottom edge below the horizon | things on tables or walls |
| 4 | Pixel size matches the ground-plane distance (0.55–1.7×) | too big / too small for a 10 cm apple |
| 5 | Not touching the image edge | half-visible blobs |
| 6 | No red continuing above it | tall red objects (fire extinguisher, cabinets) |

Detections are projected into the world and smoothed. Positions near known false targets or already-rescued apples are ignored.

### 5. Motion planning

- **Global:** A* on the inflated occupancy grid, with an extra cost near walls and in unknown cells. Re-plans every 1.5 s.
- **Path following:** pure pursuit with a 0.6 m look-ahead. If the path is behind the robot, it turns in place first.
- **Local:** a **DWA** planner tests 90 `(v, ω)` pairs (6 × 15), each simulated 1.5 s ahead on a **3 × 3 m local costmap rebuilt from the current scan only**, so the moving pedestrian never leaves ghost obstacles.

### 6. On-screen views

The robot in `apartment.wbt` has three Display devices, shown as overlays in the 3D view:

- `camera_view`: camera image with the target's bounding box (id, class, confidence, distance, bearing)
- `map`: global map with explored and searched floor, path, start, found apples and the robot
- `local_costmap`: local costmap with the DWA candidate trajectories and the chosen one

Press **`c`** in the 3D view to show the global costmap tint on the map.

## Repository structure

**Team Alien wrote `controllers/tb3_sar/`** (`tb3_sar.py` + `semantic.py`, about 1,200 lines). Everything else in this repository (the other controllers, the worlds, the apple PROTOs and the YOLO weights) was provided by the organizers as a starter kit.

```
hackathon/
├── controllers/
│   ├── tb3_sar/                 # ★ OUR WORK: autonomous search-and-rescue controller
│   │   ├── tb3_sar.py           #   mission FSM, mapping, localization, detection, planning
│   │   ├── semantic.py          #   YOLO object map + semantic frontier scoring (+ optional Jev)
│   │   └── sar.log              #   log of the final run
│   └── (starter controllers provided by the organizers, not our work)
│       tb3_teleop, tb3_teleop_sensors, tb3_teleop_cam, tb3_teleop_yolo,
│       tb3_cam, tb3_lidar, tb3_segmentation, tb3_ground_truth
├── worlds/
│   ├── apartment.wbt            # mission world (given; runs tb3_sar)
│   └── breakroom_*.wbt, empty.wbt  # given test worlds
├── protos/                      # given: Red / Green / Purple / Orange apple PROTOs
└── models/YOLO/yolo11n.pt       # given: YOLO11n (COCO) weights
```

## Getting started

### Requirements

- [Webots **R2025a**](https://cyberbotics.com/)
- Python 3.11
- Python packages:

```bash
pip install numpy opencv-python ultralytics torch
```

### Run the mission

1. Open `worlds/apartment.wbt` in Webots.
2. The `TurtleBot3Burger` already uses the `tb3_sar` controller. Press **Play**.
3. Watch the overlays, or the console / `controllers/tb3_sar/sar.log`, for `[sar]` events.

YOLO runs on Apple Silicon (`mps`) when available, otherwise on the CPU. If YOLO can't be loaded the mission still runs: the red-apple detector is colour based, and only the semantic map is lost.

### Key settings (top of `tb3_sar.py`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `START` | `(-0.3, -7.5, π)` | known start pose |
| `SEMANTIC` | `True` | semantic frontier scoring on / off |
| `YOLO_WEIGHTS` | `yolo11n.pt` | provided COCO model; feeds the semantic map only |
| `SHOW_DEBUG` | `True` | camera / map / local-costmap views |
| `SAVE_DEBUG_FILES` | `False` | save snapshots and `sem_objects.json` next to the controller |

Runtime options (add them to the robot's `controllerArgs` in the world file, only if the organizers give this information):

| Argument | Example | Effect |
| --- | --- | --- |
| `--dest=x,y` | `--dest=-4.94,-7.33` | after the search, drive to this destination before going home |
| `--count=N` | `--count=3` | as soon as N apples are confirmed, stop searching and go straight back to the start (default: `TARGET_COUNT = 2` in `tb3_sar.py`) |

Optional: set `TYPESAFE_API_KEY` (and install `typesafe_sdk`) to let Jev choose frontiers.

## Next steps

- Evaluate the **Jev** frontier chooser against the local prior.
- Compare the estimated pose with `tb3_ground_truth` to measure localization error.

## Presentation

Team Alien's slides are in [`presentation/`](presentation/):

- [`Team_Alien_Presentation.pdf`](presentation/Team_Alien_Presentation.pdf): all 13 slides, viewable on GitHub
- [`index.html`](presentation/index.html): open in a browser to present. Use → / ← or click to move, **N** for speaker notes, **F** for fullscreen.
