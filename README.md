<br>

<h1 align="center">
  2026 부산대학교 TECH WEEK:<br>
  Autonomous Mobile Robot의 Search & Rescue Mission
</h1>

<p align="center"><b>Team Alien</b> · 쳇제이야 · 칫수뛔이 · 텟까웅산 · 손옐래나</p>

<br>

---

## Overview

A **TurtleBot3 Burger** in a **Webots** apartment that runs a search-and-rescue mission on its own:

1. **Search** an apartment it has never seen for **red apples** lying on the floor.
2. **Confirm** each apple up close, so decoys (green, purple and orange apples, a red can) don't count.
3. **Return** to the start point once **2 apples** are found.

The robot only uses its own sensors: **wheel encoders, compass, accelerometer, 360° LiDAR (LDS-01) and camera**. It has no map, no GPS and no ground truth; only its start pose is known. A **pedestrian** keeps walking through the rooms, and carpets and furniture are in the way.

Our work is the controller in [`controllers/tb3_sar/`](controllers/tb3_sar/):

- [`tb3_sar.py`](controllers/tb3_sar/tb3_sar.py): mission, mapping, localization, apple detection, planning and safety
- [`semantic.py`](controllers/tb3_sar/semantic.py): semantic exploration (where to search next, from the objects YOLO recognises)

Everything else in the repository (the other controllers, the worlds, the apple models and the YOLO weights) was provided by the organizers as a starter kit.

## Team

| Team | Members |
| --- | --- |
| **Alien** 👽 | 쳇제이야 · 칫수뛔이 · 텟까웅산 · 손옐래나 |

## How it works

### 1. Mission state machine

```
SCAN ──► EXPLORE ──► APPROACH ──► HOME ──► DONE
  ▲         │  ▲          │
  └─────────┘  └──────────┘
 new area:      false alarm, or
 look 360°      apples still missing
```

| State | What the robot does |
| --- | --- |
| `SCAN` | Spins 360° to map the surroundings and look around with the camera. Repeated in every new area, because the camera only sees 60° ahead. |
| `EXPLORE` | Drives to the most promising unexplored spot (frontier): A* path, pure pursuit and the DWA local planner. |
| `APPROACH` | An apple was seen 3 times: drive to it. It counts only after 2 clear close-up sightings (under 2 m); otherwise it's a false alarm. |
| `HOME` | 2 apples found: drive back to the start pose. |
| `DONE` | Mission complete, stand still. |

### 2. Localization and mapping

- **Odometry + compass:** wheel encoders measure distance and the compass gives heading. The compass sign and offset are calibrated against the encoders during the first spin.
- **LiDAR scan matching:** every 5 steps the robot tries small shifts of its position (±15 cm) and keeps the one where the scan best matches walls already on the map. This removes wheel-slip drift.
- **Occupancy grid:** a 40 × 40 m map centred on the start (the building size is unknown), 5 cm cells, updated from every LiDAR scan.
- **Camera coverage:** floor the camera has looked at is marked as *searched*. Floor next to furniture only counts when seen from close by, since objects tucked there are easy to miss.

### 3. Semantic exploration (`semantic.py`)

YOLO (`yolo11n.pt`, COCO) labels objects in the camera image, and the LiDAR range along each box's direction places them on the map. An object is only trusted after 3 sightings with a clear class vote.

Each frontier is scored, and the lowest score wins:

```
score = path_distance − 3.0 × Σ prior[class] × share × exp(−d / 1.5 m)
```

The prior encodes general knowledge of where fruit is kept: `dining table 1.0`, `bowl 0.9`, `orange 0.9`, `refrigerator 0.8`, `oven 0.7`, `chair 0.5`, `couch 0.2`, … People are never used as landmarks. Once an apple is rescued, objects around it are ignored, so it no longer attracts the search.

### 4. Red-apple detection

Red pixels (HSV), then every blob has to pass these checks:

| # | Check | Rejects |
| --- | --- | --- |
| 1 | Round: circularity ≥ 0.5, aspect 0.6–1.6 | cans on their side, furniture edges |
| 2 | Bottom edge below the horizon | things on tables or walls |
| 3 | Pixel size matches its distance on the floor (0.55–1.7×) | too big or too small for a 10 cm apple |
| 4 | Not touching the image edge | half-visible blobs |
| 5 | No red continuing above it | tall red objects (fire extinguisher, cabinets) |

Each found apple is stored at the median of its close-range sightings. A new sighting near a found apple is treated as the same apple; the allowed distance grows with how far away the sighting was.

### 5. Driving and safety

- **Global path:** A* on the map, with walls padded by the robot's size. Re-planned every 1.5 s.
- **Local planner (DWA):** tests 90 short trajectories 1.5 s ahead on a 3 × 3 m costmap rebuilt from the current scan only, so the walking pedestrian never leaves ghost obstacles.
- **Emergency stop:** something within 20 cm in front → stop, wait, back off (only if the LiDAR shows room behind) and re-plan.
- **Low obstacles:** the LiDAR scans about 15 cm above the floor, so a carpet edge or a step is invisible to it. The **accelerometer** catches them instead: a tilt over 6° for 0.3 s while driving forward makes the robot reverse until level and mark that spot as an obstacle for both planners.
- **Getting unstuck:** after 3 back-offs within 15 s, or with no room behind, the robot turns toward the most open direction and picks a new goal.
- **Arriving home:** within 0.25 m of the start, or within 0.6 m when a wall stops it getting closer.

### 6. On-screen views

The robot in `apartment.wbt` has three Display overlays in the 3D view:

- `camera_view`: camera image with the apple's bounding box (class, confidence, distance, bearing)
- `map`: the map with explored and searched floor, path, start, found apples and the robot
- `local_costmap`: the local costmap with the DWA trajectories and the chosen one

Press **`c`** in the 3D view to tint the map with the planning costmap.

## Testing

In our Webots runs the robot found and confirmed both red apples, with no false alarms from the decoys. Testing also found four problems, each of which led to a fix:

| Problem in testing | Fix |
| --- | --- |
| Fell over on a carpet edge | Tilt sensing with the accelerometer |
| Kept backing into a corner | Back-off limit, then turn toward open space |
| Counted a found apple twice | Close-range positions; rescued apples remembered |
| Kept reversing next to the start | Home reached within 0.6 m when a wall blocks |

## Repository structure

```
hackathon/
├── controllers/
│   ├── tb3_sar/                 # ★ our controller
│   │   ├── tb3_sar.py           #   mission, mapping, localization, detection, planning, safety
│   │   ├── semantic.py          #   object map + semantic frontier scoring
│   │   └── sar.log              #   log of the latest run (written by the controller)
│   └── (starter controllers provided by the organizers)
│       tb3_teleop, tb3_teleop_sensors, tb3_teleop_cam, tb3_teleop_yolo,
│       tb3_cam, tb3_lidar, tb3_segmentation, tb3_ground_truth
├── worlds/
│   ├── apartment.wbt            # mission world
│   └── breakroom_*.wbt, empty.wbt
├── protos/                      # Red / Green / Purple / Orange apple models
├── models/YOLO/yolo11n.pt       # YOLO11n (COCO) weights
└── presentation/                # our slides (PDF + HTML)
```

## Getting started

### Requirements

- [Webots **R2025a**](https://cyberbotics.com/)
- Python 3.10
- Python packages:

```bash
pip install numpy opencv-python ultralytics torch
```

### Run the mission

1. Open `worlds/apartment.wbt` in Webots.
2. Press **Play**. The robot runs the `tb3_sar` controller.
3. Follow the `[sar]` lines in the console or in `controllers/tb3_sar/sar.log`.

YOLO runs on Apple Silicon (`mps`) when available, otherwise on the CPU. Without YOLO the mission still runs: apple detection is colour based, and only the semantic hints are lost.

### Key settings (top of `tb3_sar.py`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `START` | `(-0.3, -7.5, π)` | start pose given by the organizers |
| `TARGET_COUNT` | `2` | go home after this many apples (`None`: search the whole map) |
| `SEMANTIC` | `True` | let recognised objects guide where to search |
| `TILT_MAX` | `6°` | tilt that counts as hitting a low obstacle |
| `SHOW_DEBUG` | `True` | camera / map / local-costmap views |
| `SAVE_DEBUG_FILES` | `False` | save snapshots and `sem_objects.json` next to the controller |

The count and an optional extra destination can also be set at run time through the robot's `controllerArgs`: `--count=N` and `--dest=x,y`.

## Vision: a guide robot for blind people

Almost everything this robot does is what a robotic guide dog needs:

| Our robot today | A guide robot |
| --- | --- |
| Explores a building with no prior map | Guides someone through unfamiliar buildings |
| Dodges a walking person in real time | Walks safely through crowds |
| Feels carpet edges by tilt | Warns about steps and curbs |
| Finds target objects with its camera | Finds doors, empty seats, dropped keys |
| Remembers the way back to the start | Leads its user home |

What it would still need: a voice and haptic handle to communicate with the person, a depth camera for stairs and curbs, planning for a person walking beside it, and outdoor navigation (GPS, crosswalks, traffic lights).

## Presentation

- [`presentation/Team_Alien_Presentation.pdf`](presentation/Team_Alien_Presentation.pdf): all slides, viewable on GitHub
- [`presentation/index.html`](presentation/index.html): open in a browser to present. Use → / ← or click to move, **N** for speaker notes, **F** for fullscreen.
