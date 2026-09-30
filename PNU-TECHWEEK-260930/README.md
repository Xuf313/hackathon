<br>

<h1 align="center">
  2026 부산대학교 TECH WEEK:<br>
  Autonomous Mobile Robot의 Search & Rescue Mission
</h1>
<br>

---

<br><br>

![](부산대_TECHWEEK_Physical_AI.png)

---

## Team HKS: running the autonomous SAR robot (`tb3_sar`)

### 1. Requirements
- **Webots R2025a**. The first time a world opens, Webots downloads the standard objects from GitHub, so you need internet the first time only. After that they're cached.
- **Python 3.10+** with these packages:

```bash
pip install numpy opencv-python ultralytics torch
```

### 2. Tell Webots which Python to use
Use one of these two options:
- **Webots → Preferences → General → Python command**: set it to the Python where you installed the packages above (e.g. `/path/to/venv/bin/python3`).
- Or create `controllers/tb3_sar/runtime.ini` just for this controller:

```ini
[python]
COMMAND = /path/to/your/venv/bin/python3
```

`runtime.ini` files are git-ignored, so everyone keeps their own.

### 3. Build the YOLO-World model (once)
The robot recognises furniture with **YOLO-World**, an open-vocabulary detector given our apartment vocabulary. The model file is too large for git (328 MB), so build it once:

```bash
python models/YOLO/make_world_model.py
```

This downloads `yolov8s-worldv2.pt` (26 MB) and the CLIP text encoder (338 MB, first time only), then writes `models/YOLO/yolo_world_apartment.pt`. If you skip this step, the robot falls back to `yolo11n.pt` (COCO) and says so in its log.

### 4. Run
Open **`worlds/apartment.wbt`** (the demo world; `apartment_sar.wbt` is an identical copy) and press play. Three windows appear:
- **camera**: bounding box on the target
- **map**: labelled map with legend. The global costmap overlay shows red for no-go and orange for costly; press `c` to toggle it. A dashed square marks the local costmap window.
- **local costmap**: the 3 m rolling window with the DWA trajectories

Progress is logged to `controllers/tb3_sar/sar.log`.

Mission settings are at the top of `controllers/tb3_sar/tb3_sar.py`:
- `TARGET_NAME`: `"red apple"` (2 apples) or `"football"`
- `DEST`: the destination coordinate (known in advance, like a base station)
- `SEMANTIC` and `SHOW_ALL_OBJECTS`
- `YOLO_WEIGHTS`, looked up in `models/YOLO/`

### 5. What the robot does
- **Localization:** wheel encoders for distance, compass for heading, and LiDAR scan matching against the map to correct drift. In our test the average error was 0.17 m, down from about 2.5 m.
- **Mapping:** 5 cm log-odds occupancy grid, plus a global costmap with inflation around obstacles.
- **Detection:**
  - Red apple: HSV colour plus size, roundness and floor checks. It must be confirmed within 2 m before it counts.
  - Football: YOLO-World.
- **Exploration:**
  - Frontiers plus floor the camera hasn't checked yet. Corners next to furniture need a close look before they count as checked.
  - Semantic priors (an apple is likely near a table or fridge).
  - A 360° look-around in each new area.
- **Planning and control:** A* on the global costmap, look-ahead point, and a DWA local planner on a 3 m local costmap rebuilt from each scan.
- **Mission logic:** a state machine, SCAN → EXPLORE → APPROACH → TO_DEST → HOME → DONE.

Last full test: 2 red apples found (the bathroom corner and the dining room), destination reached, back home in 413 s of simulated time.
