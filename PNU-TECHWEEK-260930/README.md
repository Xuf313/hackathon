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

### 3. Run
Open **`worlds/apartment_sar.wbt`** and press play. Three windows appear: the robot camera, the labelled map with its legend, and the local costmap. Progress is logged to `controllers/tb3_sar/sar.log`.

Mission settings are at the top of `controllers/tb3_sar/tb3_sar.py`:
- `TARGET_COUNT` and `DEST`
- `SEMANTIC` and `SHOW_ALL_OBJECTS`
- `YOLO_WEIGHTS`, looked up in `models/YOLO/`
