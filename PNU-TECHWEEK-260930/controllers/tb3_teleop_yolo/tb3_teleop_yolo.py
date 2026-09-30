from controller import Robot, Keyboard
import cv2
import numpy as np
from ultralytics import YOLO


robot = Robot()

timestep = int(robot.getBasicTimeStep())

# 카메라 활성화
camera = robot.getDevice("camera")
camera.enable(timestep)

# 키보드 활성화
keyboard = Keyboard()
keyboard.enable(timestep)

# 모터 가져오기
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# 속도 제어 모드로 변경
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# 초기 정지
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# 바퀴 회전 속도 [rad/s]
SPEED = 3.0

width = camera.getWidth()
height = camera.getHeight()

import os, torch
model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models/YOLO/yolo11n.pt"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU: ~3.5x faster
model.to(DEVICE)


while robot.step(timestep) != -1:
    key = keyboard.getKey()

    left_speed = 0.0
    right_speed = 0.0

    if key == ord("w") or key == ord("W"):
        left_speed = SPEED
        right_speed = SPEED

    elif key == ord("s") or key == ord("S"):
        left_speed = -SPEED
        right_speed = -SPEED

    elif key == ord("a") or key == ord("A"):
        left_speed = -SPEED
        right_speed = SPEED

    elif key == ord("d") or key == ord("D"):
        left_speed = SPEED
        right_speed = -SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    image_bytes = camera.getImage()
    
    frame_arr = np.frombuffer(image_bytes, np.uint8)
    frame_bgra = frame_arr.reshape((height, width, 4))
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # YOLO 탐지
    results = model.predict(
        source=frame_bgr,
        conf=0.1,     # Confidence Threshold
        iou=0.5,      # IoU Threshold
        classes=None,  # 32: sports ball, 47: apple, 49: orange
        device=DEVICE,
        verbose=False  # per-frame printing floods the Webots console
    )
    
    output_frame = results[0].plot()

    cv2.imshow("Webots Camera", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
