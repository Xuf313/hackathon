from controller import Robot, Keyboard
import cv2
import numpy as np


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

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)


while robot.step(timestep) != -1:
    # 1. Keyboard Teleop

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


    # 2. Camera Object Detection

    image_bytes = camera.getImage()
    
    frame_arr = np.frombuffer(image_bytes, np.uint8)
    frame_bgra = frame_arr.reshape((height, width, 4))
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    
    blr = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, green_lower, green_upper)
    
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contour_lst) > 0:
        contour = max(contour_lst, key=cv2.contourArea)
        _, radius = cv2.minEnclosingCircle(contour)
        
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        cv2.circle(frame_bgr, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame_bgr, center, 5, (255, 0, 0), -1)

    cv2.imshow("Webots Camera", frame_bgr)
    # cv2.imshow("LAB Mask", mask)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
