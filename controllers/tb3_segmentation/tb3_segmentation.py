from controller import Robot
import cv2
import numpy as np


robot = Robot()

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)


while robot.step(timestep) != -1:
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

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        path = f"frame.jpg"
        cv2.imwrite(path, frame_bgr)
        print(f"Saved: {path}")

    if key == ord("q"):
        break

cv2.destroyAllWindows()
