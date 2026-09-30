from controller import Robot
import cv2
import numpy as np

robot = Robot()

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()

while robot.step(timestep) != -1:
    pass
    
#     image_bytes = camera.getImage()
    
#     frame_arr = np.frombuffer(image_bytes, np.uint8)
#     frame_bgra = frame_arr.reshape((height, width, 4))
#     frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

#     cv2.imshow("Webots Camera", frame_bgr)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("s"):
#         path = f"frame.jpg"
#         cv2.imwrite(path, frame_bgr)
#         print(f"Saved: {path}")

#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()