import math
from controller import Robot, Keyboard

robot = Robot()

timestep = int(robot.getBasicTimeStep())

display = robot.getDevice("display")
display.setFont("Arial", 24, True)

# --- 기존 teleop 코드 ---
keyboard = Keyboard()
keyboard.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

SPEED = 3.0


# --- 센서 추가 ---
# Sensor 1: LiDAR
lidar = robot.getDevice("LDS-01")
lidar.enable(100)  # lidar.enable(timestep)
lidar.enablePointCloud()

# Sensor 2: Wheel Encoder
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Sensor 3: Accelerometer
accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(timestep)

# Sensor 4: Gyroscope
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Sensor 5: Compass
compass = robot.getDevice("compass")
compass.enable(timestep)

# Sensor 6: Camera
camera = robot.getDevice("camera")
camera.enable(timestep)


prev_time = 0
while robot.step(timestep) != -1:
    # --- 기존 teleop 코드 ---
    curr_time = robot.getTime()
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

    # --- 센서 데이터 읽기 ---
    # LiDAR:
    ranges = lidar.getRangeImage()
    scan_num = len(ranges)

    # Wheel Encoder:
    left_position = left_encoder.getValue()
    right_position = right_encoder.getValue()

    # Accelerometer:
    acc = accelerometer.getValues()

    # Gyroscope:
    gyro_value = gyro.getValues()

    # Compass:
    compass_value = compass.getValues()
    theta = math.atan2(compass_value[1], compass_value[0])
    theta_deg = math.degrees(theta) + 180.

    # Camera:
    width = camera.getWidth()
    height = camera.getHeight()


    # --- 센서 데이터 확인 ---
    if curr_time - prev_time >= 0.1:
        # print(
        #     "\n===========LiDAR===========\n"
        #     f"Number of LiDAR points: {scan_num}\n"
        #     f"Front :  {ranges[180]:.2f} m\n"
        #     f"Back  :  {ranges[0]:.2f} m\n"
        #     f"Left  :  {ranges[90]:.2f} m\n"
        #     f"Right :  {ranges[270]:.2f} m\n"
        #     "\n==========Encoder==========\n"
        #     f"Left  :  {left_position:.4f} rad\n"
        #     f"Right :  {right_position:.4f} rad\n"
        #     "\n=======Accelerometer=======\n"
        #     f"x :  {acc[0]:.4f} m/s^2\n"
        #     f"y :  {acc[1]:.4f} m/s^2\n"
        #     f"z :  {acc[2]:.4f} m/s^2\n"
        #     "\n=========Gyroscope=========\n"
        #     f"x :  {gyro_value[0]:.4f} rad/s\n"
        #     f"y :  {gyro_value[1]:.4f} rad/s\n"
        #     f"z :  {gyro_value[2]:.4f} rad/s\n"
        #     "\n==========Compass==========\n"
        #     f"x :  {compass_value[0]:.4f}\n"
        #     f"y :  {compass_value[1]:.4f}\n"
        #     f"z :  {compass_value[2]:.4f}\n"
        #     f"\u03b8 :  {theta_deg:.2f}\u00b0\n"
        #     "\n==========Camera===========\n"
        #     f"width  :  {width} px\n"
        #     f"height :  {height} px\n"
        #     "\n***************************\n"
        # )

        # display.setColor(0x000000)
        # display.fillRectangle(0, 0, display.getWidth(), display.getHeight())

        # display.setColor(0xFFFFFF)
        # display.drawText("===========LiDAR===========", 40, 20)
        # display.drawText(f"Number of LiDAR points: {scan_num}", 40, 60)
        # display.drawText(f"Front :  {ranges[180]:.2f} m", 40, 100)
        # display.drawText(f"Back  :  {ranges[0]:.2f} m", 40, 140)
        # display.drawText(f"Left  :  {ranges[90]:.2f} m", 40, 180)
        # display.drawText(f"Right :  {ranges[270]:.2f} m", 40, 220)
        # display.drawText("==========Encoder==========", 40, 320)
        # display.drawText(f"Left  :  {left_position:.4f} rad", 40, 360)
        # display.drawText(f"Right :  {right_position:.4f} rad", 40, 400)
        # display.drawText("=======Accelerometer=======", 40, 500)
        # display.drawText(f"x :  {acc[0]:.4f} m/s^2", 40, 540)
        # display.drawText(f"y :  {acc[1]:.4f} m/s^2", 40, 580)
        # display.drawText(f"z :  {acc[2]:.4f} m/s^2", 40, 620)
        # display.drawText("=========Gyroscope=========", 40, 720)
        # display.drawText(f"x :  {gyro_value[0]:.4f} rad/s", 40, 760)
        # display.drawText(f"y :  {gyro_value[1]:.4f} rad/s", 40, 800)
        # display.drawText(f"z :  {gyro_value[2]:.4f} rad/s", 40, 840)
        # display.drawText("==========Compass==========", 40, 940)
        # display.drawText(f"x :  {compass_value[0]:.4f}", 40, 980)
        # display.drawText(f"y :  {compass_value[1]:.4f}", 40, 1020)
        # display.drawText(f"z :  {compass_value[2]:.4f}", 40, 1060)
        # display.drawText(f"\u03b8 :  {theta_deg:.2f}\u00b0", 40, 1100)
        # display.drawText("==========Camera===========", 40, 1200)
        # display.drawText(f"width  :  {width} px", 40, 1240)
        # display.drawText(f"height :  {height} px", 40, 1280)

        prev_time = curr_time