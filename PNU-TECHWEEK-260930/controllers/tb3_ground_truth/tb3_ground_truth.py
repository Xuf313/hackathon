import math
from controller import Supervisor, Keyboard

robot = Supervisor()

timestep = int(robot.getBasicTimeStep())

display = robot.getDevice("display")
display.setFont("Arial", 40, True)

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


# --- Ground Truth 확인용 세팅 ---
robot_node = robot.getSelf()


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


    # --- Ground Truth Pose 데이터 읽기 ---
    position = robot_node.getPosition()
    orientation = robot_node.getOrientation()

    x = position[0]
    y = position[1]

    theta = math.atan2(orientation[3],orientation[0])
    theta_deg = math.degrees(theta) + 180


    # --- Ground Truth Pose 데이터 확인 ---
    if curr_time - prev_time >= 0.1:
        display.setColor(0x000000)
        display.fillRectangle(0, 0, display.getWidth(), display.getHeight())

        display.setColor(0xFFFFFF)
        display.drawText("====== Ground Truth Position ======", 40, 40)
        display.drawText(f"x :  {x:.4f} m", 40, 100)
        display.drawText(f"y  :  {y:.4f} m", 40, 160)
        display.drawText("==== Ground Truth Orientation =====", 40, 300)
        display.drawText(f"\u03b8 :  {theta:.4f} rad", 40, 360)
        display.drawText(f"\u03b8 :  {theta_deg:.4f}\u00b0", 40, 420)

        prev_time = curr_time