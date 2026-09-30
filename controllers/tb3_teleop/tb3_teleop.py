from controller import Robot, Keyboard

robot = Robot()

timestep = int(robot.getBasicTimeStep())

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
