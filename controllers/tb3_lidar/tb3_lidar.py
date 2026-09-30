from controller import Robot
from datetime import datetime

robot = Robot()

timestep = int(robot.getBasicTimeStep())

lidar = robot.getDevice("LDS-01")
lidar.enable(100)  # lidar.enable(timestep)
lidar.enablePointCloud()

prev_time = 0

while robot.step(timestep) != -1:
    curr_time = robot.getTime()
    
    ranges = lidar.getRangeImage()
    
    if curr_time - prev_time >= 1.0:
        print("===========================")
        print("Number of LiDAR points:", len(ranges))
        print(f"Front:  {ranges[180]:.2f}")
        print(f"Back :  {ranges[0]:.2f}")
        print(f"Left :  {ranges[90]:.2f}")
        print(f"Right:  {ranges[270]:.2f}")
        
        # with open("lidar_log.txt", "a") as f:
            # f.write(
                # "===========================\n"
                # f"{datetime.now()}\n"
                # f"Number of LiDAR points: {len(ranges)}\n"
                # f"Front:  {ranges[180]:.2f}\n"
                # f"Back :  {ranges[0]:.2f}\n"
                # f"Left :  {ranges[90]:.2f}\n"
                # f"Right:  {ranges[270]:.2f}\n"
            # )
        
        prev_time = curr_time
