import os
import sys
import math
import matplotlib.pyplot as plt

sys.path.append('/usr/local/webots/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/usr/local/webots/'
from controller import Robot

robot = Robot()
time_step = int(robot.getBasicTimeStep())

# load devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
gps = robot.getDevice('gps')
gps.enable(time_step)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0)
right_motor.setVelocity(0)

# tracking
x_vals = []
y_vals = []

# run for 30 seconds
start_time = robot.getTime()
while robot.step(time_step) != -1:
    t = robot.getTime()
    if t - start_time > 30:
        break

    # case 1: constant velocity
    # left_motor.setVelocity(-1)
    # right_motor.setVelocity(1)

    # case 2: time-varying
    left_motor.setVelocity(math.sin(-t))
    right_motor.setVelocity(math.cos(2 * t))

    gps_vals = gps.getValues()
    x_vals.append(gps_vals[0])
    y_vals.append(gps_vals[1])

# stop motors
left_motor.setVelocity(0)
right_motor.setVelocity(0)

# plot
plt.figure()
plt.plot(x_vals, y_vals, label="Trajectory (30s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Robot Path (Case 2)")
plt.axis('equal')
plt.xlim(min(x_vals)-0.01, max(x_vals)+0.01)
plt.ylim(min(y_vals)-0.01, max(y_vals)+0.01)
plt.grid(True)
plt.legend()
plt.show()
