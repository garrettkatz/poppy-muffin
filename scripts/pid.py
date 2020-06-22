from pypot.creatures import PoppyHumanoid as PH
import numpy as np
import time
import matplotlib.pyplot as plt

joint_names = ["l_shoulder_y","l_shoulder_x","l_arm_z","l_elbow_y"]
# joint_names = ["l_shoulder_x","l_arm_z","l_elbow_y"]
speed_limit = 30 # degrees per second

p = PH()

def get_position():
    pos = np.array([getattr(p,jn).present_position for jn in joint_names])
    # if pos[0] > -90: pos[0] -= 360    # deal with shoulder_y wierdness
    return pos

for jn in joint_names:
    getattr(p, jn).compliant = True

raw_input("ready to record target joints? (hang left over the edge)")

target = get_position()

print(target)

raw_input("ready to run controller?")

for jn in joint_names:
    getattr(p, jn).compliant = False

velocity = np.zeros(len(target))

positions = [get_position()]
errors = [target-positions[0]]
integrals = [np.zeros(len(target))]
derivatives = [np.zeros(len(target))]
outputs = [np.zeros(len(target))]

dt = 0.005
t = time.clock()

Kp, Ki, Kd, dur = .75, 0, 0, 12
Kp, Ki, Kd, dur = 1, 0, .5, 20
# Kp, Ki, Kd, dur = 1, 0, 1, 15
Kp, Ki, Kd, dur = 1, .1, .5, 20
# Kp, Ki, Kd, dur = .5, 0, 1, 10
# Kp, Ki, Kd, dur = .5, .1, 1, 20

wear_n_tear = True
# wear_n_tear = False

for n in range(int(dur / dt)):

    position = get_position()
    error = target - position # x_shoulder_y abruptly wraps around from + to -
    integral = integrals[-1] + error * dt
    derivative = (error - errors[-1]) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    
    if wear_n_tear: output -= 10

    velocity += output*dt

    time.sleep(max(dt - (time.clock()-t), 0))
    for i,jn in enumerate(joint_names):
        # safe_output = min(max(output[i], -speed_limit), speed_limit)
        # getattr(p,jn).goal_speed = safe_output
        safe_velocity = min(max(velocity[i], -speed_limit), speed_limit)
        getattr(p,jn).goal_speed = safe_velocity
    t = time.clock()

    positions.append(position)
    errors.append(error)
    integrals.append(integral)
    derivatives.append(derivative)
    outputs.append(output)

for i,jn in enumerate(joint_names):
    getattr(p,jn).goal_speed = 0
    getattr(p,jn).goal_position = getattr(p,jn).present_position

times = np.arange(len(outputs)) * dt

print("%d timesteps"%len(times))
print("final errors:")
print(errors[-1])

plt.subplot(5,1,1)
plt.plot(times, np.array(positions))
plt.ylabel("positions")
plt.subplot(5,1,2)
plt.plot(times, np.array(errors))
plt.ylabel("errors")
plt.subplot(5,1,3)
plt.plot(times, np.array(integrals))
plt.ylabel("integrals")
plt.subplot(5,1,4)
plt.plot(times, np.array(derivatives))
plt.ylabel("derivatives")
plt.subplot(5,1,5)
plt.plot(times, np.array(outputs))
plt.ylabel("outputs")
plt.xlabel("time")
plt.tight_layout()
plt.show()

raw_input("Ready to make left arm compliant and hang?")
for jn in joint_names:
    getattr(p, jn).compliant = True

raw_input("Ready to close poppy object?")
for jn in joint_names:
    getattr(p, jn).compliant = False
p.close()
print("Done.")


