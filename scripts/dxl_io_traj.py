### load trajectory

import pickle as pk
import numpy as np

with open("poppy_opt_traj.pkl","rb") as f: trajectory = pk.load(f)

timepoints = np.empty(len(trajectory)-1)
targets = np.empty(len(trajectory)-1)
for s, (duration, angles) in enumerate(trajectory[1:]): # skip (_, init_angles)
    timepoints[s] = duration
    targets[s] = angles["l_ankle_y"]    
timepoints = timepoints.cumsum()

# import matplotlib.pyplot as pt
# pt.plot(timepoints, targets)
# pt.show()

# command trajectory

import pypot.dynamixel
try:
    import pypot.utils.pypot_time as time
except:
    import time

ports = pypot.dynamixel.get_available_ports()
print(ports)

kp = 14. # PID proportional gain
motor_id = 15 # l_ankle_y
start_pos = trajectory[0][1]["l_ankle_y"]

lower_dxl_io = pypot.dynamixel.DxlIO('/dev/ttyACM1')
lower_dxl_io.enable_torque((motor_id,))
lower_dxl_io.set_pid_gain({motor_id: (kp, 0., 0.)})

lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

# follow trajectory
n = 0
start_time = time.time()
positions = []
errors = []
end_errors = []
time_elapsed = []
while True:
    pos = lower_dxl_io.get_present_position((motor_id,))[0]
    positions.append(pos)
    time_elapsed.append(time.time() - start_time)
    errors.append(pos - targets[n])
    # if (len(positions)-1) % 20 == 0:
    #     # print("itr %d: %.3f s: %.3f deg vs %.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], pos, targets[n], n))
    #     print("itr %d: %.3f s: error=%.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], errors[-1], n))

    if time_elapsed[-1] < timepoints[n]: continue
    end_errors.append(pos - targets[n])

    while n < len(timepoints) and timepoints[n] < time_elapsed[-1]: n += 1
    if n == len(timepoints): break
    lower_dxl_io.set_goal_position({motor_id: targets[n]})

# wait at end
time.sleep(1.)

# return to start
lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

print('pid', lower_dxl_io.get_pid_gain((motor_id,))[0])

lower_dxl_io.disable_torque((motor_id,))
lower_dxl_io.close()

print('mean absolute error:')
print(np.mean(np.fabs(errors)), np.mean(np.fabs(end_errors)))

print('max absolute error:')
print(np.fabs(errors).max(), np.fabs(end_errors).max())

