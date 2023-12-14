# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

### load trajectory

import pickle as pk
import numpy as np

# with open("poppy_opt_traj.pkl","rb") as f: trajectory = pk.load(f)

trajectory = [
    (0.5, {'l_ankle_y': 30.0}), # init angles
    (0.5, {'l_ankle_y': 15.0}), (0.5, {'l_ankle_y': 0}), (0.5, {'l_ankle_y': -15.0}), (0.5, {'l_ankle_y': -30.0})]

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

kp = 4. # PID proportional gain
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
dup_targets = []
mid_errors = []
end_errors = []
time_elapsed = []
for n in range(len(timepoints)):

    # skip intermediate waypoints that are behind schedule
    time_passed = time.time() - start_time
    if n + 1 != len(trajectory) and time_passed >= timepoints[n]: continue

    # otherwise set goal for current waypoint
    lower_dxl_io.set_goal_position({motor_id: targets[n]})

    # wait until reached
    while True:

        # record position
        pos = lower_dxl_io.get_present_position((motor_id,))[0]
        positions.append(pos)
        dup_targets.append(targets[n])
        mid_errors.append(pos - targets[n])
        time_elapsed.append(time.time() - start_time)
        # if (len(positions)-1) % 20 == 0:
        #     # print("itr %d: %.3f s: %.3f deg vs %.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], pos, targets[n], n))
        #     print("itr %d: %.3f s: error=%.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], mid_errors[-1], n))

        if time_elapsed[-1] > timepoints[n]: break

    end_errors.append(pos - targets[n])

# wait at end
time.sleep(1.)

# return to start
lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

print('mean absolute error (mid, end):')
print(np.mean(np.fabs(mid_errors)), np.mean(np.fabs(end_errors)))

print('max absolute error (mid, end):')
print(np.fabs(mid_errors).max(), np.fabs(end_errors).max())

with open("dxl_traj.pkl", "wb") as f: pk.dump((time_elapsed, positions, dup_targets), f)

print('pid', lower_dxl_io.get_pid_gain((motor_id,))[0])

input("continue with overshoot...")

positions = []
dup_targets = []
mid_errors = []
end_errors = []
time_elapsed = []
start_time = time.time()
for n in range(len(timepoints)):

    # skip intermediate waypoints that are behind schedule
    time_passed = time.time() - start_time
    if n + 1 != len(timepoints) and time_passed >= timepoints[n]: continue

    do_overshoot = True

    # only do overshoot before last waypoint
    if n + 1 == len(timepoints):
        do_overshoot = False

    # don't overshoot extremals
    else:
        pos = lower_dxl_io.get_present_position((motor_id,))[0]
        curdir = np.sign(targets[n] - pos)
        nexdir = np.sign(targets[n+1] - targets[n])
        if curdir != nexdir: do_overshoot = False

    # limit speed based on target duration
    distance = np.fabs(targets[n] - pos)
    duration = timepoints[n] - time_passed
    max_speed = distance / duration # units = degs / sec
    # max_speed = int(max_speed * 60. / 360. /.229) # units = .229 rotations / min
    max_speed = int(max_speed * 60. / 360. /.114) # units = .114 rotations / min
    max_speed = min(max_speed, 500) # don't go too fast

    # apply overshoot
    goal_pos = targets[n]
    if do_overshoot: goal_pos += 30 * curdir

    # send commands
    lower_dxl_io.set_moving_speed({motor_id: max_speed})
    lower_dxl_io.set_goal_position({motor_id: goal_pos})

    # wait until reached
    while True:

        # record position
        pos = lower_dxl_io.get_present_position((motor_id,))[0]
        positions.append(pos)
        dup_targets.append(targets[n])
        mid_errors.append(pos - targets[n])
        time_elapsed.append(time.time() - start_time)
        # if (len(positions)-1) % 20 == 0:
        #     # print("itr %d: %.3f s: %.3f deg vs %.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], pos, targets[n], n))
        #     print("itr %d: %.3f s: error=%.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], mid_errors[-1], n))

        if time_elapsed[-1] > timepoints[n]: break

    end_errors.append(pos - targets[n])

# # follow trajectory
# n = 0
# start_time = time.time()
# positions = []
# dup_targets = []
# mid_errors = []
# end_errors = []
# time_elapsed = []
# while True:
#     pos = lower_dxl_io.get_present_position((motor_id,))[0]
#     positions.append(pos)
#     dup_targets.append(targets[n])
#     mid_errors.append(pos - targets[n])
#     time_elapsed.append(time.time() - start_time)
#     # if (len(positions)-1) % 20 == 0:
#     #     # print("itr %d: %.3f s: %.3f deg vs %.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], pos, targets[n], n))
#     #     print("itr %d: %.3f s: error=%.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], mid_errors[-1], n))

#     if time_elapsed[-1] < timepoints[n]:
#         # cap speed to avg between current and next target
#         distance = targets[n] - pos
#         avg_speed = np.fabs(distance) / (timepoints[n] - time_elapsed[-1]) # deg / s
#         moving_speed = int(avg_speed * 60 * (1./360.) * (1./.229)) # .229 rotations / minute = 1 unit
#         lower_dxl_io.set_moving_speed({motor_id: moving_speed})
#         continue

#     # log final error when it should have arrived
#     end_errors.append(pos - targets[n])

#     # advance to next waypoint
#     while n < len(timepoints) and timepoints[n] <= time_elapsed[-1]: n += 1
#     if n == len(timepoints): break

#     # cap speed again
#     distance = targets[n] - pos
#     avg_speed = np.fabs(distance) / (timepoints[n] - time_elapsed[-1]) # deg / s
#     moving_speed = int(avg_speed * 60 * (1./360.) * (1./.229)) # .229 rotations / minute = 1 unit
#     lower_dxl_io.set_moving_speed({motor_id: moving_speed})

#     # overshoot to maintain velocity before final waypoint
#     if n + 1 < len(timepoints):
#         overshoot = pos + 2*distance
#     else:
#         overshoot = targets[n]

#     lower_dxl_io.set_goal_position({motor_id: overshoot})

# wait at end
time.sleep(1.)

# return to start
lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

print('mean absolute error (mid, end):')
print(np.mean(np.fabs(mid_errors)), np.mean(np.fabs(end_errors)))

print('max absolute error (mid, end):')
print(np.fabs(mid_errors).max(), np.fabs(end_errors).max())

with open("os_traj.pkl", "wb") as f: pk.dump((time_elapsed, positions, dup_targets), f)

lower_dxl_io.disable_torque((motor_id,))
lower_dxl_io.close()

