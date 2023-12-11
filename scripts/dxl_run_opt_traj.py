"""
$ python dxl_run_opt_traj.py <traj file>.pkl
    pkl data should be [... (duration, angles) ...] in (seconds, rad)
    e.g. poppy_opt_traj.pkl
saves output data as
    dxl_opt_traj_result.pkl: (motor_names, time_elapsed, positions, targets)
        motor_names[j]: jth motor name (across lower and upper)
        time_elapsed[n]: elapsed time at nth timepoint
        positions[n]: joint angle array at nth timepoint
        targets[n]: target angle array at nth timepoint
"""
import sys
import pickle as pk
import numpy as np

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

# load planned trajectory
traj_file = sys.argv[1]
with open(traj_file, "rb") as f:
    trajectory = pk.load(f)

# setup motor configuration
lower_names, lower_ids = zip(*(
    # legs
    ("l_hip_x", 11),
    ("l_hip_z", 12),
    ("l_hip_y", 13),
    ("l_knee_y", 14),
    ("l_ankle_y", 15),
    ("r_hip_x", 21),
    ("r_hip_z", 22),
    ("r_hip_y", 23),
    ("r_knee_y", 24),
    ("r_ankle_y", 25),
))
upper_names, upper_ids = zip(*(
    # trunk
    ("abs_y", 31),
    ("abs_x", 32),
    ("abs_z", 33),
    ("bust_y", 34),
    ("bust_x", 35),
    ("head_z", 36),
    ("head_y", 37),
    # arms
    ("l_shoulder_y", 41),
    ("l_shoulder_x", 42),
    ("l_arm_z", 43),
    ("l_elbow_y", 44),
    ("r_shoulder_y", 51),
    ("r_shoulder_x", 52),
    ("r_arm_z", 53),
    ("r_elbow_y", 54),
))

all_motor_names = lower_names + upper_names
name_index = {name: i for i, name in enumerate(all_motor_names)}

kp = 2. # PID P gain

# remap joint angles based on direct and offset
# https://github.com/poppy-project/poppy-humanoid/blob/master/software/poppy_humanoid/configuration/poppy_humanoid.json#L144
# https://github.com/poppy-project/pypot/blob/master/pypot/dynamixel/motor.py#L56
for (_, angles) in trajectory:
    angles['r_shoulder_x'] -= +90.
    angles['r_shoulder_y'] -= +90.
    angles['l_shoulder_x'] -= -90.
    angles['l_shoulder_y'] -= -90. # +90 but indirect?
    angles['head_y'] -= 20.
    angles['l_hip_y'] -= 2.

# reformat trajectory
_, init_angles = trajectory[0]
init_waypoint = np.array([init_angles[name] for i, name in enumerate(all_motor_names)])

timepoints = np.empty(len(trajectory)-1)
waypoints = np.empty((len(trajectory)-1, len(all_motor_names)))
for n, (duration, angles) in enumerate(trajectory[1:]): # skip init_angles
    timepoints[n] = duration
    for i, name in enumerate(all_motor_names):
        waypoints[n, i] = angles[name]
timepoints = timepoints.cumsum()

print("connecting...")
import pypot.dynamixel
import pypot.utils.pypot_time as time

ports = pypot.dynamixel.get_available_ports()
print(ports)

lower_dxl_io = pypot.dynamixel.DxlIO('/dev/ttyACM1')
upper_dxl_io = pypot.dynamixel.DxlIO('/dev/ttyACM0')

lower_upper = (
    (lower_dxl_io, lower_names, lower_ids),
    (upper_dxl_io, upper_names, upper_ids))

print("current angles:")
pos = np.array(
    lower_dxl_io.get_present_position(lower_ids) + \
    upper_dxl_io.get_present_position(upper_ids))
print({name: pos[i] for i, name in enumerate(all_motor_names)})

print('init angles:')
print(init_angles)

input('[Enter] to turn off compliance')

for dxl_io, _, ids in lower_upper:

    dxl_io.enable_torque(ids)
    dxl_io.set_pid_gain({motor_id: (kp, 0., 0.) for motor_id in ids})

input('[Enter] for init angles (may want to hold up by strap)')

for dxl_io, names, ids in lower_upper:

    dxl_io.set_moving_speed({motor_id: 10 for motor_id in ids})
    dxl_io.set_goal_position({
        motor_id: init_angles[motor_name]
        for motor_id, motor_name in zip(ids, names)})

time.sleep(2.)

input('[Enter] to run trajectory')


# init angles
time_elapsed = [0.]
positions = [pos]
targets = [init_waypoint]
n = 0
start_time = time.time()
while True:

    # record current state
    pos = np.array(
        lower_dxl_io.get_present_position(lower_ids) + \
        upper_dxl_io.get_present_position(upper_ids))

    positions.append(pos)
    targets.append(waypoints[n])
    time_elapsed.append(time.time() - start_time)

    # if (len(positions)-1) % 20 == 0:
    #     # print("itr %d: %.3f s: %.3f deg vs %.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], pos, waypoints[n], n))
    #     print("itr %d: %.3f s: error=%.3f deg (n=%d)" % (len(positions)-1, time_elapsed[-1], mid_errors[-1], n))

    # continue busy loop until current time interval over
    if time_elapsed[-1] < timepoints[n]: continue

    # when over, advance to next timepoint
    while n < len(timepoints) and timepoints[n] < time_elapsed[-1]: n += 1

    # stop if last waypoint finished
    if n == len(timepoints): break

    # otherwise update goal position
    for dxl_io, names, ids in lower_upper:
        dxl_io.set_goal_position({
            motor_id: waypoints[n, name_index[motor_name]]
            for motor_name, motor_id in zip(names, ids)})

# wait at end
time.sleep(1.)

time_elapsed = np.array(time_elapsed)
positions = np.stack(positions)
targets = np.stack(targets)

with open('dxl_opt_traj_result.pkl', "wb") as f:
    pk.dump((all_motor_names, time_elapsed, positions, targets), f)

# input('[Enter] to return to init (may want to hold up by strap)')

# for dxl_io, names, ids in lower_upper:

#     dxl_io.set_moving_speed({motor_id: 50 for motor_id in ids})
#     dxl_io.set_goal_position({motor_id: 0. for motor_id in ids})

# time.sleep(2.)

input('[Enter] to go compliant and close (may want to hold up by strap)')

for dxl_io, _, ids in lower_upper:
    dxl_io.disable_torque(ids)
    dxl_io.close()

print("closed.")


