import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
import glob

Kp = 0.1 # "proportional gain" for ILC update

ilc_results = glob.glob(f"ilc_dxl_opt_traj_result_*.pkl")
ilc_itr = len(ilc_results)
print(ilc_results)

with open("dxl_opt_traj_result.pkl", "rb") as f:
    (motor_names, time_elapsed, positions, targets) = pk.load(f, encoding='latin1')

# get end points of each target change
endpoints = np.flatnonzero((targets[:-1] != targets[1:]).any(axis=1))

# get errors
errors = targets[endpoints] - positions[endpoints]

# create new trajectory
new_waypoints = targets[endpoints] + Kp * errors
durations = np.diff(time_elapsed[endpoints], prepend=1.)
trajectory = []
for duration, new_waypoint in zip(durations, new_waypoints):
    angle_dict = {name: new_waypoint[n] for n, name in enumerate(motor_names)}
    trajectory.append((duration, angle_dict))

# remap a few joints for some reason
for (_, angles) in trajectory:
    for name in ("r_shoulder_x", "r_shoulder_y"):
        angles[name] += 90.
    for name in ("l_shoulder_x", "l_shoulder_y"):
        angles[name] -= 90.

# save current iteration results
with open(f"ilc_dxl_opt_traj_result_{ilc_itr}.pkl", "wb") as f:
    pk.dump((motor_names, time_elapsed, positions, targets), f)
with open(f"poppy_opt_traj_ilc_{ilc_itr}.pkl", "wb") as f: pk.dump(trajectory, f, protocol=2)
with open("poppy_opt_traj_ilc.pkl", "wb") as f: pk.dump(trajectory, f, protocol=2)

# load all iteration results
all_results = {}
for past_itr in range(ilc_itr+1):
    with open(f"ilc_dxl_opt_traj_result_{past_itr}.pkl", "rb") as f:
        all_results[past_itr] = pk.load(f)

for past_itr, result in all_results.items():

    (motor_names, time_elapsed, positions, targets) = result
    endpoints = np.flatnonzero((targets[:-1] != targets[1:]).any(axis=1))

    idx = motor_names.index("l_ankle_y")
    
    pt.subplot(2, len(all_results), past_itr+1)
    # pt.plot(time_elapsed, targets, 'r-')
    pt.plot(time_elapsed[endpoints], targets[endpoints, idx], 'ro-', label='targets')
    if past_itr == ilc_itr:
        pt.plot(time_elapsed[endpoints], new_waypoints[:, idx], 'b+-', label='new targets')
    pt.plot(time_elapsed, positions[:,idx], 'k-', label='positions')
    pt.legend()
    if past_itr == 0: pt.ylabel("angle (deg)")
    pt.title(f"ILC itr {past_itr}")
    
    pt.subplot(2, len(all_results), len(all_results) + past_itr + 1)
    pt.plot(time_elapsed[endpoints], (targets[endpoints] - positions[endpoints]), 'k-')
    if past_itr == 0: pt.ylabel("error (deg)")
    pt.xlabel("Time")

pt.show()

