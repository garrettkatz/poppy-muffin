"""
Workflow:
0. fresh start with rm -fr {traj_base}_ilc_*.pkl ilc_pw_{traj_base}_result_*.pkl
1. copy {traj_base}.pkl to {traj_base}_ilc.pkl on robot
2. execute {traj_base}_ilc.pkl on robot (see bottom of launch_poppy.py)
3. copy traj_buf.pkl to local
4. run this script
5. copy {traj_base}_ilc.pkl to robot
6. repeat 2-6 until satisfied

run this script
"""
import sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
import glob

traj_base = sys.argv[1]
# traj_base = 'opt_traj_med'

ilc_kp = .05 # "proportional gain" for ILC update

# every run of this script saves result_{itr}
# itr is the number of results saved so far
ilc_results = glob.glob(f"ilc_pw_{traj_base}_result_*.pkl")
ilc_itr = len(ilc_results)
print(ilc_results)

# get the most recent result, assumes traj buf for last iteration has been scp'd
with open("traj_buf.pkl", "rb") as f:
    (buffers, time_elapsed, motor_names) = pk.load(f, encoding='latin1')

# get the planned trajectory
with open(f'{traj_base}.pkl', 'rb') as f:
    trajectory = pk.load(f, encoding='latin1')
planned = np.array([[angs[name] for name in motor_names] for (_, angs) in trajectory])

# get end points of each target change
targets = buffers['target']
endpoints = np.flatnonzero((targets[:-1] != targets[1:]).any(axis=1))
endpoints = np.append(endpoints, len(time_elapsed)-1)

# get actual angles at endpoints
actuals = buffers['position']
actuals = actuals[endpoints]

# get errors
errors = planned - actuals

# create new trajectory
new_waypoints = planned + ilc_kp * errors
new_trajectory = []
for n, (duration, _) in enumerate(trajectory):
    angle_dict = {name: new_waypoints[n, idx] for idx, name in enumerate(motor_names)}
    new_trajectory.append((duration, angle_dict))

# save current iteration results
# assumes you will scp and execute traj_base_ilc.pkl to get next traj buf
with open(f"ilc_pw_{traj_base}_result_{ilc_itr}.pkl", "wb") as f:
    pk.dump((buffers, time_elapsed, motor_names), f)
with open(f"{traj_base}_ilc_{ilc_itr}.pkl", "wb") as f:
    pk.dump(new_trajectory, f, protocol=2)
with open(f"{traj_base}_ilc.pkl", "wb") as f:
    pk.dump(new_trajectory, f, protocol=2)

# load all iteration results for visualization
all_iters = {}
for past_itr in range(ilc_itr+1):
    with open(f"ilc_pw_{traj_base}_result_{past_itr}.pkl", "rb") as f:
        result = pk.load(f)
    # get the planned trajectory
    with open(f'{traj_base}_ilc_{past_itr}.pkl', 'rb') as f:
        trajectory = pk.load(f, encoding='latin1')
    corrected = np.array([[angs[name] for name in motor_names] for (_, angs) in trajectory])
    all_iters[past_itr] = (result, corrected)

# idx = [motor_names.index("l_ankle_y")]
idx = list(range(len(motor_names))) # all

for past_itr, (result, corrected) in all_iters.items():

    (buffers, time_elapsed, motor_names) = result

    targets = buffers['target']
    endpoints = np.flatnonzero((targets[:-1] != targets[1:]).any(axis=1))
    endpoints = np.append(endpoints, len(time_elapsed)-1)

    actuals = buffers['position'][endpoints]
    time_elapsed = np.array(time_elapsed)[endpoints]
    
    pt.subplot(2, len(all_iters), past_itr+1)
    pt.plot(time_elapsed, corrected[:, idx], 'g-')
    pt.plot(time_elapsed, planned[:, idx], 'b-')
    pt.plot(time_elapsed, actuals[:, idx], 'k-')
    # pt.legend()

    if past_itr == 0: pt.ylabel("angles (deg)")
    pt.title(f"ILC itr {past_itr}")
    
    pt.subplot(2, len(all_iters), len(all_iters) + past_itr + 1)
    pt.plot(time_elapsed, (actuals[:, idx] - planned[:, idx]), 'r-')
    if past_itr == 0: pt.ylabel("error (deg)")
    pt.xlabel("Time")

    print(f"{past_itr}: mean abs error = {np.fabs(actuals[:,idx] - planned[:,idx]).mean()}")

pt.show()

