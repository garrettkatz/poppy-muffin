import numpy as np
import matplotlib.pyplot as pt
from cosine_trajectory import make_cosine_trajectory
import pickle as pk

with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)

bump_trajs = []

waypoints = [(0., trajs[0][0][1])]
total_dur = 0.
for traj in trajs:
    bump_traj = []
    for i, (dur,ang) in enumerate(traj):
        # print(i,dur,ang['l_hip_x'], ang['r_hip_x'])
        total_dur += dur

        bump_traj.append( (dur, ang) )
        if i in [2,3]:
            # bump_traj[-1][1]["l_hip_x"] *= 5.
            bump_traj[-1][1]["l_hip_x"] += 2
            bump_traj[-1][1]["r_hip_x"] -= 2

    bump_trajs.append(tuple(bump_traj))
    waypoints.append( (total_dur, ang) )
    print(waypoints[-1])

all_joint_names = list(waypoints[0][1].keys())
timepoints, trajectory = make_cosine_trajectory(waypoints, all_joint_names, fps=5.)

for name in all_joint_names:
    pt.plot(timepoints, [a[name] for (_,a) in trajectory])
pt.show()

with open('pypot_traj1_smoothed.pkl', "wb") as f: pk.dump(trajectory, f, protocol=2)

with open('pypot_traj1_bumped.pkl', "wb") as f: pk.dump(bump_trajs, f, protocol=2)

