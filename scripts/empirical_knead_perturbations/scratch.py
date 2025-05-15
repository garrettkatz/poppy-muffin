import os
import glob
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

num_success = {}

sessions = glob.glob("*stdev*")
for session in sessions:

    # get session info
    date_str, stdev_str = session.split("_stdev")
    stdev = float(stdev_str)
    num_success[stdev] = []
    print(date_str, stdev)

    # get success info
    bufs_files = glob.glob(os.path.join(session, "bufs*pkl"))
    for bufs_file in bufs_files:
        nums = int(bufs_file[-len(".pkl")-1])
        num_success[stdev].append(nums)

# Visualize success/fall vs stdev
stdevs, nums = zip(*num_success.items())
sorter = np.argsort(stdevs)
stdevs = [stdevs[s] for s in sorter]
nums = [nums[s] for s in sorter]
pt.hist(nums, bins = np.linspace(-.5, 6.5, 8), rwidth=.8, label=[f"stdev = {stdev:.3f}" for stdev in stdevs])
pt.legend()
pt.xlabel("Number of successful consecutive steps")
pt.ylabel("Frequency")
pt.show()

# with open(traj_name, "rb") as f:
#     trajectory = pickle.load(f, encoding='latin1')

"""
Goals:
Avoid falls
Less load/temperature on the motors
Straight line motion (a la optical flow or scaled cross-corr)
Closed-loop robustness (adjust next waypoint to counteract perturbations)

Theme: "Sample Efficient Reward Maximization"
One framework for these different objectives
constraint/multi-objective
not "RL", but quickly optimizes with limited data

Compare with RL baselines in simulation?
Another possible baseline: System ID followed by trajectory optimization seeded by hand trajectory
"""

# inspect energy related senses
# most load/voltage  on knees, hip, shoulder
# temperature always rises steadily but slowly during session
# actually highest temperature on head_y, maybe from drag on cables and/or cheaper motor
# does not necessarily seem worth optimizing; no extremes and maybe mostly determined by operating time
# unless you find significant difference across different perturbations
keys = ("voltage", "load", "temperature")
all_bufs = {}
for session in sessions:

    # get session info
    date_str, stdev_str = session.split("_stdev")
    stdev = float(stdev_str)
    all_bufs[stdev] = {key: {} for key in keys}
    print(date_str, stdev)

    # get success info
    bufs_files = glob.glob(os.path.join(session, "bufs*pkl"))
    for bufs_file in bufs_files:
        with open(bufs_file, "rb") as f:
            (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

        n = int(bufs_file[-len("_?.pkl")-2:-len("_?.pkl")])
        for key in keys:
            all_bufs[stdev][key][n] = buffers[key]

    for key in keys:
        bufs = all_bufs[stdev][key]
        all_bufs[stdev][key] = [bufs[n] for n in range(len(bufs))]

for stdev in stdevs:
    _, axs = pt.subplots(len(keys),1)
    for k, key in enumerate(keys):
        data = np.concatenate(all_bufs[stdev][key], axis=0)
        axs[k].plot(data, alpha=.5)
        axs[k].plot(data.mean(axis=1), 'k-')
        axs[k].set_ylabel(key)
        axs[k].set_title(f"stdev {stdev} max motor {all_motor_names[data.mean(axis=0).argmax()]}")
    axs[-1].set_xlabel("Session timestep")
    pt.show()
