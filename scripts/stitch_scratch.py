import os, glob
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

# list of (traj, buf) filename pairs
runs = []

path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", "*stdev*")
print(path)

folders = glob.glob(path)
for folder in folders:
    traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
    for traj_file in traj_files:
        buf_file = traj_file.replace("traj", "bufs")
        runs.append( (traj_file, buf_file) )

#runs = runs[:10]

do_sample = []
dt_sample = []

for r1 in range(len(runs)-1):
    _, buf_file1 = runs[r1]
    with open(buf_file1, "rb") as f:
        (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

    pos1 = buffers["position"]
    dur1 = np.array(elapsed)

    for r2 in range(r1+1, len(runs)):
        print(f" up to {r1},{r2} of {len(runs)}")
        _, buf_file2 = runs[r2]
        with open(buf_file2, "rb") as f:
            (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')
        pos2 = buffers["position"]
        dur2 = np.array(elapsed)

        dists = np.fabs(pos1[:,None,:] - pos2[None,:,:]).mean(axis=2).flatten()
        dts = (dur1[:,None] - dur2[None,:]).flatten()

        sample_idx = np.random.choice(range(len(dists)), size=3, replace=False)
        #sample_idx = np.arange(len(dists))#[np.random.rand(len(dists)) < .05]
        do_sample.append(dists[sample_idx])
        dt_sample.append(dts[sample_idx])

        #break
    #break

do_sample = np.concatenate(do_sample)
dt_sample = np.concatenate(dt_sample)

print(f"{len(do_sample)} points")

pt.plot(dt_sample, do_sample, 'k.')
pt.xlabel("dt")
pt.ylabel("do")
pt.yscale("log")
pt.show()



