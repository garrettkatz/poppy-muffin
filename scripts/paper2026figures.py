import os, glob
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import stitch_preanalysis as sp

## open-loop dataset divergence from success average

run_filepaths = sp.get_run_filepaths()
num_interp = 200
total_duration = 13.5 # (1.25 + 4*0.25) * 6
timepoints = np.linspace(0, total_duration, num_interp)

positions = np.empty((len(run_filepaths), num_interp, 25))
last_steps = np.empty(len(run_filepaths), dtype=int)

for r, (_, buf_file, _, numsteps) in enumerate(run_filepaths):

    last_steps[r] = numsteps

    # load the data
    with open(buf_file, "rb") as f:
        (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

    # interpolate positions
    for j in range(25):
        positions[r,:,j] = np.interp(timepoints, elapsed, buffers["position"][:,j])

    # save the initial target (same for all runs)
    # initial target is the *last* waypoint of every cycle (and the command send *before the first* waypoint of every cycle)
    init_target = buffers["target"][-1]

# average success run
mean_run = positions[last_steps==6].mean(axis=0)

# distances to mean run
distances = np.linalg.norm(positions - mean_run, axis=-1)

# plot
pt.figure(figsize=(8,4))

pt.subplot(1,2,1)
for d, dist in enumerate(distances):
    if last_steps[d] == 6:
        pt.plot(timepoints, dist, '-', color='b', alpha=.05)
    else:
        prefall = (timepoints <= last_steps[d] * 2.25)
        pt.plot(timepoints[prefall], dist[prefall], '-', color='r', alpha=.05)
        distances[d][~prefall] = np.nan

pt.plot(timepoints, distances[last_steps==6].mean(axis=0), 'b-', label='success')
pt.plot(timepoints, np.nanmean(distances[last_steps<6], axis=0), 'r-', label='fall')
pt.legend()
pt.xlabel("Time (sec)")
pt.ylabel("Distance (deg)")
pt.title("Success Distance")

pt.subplot(1,2,2)

# cycle boundaries
idx = np.linspace(0, num_interp-1, 4).astype(int)
returns = positions[:,idx,:]
deviance = np.linalg.norm(returns - init_target, axis=-1)
print(returns.shape, init_target.shape, deviance.shape)

for r, ls in enumerate(last_steps):
    deviance[r, ls//2+1:] = np.nan
    pt.plot(deviance[r], '-', marker=('o' if ls==6 else 'x'), color=('b' if ls==6 else 'r'), alpha=.05)

pt.plot(np.nanmean(deviance[last_steps==6],axis=0), 'b-', label='success')
pt.plot(np.nanmean(deviance[last_steps!=6],axis=0), 'r-', label='fall')

pt.legend()
pt.xlabel("Footstep Cycle")
pt.xticks(range(4), range(4))
#pt.ylabel("Distance (deg)")
pt.title("Return Distance")

pt.tight_layout()
pt.savefig("open_loop_observations.pdf")
pt.show()