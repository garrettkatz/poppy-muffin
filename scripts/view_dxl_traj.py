import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

with open("dxl_traj.pkl", "rb") as f: (time_elapsed, positions, dup_targets) = pk.load(f, encoding='latin1')

pt.subplot(1,3,1)
pt.plot(time_elapsed, positions, 'k+-')
pt.plot(time_elapsed, dup_targets, 'r--')
pt.legend(("position", "target"))

with open("os_traj.pkl", "rb") as f: (time_elapsed, positions, dup_targets) = pk.load(f, encoding='latin1')

pt.subplot(1,3,2)
pt.plot(time_elapsed, positions, 'k-')
pt.plot(time_elapsed, dup_targets, 'r--')
pt.legend(("position", "target"))

with open("pw_traj.pkl", "rb") as f: (motor_names, bufs) = pk.load(f, encoding='latin1')

flags, buffers, elapsed = zip(*bufs)
actuals = np.concatenate([buffers[i]['position'] for i in range(len(buffers))], axis=0)
targets = np.concatenate([buffers[i]['target'] for i in range(len(buffers))], axis=0)
# accumulate elapsed time over multiple waypoints
for i in range(1, len(elapsed)):
    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
elapsed = np.concatenate(elapsed)
idx = motor_names.index("l_ankle_y")

pt.subplot(1,3,3)
pt.plot(elapsed, targets[:, idx], linestyle='--', color='r', label='Target')
pt.plot(elapsed, actuals[:, idx], linestyle='-', color='k', label='Actual')
pt.legend()
pt.show()
