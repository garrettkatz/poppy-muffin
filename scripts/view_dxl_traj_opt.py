import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

with open("dxl_opt_traj_result.pkl", "rb") as f:
    (motor_names, time_elapsed, positions, targets) = pk.load(f, encoding='latin1')

idx = motor_names.index("l_ankle_y")
positions = positions[:,idx]
targets = targets[:,idx]

pt.subplot(1,2,1)
pt.plot(time_elapsed, targets, 'r--')
pt.plot(time_elapsed, positions, 'k-')
pt.legend(("position", "target"))

pt.subplot(1,2,2)
pt.plot(time_elapsed, (targets - positions), 'k-')
pt.ylabel("error")

pt.show()

