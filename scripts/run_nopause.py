import sys
import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid
    from pypot.sensor import OpenCVCamera
    import pypot.utils.pypot_time as time
except:
    from mocks import PoppyHumanoid
    from mocks import OpenCVCamera
    import time

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

# mirror joints laterally (from poindexter/hand.py)
def get_mirror(angle_dict):
    mirrored = dict(angle_dict)
    for name, angle in angle_dict.items():

        # don't negate y-axis rotations
        sign = 1 if name[-2:] == "_y" else -1

        # swap right and left
        mirror_name = name
        if name[:2] == "l_": mirror_name = "r_" + name[2:]
        if name[:2] == "r_": mirror_name = "l_" + name[2:]

        # assign mirrored angle
        mirrored[mirror_name] = angle * sign

    return mirrored

with open('nopause_trajectories.pkl', 'rb') as f:
    init_angles, trajectories = pk.load(f)

# show what traj will look like
trajectory = trajectories["init","shift"]

import matplotlib.pyplot as pt
durs = [dur for (dur,_) in trajectory]
angs = [targ['r_knee_y'] for (_, targ) in trajectory]
pt.plot(np.cumsum(durs), angs, '.-', label='r_knee_y')
angs = [targ['l_knee_y'] for (_, targ) in trajectory]
pt.plot(np.cumsum(durs), angs, '.-', label='l_knee_y')
angs = [targ['abs_y'] for (_, targ) in trajectory]
pt.plot(np.cumsum(durs), angs, '.-', label='abs_y')
angs = [targ['r_shoulder_x'] for (_, targ) in trajectory]
pt.plot(np.cumsum(durs), angs, '.-', label='r_shoulder_x')
pt.legend()
pt.show()


poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    # OpenCVCamera("poppy-cam", 0, 10),
)

input("[Enter] to enable torques")
poppy.enable_torques()

input("[Enter] to goto init (suspend first)")
_ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

input("[Enter] to go through trajectory")
buffers, time_elapsed = poppy.track_trajectory(trajectory, overshoot=1.)

input("[Enter] to go compliant (hold strap first)")
poppy.disable_torques()

fall = input("Fall direction: Combo of [f]ront or [b]ack and [l]eft or [r]ight, or [s]uccess: ")

print("don't forget poppy.close()")

