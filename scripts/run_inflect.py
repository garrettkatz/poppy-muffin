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

# whether to show plot (only when X server available)
show = False

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

with open('inflect_trajectory.pkl', 'rb') as f:
    trajectory = pk.load(f)

_, init_angles = trajectory[0]


# show what traj will look like
if show:
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

# PID tuning
K_p, K_i, K_d = 8.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

input("[Enter] to enable torques")
poppy.enable_torques()

input("[Enter] to goto init (suspend first)")
_ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

input("[Enter] to go through trajectory")
buffers, time_elapsed = poppy.track_trajectory(trajectory, overshoot=1.)

input("[Enter] to go compliant (hold strap first)")
poppy.disable_torques()

# reset PID
K_p, K_i, K_d = 4.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

with open("inflect_buffers.pkl", "wb") as f:
    pk.dump((buffers, time_elapsed, poppy.motor_names), f)

# print("don't forget poppy.close()")
print("closing poppy...")
poppy.close()
print("closed.")

