import time
import sys, os
import pickle as pk
import random
import numpy as np
import poppy_wrapper as pw
from cosine_trajectory import make_cosine_trajectory
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

if __name__ == "__main__":

    with open('pypot_traj1_smoothed.pkl', "rb") as f: trajectory = pk.load(f)

    # initialize hardware
    poppy = pw.PoppyWrapper(
        PoppyHumanoid(),
        # OpenCVCamera("poppy-cam", 0, 10),
    )
    
    # PID tuning
    K_p, K_i, K_d = 20.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    input("[Enter] to enable torques")
    poppy.enable_torques()

    input("[Enter] go to initial (hold strap first)")
    _ = poppy.track_trajectory([(1., trajectory[0][1])], overshoot=1.)

    input("[Enter] to go through cycle")
    _ = poppy.track_trajectory(trajectory, overshoot=1.)

    input("[Enter] to go to zero and then compliant (hold strap first)")
    _ = poppy.track_trajectory([(1., {name: 0. for name in poppy.motor_names)], overshoot=1.)
    poppy.disable_torques()

    # revert PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # close connection
    print("Closing poppy...")
    poppy.close()
    print("closed.")

