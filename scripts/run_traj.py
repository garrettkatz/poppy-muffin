"""
$ python run_traj.py <trajectory_file>.pkl
trajectory file should contain a single list of form
    [(dur0, waypoint0), ..., (durN, waypointN)]
saves buffers as <trajectory_file>_buffers.pkl
"""
import sys
import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid as PH
    from pypot.sensor import OpenCVCamera
    import pypot.utils.pypot_time as time
except:
    from mocks import PoppyHumanoid as PH
    from mocks import OpenCVCamera
    import time

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

trajectory_base = sys.argv[1]

poppy = pw.PoppyWrapper(PH())
    
# load planned trajectory
with open(trajectory_base + '.pkl', "rb") as f: trajectory = pk.load(f)

# get initial angles
_, init_angles = trajectory[0]

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

with open(trajectory_base + "_buffers.pkl", "wb") as f:
    pk.dump((buffers, time_elapsed, poppy.motor_names), f)

# print("don't forget poppy.close()")
print("closing poppy...")
poppy.close()
print("closed.")

