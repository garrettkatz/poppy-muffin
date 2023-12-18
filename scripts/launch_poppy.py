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

poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    # OpenCVCamera("poppy-cam", 0, 10),
)

print("Created poppy with 10 fps camera.  Don't forget to poppy.close() before quit() when you are finished to clean up the motor state.")

traj_name = sys.argv[1] # e.g. poppy_opt_traj_40_ilc.pkl

# ILC
input("[Enter] to enable torques and run %s ..." % traj_name)
with open(traj_name,"rb") as f: traj = pk.load(f)

# go to initial
poppy.enable_torques()
_ = poppy.track_trajectory([(.5, traj[0][1])], overshoot=10.)
time.sleep(1.)

# run full
_ = poppy.track_trajectory(traj, overshoot=10.)

print("Don't forget to disable_torques() and close()")

# # dbg
# traj=[(0.5, {'l_ankle_y': 15.0}), (0.5, {'l_ankle_y': 0}), (0.5, {'l_ankle_y': -15.0}), (0.5, {'l_ankle_y': -30.0})]
# traj2=[(0.5, {'l_knee_y': 120.0}), (0.5, {'l_knee_y': 110.0}), (0.5, {'l_knee_y': 100.0}), (0.5, {'l_knee_y': 90.0})]
# traj3=[(0.5, {'r_ankle_y': 15.0}), (0.5, {'r_ankle_y': 0}), (0.5, {'r_ankle_y': -15.0}), (0.5, {'r_ankle_y': -30.0})]
# traj4=[(0.5, {'l_shoulder_x': 110.}), (0.5, {'l_shoulder_x': 120.}), (0.5, {'l_shoulder_x': 130.}), (0.5, {'l_shoulder_x': 140.})]
# with open("poppy_opt_traj.pkl","rb") as f: opt_traj = pk.load(f)
# opt_traj_ankles = [(d, {name: a[name] for name in ('l_ankle_y', 'r_ankle_y')}) for (d, a) in opt_traj]
# opt_traj_slow = [(.5, a) for (_, a) in opt_traj]
# opt_traj_slow_ankles = [(.5, {name: a[name] for name in ('l_ankle_y', 'r_ankle_y')}) for (_, a) in opt_traj]
# opt_traj_left_slow = [(.5, {'l_ankle_y': a['l_ankle_y']}) for (d, a) in opt_traj]
# traj5 = [(d, {'l_ankle_y': float(a['l_ankle_y'])}) for (d,a) in opt_traj]

# poppy.comply()
# input("Position ankle and press enter...")
# # poppy.comply(False) # high-level
# poppy.enable_torques() # low-level
# result = poppy.track_trajectory(traj, overshoot=30.)
# # poppy.comply() # high-level
# poppy.disable_torques() # low-level

