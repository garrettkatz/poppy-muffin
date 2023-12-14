import sys
import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid
    from pypot.sensor import OpenCVCamera
except:
    from mocks import PoppyHumanoid
    from mocks import OpenCVCamera

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    # OpenCVCamera("poppy-cam", 0, 10),
)

print("Created poppy with 10 fps camera.  Don't forget to poppy.close() before quit() when you are finished to clean up the motor state.")

# dbg
traj=[(0.5, {'l_ankle_y': 15.0}), (0.5, {'l_ankle_y': 0}), (0.5, {'l_ankle_y': -15.0}), (0.5, {'l_ankle_y': -30.0})]
traj2=[(0.5, {'l_knee_y': 120.0}), (0.5, {'l_knee_y': 110.0}), (0.5, {'l_knee_y': 100.0}), (0.5, {'l_knee_y': 90.0})]
traj3=[(0.5, {'r_ankle_y': 15.0}), (0.5, {'r_ankle_y': 0}), (0.5, {'r_ankle_y': -15.0}), (0.5, {'r_ankle_y': -30.0})]
poppy.comply()
input("Position ankle and press enter...")
poppy.comply(False) # high-level
# poppy.enable_torques() # low-level
result = poppy.track_trajectory(traj, overshoot=30.)
poppy.comply(True) # high-level
# poppy.disable_torques() # low-level

