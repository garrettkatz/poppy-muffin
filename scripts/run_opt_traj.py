"""
$ python run_opt_traj.py <traj file>.pkl
    pkl data should be [... (duration, angles) ...] in (seconds, rad)
saves output data as
    opt_traj_result.pkl: (motor_names, bufs)
        motor_names[j]: jth motor name
        bufs[n]: buffers for nth trajectory waypoint
    opt_traj_frames.pkl: vids
        vids[n]: camera frames for nth trajectory waypoint
    (overwrites previous run due to storage limitations)
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

# load planned trajectory
traj_file = sys.argv[1]
with open(traj_file, "rb") as f:
    trajectory = pk.load(f)

save_images = False
if save_images:
    poppy = pw.PoppyWrapper(PH(), OpenCVCamera("poppy-cam", 0, 24))
    binsize = 5
else:
    poppy = pw.PoppyWrapper(PH())
    binsize = None

# get initial angles
_, init_angles = trajectory[0]

# # PID tuning
for m in poppy.motors:
    if hasattr(m, 'pid'): print(m.name, m.pid) # default is (4., 0., 0.)
K_p, K_i, K_d = 20.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

input('[Enter] to turn off compliance')
poppy.comply(False)

input('[Enter] for init angles (may want to hold up by strap)')

poppy.goto_position(init_angles, duration=1, bufsize=10, speed_ratio=-1) # don't abort for speed

input('[Enter] to run trajectory')

bufs = []
vids = []
success = True
    
for s, (duration, angles) in enumerate(trajectory[1:]): # skip (_, init_angles)
    buf = poppy.goto_position(angles, duration, bufsize=10, speed_ratio=-1, binsize=binsize)
    if save_images:
        vid = buf[1].pop('images')
        vids.append(vid)
    bufs.append(buf)
    success = buf[0]
    print('  success = %s' % str(success))
    if not success: break

input('[Enter] to return to rest and go compliant (may want to hold up by strap)')
poppy.goto_position({name: 0. for name in poppy.motor_names}, 3, bufsize=10, speed_ratio=-1)
poppy.comply()

print("closing...")
poppy.close()
print("closed.")

with open('opt_traj_result.pkl', "wb") as f:
    pk.dump((poppy.motor_names, bufs), f)

# overwrite frame file every time due to limited storage space on chip
with open('opt_traj_frames.pkl', "wb") as f:
    pk.dump(vids, f)



