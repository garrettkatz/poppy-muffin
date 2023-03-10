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

# [0, 3, 4, 5, 2, 6, 8, 9, 1, 7]
# [8, 4, 0, 3, 6, 9, 5, 1, 7, 2]
# [3, 8, 7, 5, 4,||| 1,||| 2, 6, 9, 0]

sample = int(sys.argv[1])

save_images = True
if save_images:
    poppy = pw.PoppyWrapper(PH(), OpenCVCamera("poppy-cam", 0, 24))
    binsize = 10
else:
    poppy = pw.PoppyWrapper(PH())
    binsize = None
    
# load planned trajectory
if sample < 0:
    with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)
elif sample >= 100:
    with open('pypot_traj_star.pkl', "rb") as f: trajs = pk.load(f)
else:
    with open('walk_samples/pypot_sample_trajectory_%d.pkl' % sample, "rb") as f: trajs = pk.load(f)

# get initial angles
_, init_angles = trajs[0][0]

# PID tuning
K_p, K_i, K_d = 20.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

input('[Enter] to turn off compliance')
for m in poppy.motors: m.compliant = False

input('[Enter] for init angles (may want to hold up by strap)')

init_buf = poppy.goto_position(init_angles, duration=1, bufsize=10, speed_ratio=-1) # don't abort for speed
bufs = [[init_buf]]

input('[Enter] to begin walking')

num_cycles, stop_traj = 1, 5 # one step, return to init
if np.abs(sample) >= 100:
    num_cycles = 10
    stop_traj = len(trajs)

success = True
for cycle in range(num_cycles):
    for t, traj in enumerate(trajs[:stop_traj]):
    
        # settle briefly at waypoint
        if t not in [3]: # don't wait before kick
            time.sleep(0.1)
    
        bufs.append([])
        for s, (duration, angles) in enumerate(traj[1:]): # skip (0, start)
            buf = poppy.goto_position(angles, duration, bufsize=10, speed_ratio=-1, binsize=binsize)
            bufs[-1].append(buf)
            success = buf[0]
            print('  success = %s' % str(success))
            if not success: break
        if not success: break

    # wait at final pose for stability
    if success:
        print('settling into init...')
        time.sleep(3)

    # check for next step
    if num_cycles > 1:
        cmd = input('[Enter] for next step, [q] to abort: ')
        if cmd == 'q': break

input('[Enter] to return to rest and go compliant (may want to hold up by strap)')
poppy.goto_position({name: 0. for name in poppy.motor_names}, 3, bufsize=10, speed_ratio=-1)
for m in poppy.motors: m.compliant = True

print("closing...")
poppy.close()
print("closed.")

while True:
    try:
        if num_cycles == 1:
            print("How far did Poppy get?")
            print("0 - nowhere")
            print("1 - to shift")
            print("2 - to push")
            print("3 - to kick")
            print("4 - all the way")
            result = int(input("Enter the result: "))
            assert result in [0,1,2,3,4]
            break
        else:
            print("How many steps did Poppy get?")
            result = int(input("Enter the result: "))
            assert result in list(range(0, 2*num_cycles+1))
            break
    except:
        print("Invalid input.")

with open('walk_samples/results_%d.pkl' % sample, "wb") as f:
    pk.dump((poppy.motor_names, result, bufs), f)

