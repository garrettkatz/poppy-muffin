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

    # initialize hardware
    poppy = pw.PoppyWrapper(
        PoppyHumanoid(),
        # OpenCVCamera("poppy-cam", 0, 10),
    )
    
    # PID tuning
    K_p, K_i, K_d = 16.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    # # floor-less swing
    # waypoints = [
    #     # init
    #     (0., {
    #         'l_hip_y': -3. + 1., 'l_ankle_y': -1., 'r_hip_y': -1., 'r_ankle_y': +1.,
    #         'abs_y': 9., 'r_shoulder_y': -9., 'l_shoulder_y': -9.,}),
    #     # mirror
    #     (1., {
    #         'l_hip_y': -3. - 1., 'l_ankle_y': +1., 'r_hip_y': +1., 'r_ankle_y': -1.,
    #         'abs_y': 9., 'r_shoulder_y': -9., 'l_shoulder_y': -9.,}),
    # ]

    # symmetric knee bend at halfway to clear floor
    # sway=17 actually looked kind of close
    start = 2.
    bend = 10.
    for sway in [9, 12, 15, 17, 19, 21, 23, 25]:
        print("sway = %d" % sway)

        sway = float(sway)
        sarm = sway
        waypoints = [
            # init
            (0., {
                'l_hip_y': -3. + start, 'l_knee_y': 0., 'l_ankle_y': -start,
                'r_hip_y': -start, 'r_knee_y': 0., 'r_ankle_y': +start,
                'abs_y': 9., 'r_shoulder_y': -9., 'l_shoulder_y': -9.,
                'r_shoulder_x': 0., 'l_shoulder_x': 0., 'abs_x': 0.}),
            # delay
            (0.2, {
                # 'l_hip_y': -3. + start, 'l_knee_y': 0., 'l_ankle_y': -start,}),
                'l_hip_y': -3. + start, 'l_knee_y': 0., 'l_ankle_y': 0.,}), # ankle push
            # bend
            (0.4, {
                'l_hip_y': -3. - bend, 'l_knee_y': 2*bend, 'l_ankle_y': -bend,
                'r_shoulder_x': -sarm, 'abs_x': +sway}),
            # mirror init
            (0.8, {
                'l_hip_y': -3. - start, 'l_knee_y': 0., 'l_ankle_y': +start,
                'r_hip_y': +start, 'r_knee_y': 0., 'r_ankle_y': -start,
                'abs_y': 9., 'r_shoulder_y': -9., 'l_shoulder_y': -9.,
                'r_shoulder_x': 0., 'l_shoulder_x': 0., 'abs_x': 0.}),
            # mirror delay
            (1.0, {
                # 'r_hip_y': +start, 'r_knee_y': 0., 'r_ankle_y': -start,}),
                'r_hip_y': +start, 'r_knee_y': 0., 'r_ankle_y': 0.,}),
            # mirror bend
            (1.2, {
                'r_hip_y': -bend, 'r_knee_y': 2*bend, 'r_ankle_y': -bend,
                'l_shoulder_x': +sarm, 'abs_x': -sway}),
            # init
            (1.6, {
                'l_hip_y': -3. + start, 'l_knee_y': 0., 'l_ankle_y': -start,
                'r_hip_y': -start, 'r_knee_y': 0., 'r_ankle_y': +start,
                'abs_y': 9., 'r_shoulder_y': -9., 'l_shoulder_y': -9.,
                'r_shoulder_x': 0., 'l_shoulder_x': 0., 'abs_x': 0.}),
        ]
        timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)
        print("len(traj) = %d" % len(trajectory))
    
        # import matplotlib.pyplot as pt
        # pt.plot(timepoints, [a['r_hip_y'] for (_,a) in trajectory], 'b.-')
        # pt.show()
    
        input("[Enter] to enable torques and goto init")
        poppy.enable_torques()
        _ = poppy.track_trajectory([(1., trajectory[0][1])], overshoot=1.)
    
        for i in range(5):
    
            q = input("[Enter] for cycle (q to abort)")
            if q == "q": break
            bufs, _ = poppy.track_trajectory(trajectory, overshoot=1.)
        
            # q = input("[Enter] for reset (q to abort)")
            # if q == "q": break
            # bufs, _ = poppy.track_trajectory(trajectory[::-1], overshoot=1.)
    
        input("[Enter] to go to zero and then compliant (hold strap first)")
        _ = poppy.track_trajectory([(1., {m.name: 0. for m in poppy.motors})], overshoot=1.)
        poppy.disable_torques()

    # revert PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # close connection
    print("Closing poppy...")
    poppy.close()
    print("closed.")
