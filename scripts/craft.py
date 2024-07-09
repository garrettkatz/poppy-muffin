import time
import sys, os
import pickle as pk
import random
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

def cos_traj(dur, ang0, ang1):
    # apply sinusoid interpolation

    # setup timing
    fps = 5. # frames per second, hardware does not like much more than 5
    num_pts = int(dur * fps)
    spf = dur / num_pts # seconds per frame

    # cosine interpolants
    ts = np.linspace(0, dur, num_pts+1) # +1 for inclusive linspace
    A, B = (ang0 - ang1)/2, (ang0 + ang1)/2
    angs = A*np.cos(np.pi*ts/dur) + B

    return angs, spf

# import matplotlib.pyplot as pt
# angs, spf = cos_traj(2., 5., -3.)
# pt.plot(np.arange(len(angs))*spf, angs)
# pt.show()

if __name__ == "__main__":


    # initialize hardware
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

    # lean 3 might not be enough
    input("[Enter] to goto init")
    zero_angles = {name: 0. for name in poppy.motor_names}
    init_angles = dict(zero_angles)
    init_angles["abs_y"] = 7.
    _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

    increase sway to 20 based on below
    also motors seem to be suffering with big sways like this, try slightly increasing duration?
    and before a full bend, try just swaying right, bending left up, and freezing there to keep balance

    # # sway = 10 is not far enough to take much balance off left, but stands up to bend=5
    # # besides, if 18 sway wasn't enough to fall over, none of these should do what you want!
    # # but >=18 makes the strap taut...
    # sway = 15.
    # sway_angs, spf = cos_traj(1., 0., float(sway))
    # sway_traj = []
    # for a in range(len(sway_angs)):
    #     angs = dict(init_angles)
    #     # sway
    #     angs["abs_x"] = sway_angs[a]
    #     angs["r_shoulder_x"] = -sway_angs[a]
    #     angs["l_shoulder_x"] = +sway_angs[a]
    #     sway_traj.append((spf, angs))

    # for bend in range(1, 11):

    #     bend_angs, spf = cos_traj(1., 0., float(bend))
    #     bend_traj = []
    #     for a in range(len(bend_angs)):
    #         angs = dict(sway_traj[-1][1])
    #         # bend
    #         angs["l_hip_y"] = -bend_angs[a]
    #         angs["l_knee_y"] = +bend_angs[a]
    #         angs["l_ankle_y"] = -bend_angs[a]
    #         bend_traj.append((spf, angs))

    #     # full trajectory round and back
    #     traj = sway_traj + bend_traj
    #     traj = traj + traj[::-1]

    #     # sway right and back
    #     input("[Enter] to bend to %f" % bend)
    #     res = poppy.track_trajectory(traj, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    # # (lean, sway, bend)
    # # (5, 10, 3) fell backwards    
    # # (5, 10, 4) fell backwards on return
    # # (7, 10, 0-5) stood but didn't really lift left
    # # repeated and around bend=5, some motors seemed to give out
    # # at first seemed like it did the first of two track trajs but not the second
    # sway = 10.
    # for bend in range(1, 11):
    #     sway_angs, spf = cos_traj(1., 0., float(sway))
    #     bend_angs, spf = cos_traj(1., 0., float(bend))
    #     traj = []
    #     for a in range(len(sway_angs)):
    #         angs = dict(init_angles)
    #         # sway
    #         angs["abs_x"] = sway_angs[a]
    #         angs["r_shoulder_x"] = -sway_angs[a]
    #         angs["l_shoulder_x"] = +sway_angs[a]
    #         # bend
    #         angs["l_hip_y"] = -bend_angs[a]
    #         angs["l_knee_y"] = +bend_angs[a]
    #         angs["l_ankle_y"] = -bend_angs[a]
    #         traj.append((spf, angs))

    #     # sway right and back
    #     input("[Enter] to bend to %f" % bend)
    #     res1 = poppy.track_trajectory(traj, overshoot=1.)
    #     res2 = poppy.track_trajectory(traj[::-1], overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    # keeps balance up to sway = 18., but strap goes taut around then
    # 12. is already pushing it, head + wires pressing into right strap
    # (lean, sway) = (7, 21) finally fell right and slightly forward, but up to 20 still worked
    # two more 21 tries all still fell right.
    # for sway in range(3, 30, 3):
    # for sway in range(18, 30, 3):
    # for sway in range(18, 21):
    for sway in [21,21,21]:
        angs, spf = cos_traj(1., 0., float(sway))
        traj = []
        for ang in angs:
            sway_angles = dict(init_angles)
            sway_angles["abs_x"] = ang
            sway_angles["r_shoulder_x"] = -ang # clear right arm during sway
            sway_angles["l_shoulder_x"] = +ang # clear left arm during sway
            traj.append((spf, sway_angles))

        # sway right and back
        input("[Enter] to sway to %f" % sway)
        res1 = poppy.track_trajectory(traj, overshoot=1.)
        res2 = poppy.track_trajectory(traj[::-1], overshoot=1.)

        q = input("Enter [q] to abort")
        if q == "q": break


    input("[Enter] to go to zero and then compliant (hold strap first)")
    _ = poppy.track_trajectory([(1., zero_angles)], overshoot=1.)
    poppy.disable_torques()

    # revert PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # close connection
    print("Closing poppy...")
    poppy.close()
    print("closed.")

