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
    K_p, K_i, K_d = 16.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    input("[Enter] to enable torques")
    poppy.enable_torques()

    lean = 7.
    l_hip_0 = -3.

    # lean 3 might not be enough
    input("[Enter] to goto init")
    zero_angles = {name: 0. for name in poppy.motor_names}
    init_angles = dict(zero_angles)
    init_angles["abs_y"] = lean
    init_angles["l_hip_y"] = l_hip_0
    _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

    # sway,bank: (bend=7, l_hip_y=0)
    # 15, 5 already fell right, and when returning, did the typical rotate on its feet to face left
    # 14, 7 looked a bit unstable but kept balance at sway.  However, started to fall right on return (swing leg straight too fast?)
    # 14, 6 worked surprisingly well! kept balance and returned with toe in nearly correct position 
    # 14, 8 was similar to 14, 7
    # 14, 5 surprisingly also fell right, then after returning left ankle motor soon crapped out and fall forward
    # surprising 6 worked but 5 didn't; maybe because of motor overheat and not angle setting
    # ***___ need to retry 14, 6.  and maybe 13 is also more stable?

    # (14,6) not reproducible, fell right on sway, same for (14,5) again
    # (13,6) worked once, (13,5) fell right
    # wide distance between feet might help

    # (13,6) worked again, (13,5) fell right again (feet might have been closer on 5)
    # (12,6) worked, but at max sway, the left foot loosing pressure on the ground visible swung closer to the right before stabilizing.  return still worked but feet much closer together.
    # (12,5) worked and left foot swung was less pronounced.

    # (12,6) again had slight left swing-in
    # (12,5) and 4 did too, but all 3 got back without falling
    # (11,6) almost had a little wobble back left when it hit max sway, and left foot position more stable
    # (11,5) similar, but (11,4) was a fall (which way? repeat)

    # (11,7) worked, left foot ends slightly closer and back
    # (11,6) same, wobble left before left foot catches
    # (11,5) worked, left foot ends better, but almost looked like it was going to fall back-left direction
    # (11,4) did fall back-left at max sway

    # after this, inspect the initial zero angles vs what is actually aligned feet angles
    # buffers somewhat match expectation:
    # at beginning, l_hip_y,r_hip_y = -.29, .13, so left thigh is forward more than assigned and right thigh is backward more than assigned

    # try switching to initial/final l_hip_y = -1 instead of 0?
    # init_angles with l_hip_0 = -1. still has left leg slightly back
    # (12,7) looks pretty good but still slight left foot in,back
    # (12,6) too and pronounced
    # (12,5) looked borderline fall left-back
    # (11,7) similar wobbled left
    # (11,6) too, it's falling onto its left outer side foot when should only be toe
    # (11,5) fell backleft
    
    # l_hip_0 = -2
    # init angles slightly closer, maybe just one more to 3
    # (13,6) worked but weight is all along left outer side, lean more?
    # (12,8) moreso and left ends more backin
    # (12,7) like (13,6)
    # (12,6) also same

    # l_hip_0 = -3, abs_y = lean+1 at max sway
    # (13,6) worked but left foot backin alot
    # (12,8) too, so much so that it fell right after straightening
    # (12,7) some left backin but worked
    # (12,6) fell right
    # (11,7) fell right
    # (11,6) motors ignored and no motion, FFFFFFFFFF

    # going back to just abs_y = lean always
    # instead make l_hip_x = 1. at sway to compensate swing-in
    # (13,6) worked pretty well
    # (12,8), (12,7), (12,6) fell right
    # (11,7) worked at max sway with left heel lifted, but fell right on straighten
    # (11,6) worked with heel up and straighten, but end left foot was a bit back and in

    # trying a few (11,6) with lean down to 5. since l_hip_y = -3 would be slight bend forward
    # wobbly but worked but heel not up.  clears momentarily but falls back on it
    # trying a bit slower and later lift to avoid backlash after sway
    # (.4, 1.) -> (.6, 1.2)
    # seems a bit closer but same problem, lean up to 6
    # first had same problem, next two actually worked but because left toe accidentally landed further back
    # need to move CoM slightly forward and/or right, but don't want to overload lower joints
    # try swinging right arm forward a bit

    # !! in fact, maybe lean less and move both arms forward instead? easier to control? and if nothing else, less strain on shoulders
    
    # sarm = -5. qualitatively the same, but seemed a bit stable
    # sarm = -10. similar to -5.  first rep had toe in pretty good spot and slight heel clearance
    # sarm = -20. one rep also looked pretty good, but bust motor crapped out
    # bust motor might be same principle: why force it to flex in so many motors like shoulders etc
    # maybe you can use higher motors to change CoM instead of abs

    # going back to lean = 7 with higher Kp
    # sarm = 0: qualitatively same

    # trying bigger bend/bank, (sway,bank)
    # (11,15) fell right(left?) but could keep balance after
    # (11,12) fell left but kept balance after
    # this suggests that foot clears too fast, too early; start it later
    # changing to .8 instead of .6: still same problem and still too much weight on left

    bufs = {}

    sarm = 0
    bend = 15
    # sway = 14
    # for bank in [7,6,8,5,9]:
    # for (sway, bank) in [(13,6), (12,8), (12,7), (12,6), (11,7), (11,6)]:
    for (sway, bank) in [(13,18),(13,15),(13,12),(13,9),(13,6)]:
        waypoints = [
            (0., {"abs_y": float(lean), "abs_x": 0., "r_shoulder_x": 0., "r_shoulder_y": 0.,
                  "l_hip_y": l_hip_0, "l_knee_y": 0., "l_ankle_y": 0., "l_hip_x": 0.}),
            (.8, {"l_hip_y": l_hip_0, "l_knee_y": 0., "l_ankle_y": 0., "l_hip_x": 0.}),
            (1.2, {"abs_y": float(lean), "abs_x": float(sway), "r_shoulder_x": -3.*float(sway), "r_shoulder_y": float(sarm),
                  "l_hip_y": -float(bend), "l_knee_y": 2.*float(bend), "l_ankle_y": -float(bank), "l_hip_x": 1.}),
        ]
    
        timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)

        input("[Enter] to sway,bank to %f,%f" % (sway,bank))
        bufs[sway,bank,0], _ = poppy.track_trajectory(trajectory, overshoot=1.)

        q = input("Enter [q] to abort")
        if q == "q": break

        # go back more slowly
        input("[Enter] to go back")
        reverse = [(2*d, a) for (d,a) in trajectory[::-1]]
        bufs[sway,bank,1], _  = poppy.track_trajectory(reverse, overshoot=1.)

        q = input("Enter [q] to abort")
        if q == "q": break

    # # with this version 14 already gets up on the toe, and 16 is already unstable to the right
    # # (good news for taut strap)
    # bend = 5.
    # for sway in range(14, 20, 1):
    #     waypoints = [
    #         (0., {"abs_y": 7., "abs_x": 0., "r_shoulder_x": 0.,
    #               "l_hip_y": 0., "l_knee_y": 0., "l_ankle_y": 0.}),
    #         (.4, {"l_hip_y": 0., "l_knee_y": 0., "l_ankle_y": 0.}),
    #         (1., {"abs_y": 7., "abs_x": float(sway), "r_shoulder_x": -3.*float(sway),
    #               "l_hip_y": -bend, "l_knee_y": 2.*bend, "l_ankle_y": -bend}),
    #     ]
    
    #     timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)

    #     input("[Enter] to sway to %f" % sway)
    #     res = poppy.track_trajectory(trajectory, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    #     # go back more slowly
    #     input("[Enter] to sway back")
    #     reverse = [(2*d, a) for (d,a) in trajectory[::-1]]
    #     res = poppy.track_trajectory(reverse, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    # # didn't exactly keep balance on one leg, but ended up unintentionally lifting swing heel (toe still made contact)
    # # when returning though, swing foot would end up further back; maybe too much pressure on toe
    # bend = 5.
    # for sway in range(14, 20, 1):
    #     waypoints = [
    #         (0., {"abs_y": 7., "abs_x": 0., "r_shoulder_x": 0.,
    #               "l_hip_y": 0., "l_knee_y": 0., "l_ankle_y": 0.}),
    #         (1., {"abs_y": 7., "abs_x": float(sway), "r_shoulder_x": -3.*float(sway),
    #               "l_hip_y": 0., "l_knee_y": 0., "l_ankle_y": 0.}),
    #         (2., {"abs_y": 7., "abs_x": float(sway), "r_shoulder_x": -3.*float(sway),
    #               "l_hip_y": -bend, "l_knee_y": 2*bend, "l_ankle_y": -bend}),
    #     ]
    
    #     timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)

    #     input("[Enter] to sway to %f" % sway)
    #     res = poppy.track_trajectory(trajectory, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    #     # go back more slowly
    #     input("[Enter] to sway back")
    #     reverse = [(2*d, a) for (d,a) in trajectory[::-1]]
    #     res = poppy.track_trajectory(reverse, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    # # 17 is also unstable at pause point, a little noise makes it fall right
    # for sway in range(14, 20, 1):
    #     waypoints = [
    #         (0., {"abs_y": 7., "abs_x": 0., "r_shoulder_x": 0.}),
    #         (1., {"abs_y": 7., "abs_x": float(sway), "r_shoulder_x": -3.*float(sway)}),
    #     ]
    
    #     timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)

    #     input("[Enter] to sway to %f" % sway)
    #     res = poppy.track_trajectory(trajectory, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    #     # go back more slowly
    #     input("[Enter] to sway back")
    #     reverse = [(2*d, a) for (d,a) in trajectory[::-1]]
    #     res = poppy.track_trajectory(reverse, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

    # # sway 21 falls at 1s motion but not 2s or 1.6s; velocity is a factor
    # # even 18 fell right at 1s motion (maybe now with the shoulder tweak)
    # for sway in range(3, 22, 3):
    #     waypoints = [
    #         (0., {"abs_y": 7., "abs_x": 0., "r_shoulder_x": 0.}),
    #         (1., {"abs_y": 7., "abs_x": float(sway), "r_shoulder_x": -3.*float(sway)}),
    #     ]
    
    #     timepoints, trajectory = make_cosine_trajectory(waypoints, poppy.motor_names, fps=5.)
    #     trajectory = trajectory + trajectory[::-1] # sway and return

    #     input("[Enter] to sway to %f" % sway)
    #     res = poppy.track_trajectory(trajectory, overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

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

    # # keeps balance up to sway = 18., but strap goes taut around then
    # # 12. is already pushing it, head + wires pressing into right strap
    # # (lean, sway) = (7, 21) finally fell right and slightly forward, but up to 20 still worked
    # # two more 21 tries all still fell right.
    # # for sway in range(3, 30, 3):
    # # for sway in range(18, 30, 3):
    # # for sway in range(18, 21):
    # for sway in [21,21,21]:
    #     angs, spf = cos_traj(1., 0., float(sway))
    #     traj = []
    #     for ang in angs:
    #         sway_angles = dict(init_angles)
    #         sway_angles["abs_x"] = ang
    #         sway_angles["r_shoulder_x"] = -ang # clear right arm during sway
    #         sway_angles["l_shoulder_x"] = +ang # clear left arm during sway
    #         traj.append((spf, sway_angles))

    #     # sway right and back
    #     input("[Enter] to sway to %f" % sway)
    #     res1 = poppy.track_trajectory(traj, overshoot=1.)
    #     res2 = poppy.track_trajectory(traj[::-1], overshoot=1.)

    #     q = input("Enter [q] to abort")
    #     if q == "q": break

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

