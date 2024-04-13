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

# set upper body (from poindexter/hand.py without degree conversion)
def set_upper(angle_dict, abs_x, abs_y):
    angle_dict = dict(angle_dict)
    angle_dict['abs_x'] = abs_x
    angle_dict['abs_y'] = abs_y
    angle_dict['bust_x'] = -abs_x
    angle_dict['bust_y'] = -abs_y

    # slightly wide arms to reduce self-collision
    angle_dict['r_shoulder_x'] = min(-5., -abs(abs_x))
    angle_dict['l_shoulder_x'] = max(+5., +abs(abs_x))

    return angle_dict

# mirror joints laterally (from poindexter/hand.py)
def get_mirror(angle_dict):
    mirrored = dict(angle_dict)
    for name, angle in angle_dict.items():

        # don't negate y-axis rotations
        sign = 1 if name[-2:] == "_y" else -1

        # swap right and left
        mirror_name = name
        if name[:2] == "l_": mirror_name = "r_" + name[2:]
        if name[:2] == "r_": mirror_name = "l_" + name[2:]

        # assign mirrored angle
        mirrored[mirror_name] = angle * sign

    return mirrored

with open('hand_init.pkl', 'rb') as f:
    init = pk.load(f)

with open('hand_trajectories.pkl', 'rb') as f:
    transitions, interpolants = pk.load(f)

# config
do_mirror = False
transition_duration = 1.
init_abs_x = 0
init_abs_y = 9
swing_abs_x = 18
swing_abs_y = 18
# (20,20) was fr

swing_abs = [(swing_abs_x, swing_abs_y)
    for (swing_abs_x, swing_abs_y) in [
        (35,18)]]
    # done
        # (30,18)]]
        # (26,18)]]
        # (25,18)]]
    # for swing_abs_x in range(22,25,1)
    #     for swing_abs_y in range(15,18,1)]
    # # done
    # for swing_abs_x in range(20,23,1)
    #     for swing_abs_y in range(13,16,1)]
    # # done, towards the end these were getting through to lift phase but com was still too far left
    # for (swing_abs_x, swing_abs_y) in [
    #     (17,14),
    #     (18,14), (18,15), (18,16),
    #     (19,13), (19,14), (19,15), (19,16), 
    #     ]]
    # # done
    # for swing_abs_x in range(12,19,2)
    #     for swing_abs_y in range(9,14,2)]
    # # done
    # for swing_abs_x in range(11,18,2)
    #     for swing_abs_y in range(11,18,2)]
    # # done
    # for swing_abs_x in range(10,17,2)
    #     for swing_abs_y in range(14,25,2)]

def setup_transitions(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, mirror=False):

    mid_abs_x = (init_abs_x + swing_abs_x)/2.
    mid_abs_y = (init_abs_y + swing_abs_y)/2.
    upways = {
        ("init","shift"): ((init_abs_x, init_abs_y), (mid_abs_x, mid_abs_y)),
        ("shift","lift"): ((mid_abs_x, mid_abs_y), (swing_abs_x, swing_abs_y)),
        ("lift","plant"): ((swing_abs_x, swing_abs_y), (mid_abs_x, mid_abs_y)),
        ("plant","mirror"): ((mid_abs_x, mid_abs_y), (init_abs_x, init_abs_y)),
    }

    # deterministic key order for py2.7
    keys = [
        ("init","shift"),
        ("shift","lift"),
        ("lift","plant"),
        ("plant","mirror"),
    ]

    # set uppers in flat single trajectory
    uppered = []
    # for key, transition in transitions.items():
    for key in keys:
        transition = transitions[key]
        old_abs, new_abs = upways[key]
        old_abs_x, old_abs_y = old_abs
        new_abs_x, new_abs_y = new_abs

        for t, (dur, targ) in enumerate(transition):

            abs_x = interpolants[key][t]*old_abs_x + (1-interpolants[key][t])*new_abs_x
            abs_y = interpolants[key][t]*old_abs_y + (1-interpolants[key][t])*new_abs_y

            targ = set_upper(targ, abs_x, abs_y)
            if mirror: targ = get_mirror(targ)

            # hand.py scaled transition durations to 1, rescale here
            uppered.append((dur*transition_duration, targ))

    return uppered

uppered = setup_transitions(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, mirror=do_mirror)

### show what traj will look like

# import matplotlib.pyplot as pt
# for key in init:
#     durs = [dur for (dur,_) in uppered]
#     angs = [targ[key] for (_, targ) in uppered]
#     pt.plot(np.cumsum(durs), angs, '.-', label=key)
# pt.legend()
# pt.show()

# import matplotlib.pyplot as pt
# durs = [dur for (dur,_) in uppered]
# angs = [targ['r_knee_y'] for (_, targ) in uppered]
# pt.plot(np.cumsum(durs), angs, '.-', label='r_knee_y')
# angs = [targ['l_knee_y'] for (_, targ) in uppered]
# pt.plot(np.cumsum(durs), angs, '.-', label='l_knee_y')
# angs = [targ['abs_y'] for (_, targ) in uppered]
# pt.plot(np.cumsum(durs), angs, '.-', label='abs_y')
# angs = [targ['r_shoulder_x'] for (_, targ) in uppered]
# pt.plot(np.cumsum(durs), angs, '.-', label='r_shoulder_x')
# pt.legend()
# pt.show()

poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    # OpenCVCamera("poppy-cam", 0, 10),
)

# PID tuning
K_p, K_i, K_d = 10.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

# wrap low-level trajectory for one waypoint
def goto_pos(goal_positions, dur=1.):
    _ = poppy.track_trajectory([(dur, goal_positions)], overshoot=1.)
    time.sleep(.25) # catch up end pos
    end_pos = poppy.remap_low_to_high(poppy.get_present_positions())
    return poppy.dict_from(end_pos)

def dsa():
    poppy.disable_torques()

# for trial, (swing_abs_x, swing_abs_y) in enumerate(swing_abs):
#     print("trial %s: abs_x=%d, abs_y=%d" % (trial, swing_abs_x, swing_abs_y))
trial, abort, results = 0, "", []
while abort != "a":
    for res in results: print(res)
    swing_abs_x, swing_abs_y = map(int, input("Enter [abs_x],[abs_y]: ").split(","))
    print("trial %s: abs_x=%d, abs_y=%d" % (trial, swing_abs_x, swing_abs_y))
    trial += 1

    trajectory = setup_transitions(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, mirror=do_mirror)
    
    try:
    
        input("[Enter] to enable torques")
        poppy.enable_torques()
        
        input("[Enter] to goto init (suspend first)")
        _ = goto_pos(trajectory[0][-1], dur = 1.)
        
        input("[Enter] to go through trajectory")
        buffers, time_elapsed = poppy.track_trajectory(trajectory, overshoot=1.)
        
        input("[Enter] to go compliant (hold strap first)")
        dsa()

        fall = input("Fall direction: Combo of [f]ront or [b]ack and [l]eft or [r]ight, or [s]uccess: ")

        results.append((swing_abs_x, swing_abs_y, fall))

        suffix = "_%d_%d_%d_%d_%s" % (init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, fall)
        if do_mirror: suffix += "_m"
        
        with open("hand_tune_buffers%s.pkl" % suffix, "wb") as f:
            pk.dump((buffers, time_elapsed, poppy.motor_names), f)
        
        with open("hand_tune_uppered%s.pkl" % suffix, "wb") as f:
            pk.dump(trajectory, f)

        abort = input("[Enter] for next, or [a]bort: ")
        if abort == "a": break

    
    except:
        print("something broke")
        abort = "a"

# reset PID
K_p, K_i, K_d = 4.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

if abort != "a":
    print("closing poppy...")
    poppy.close()

else:
    print("Don't forget to poppy.close()")


# ## grid search
# def check_swing_abs(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y):

#     uppered = setup_transitions(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y)

#     input("[Enter] to init (suspend first)")
#     _ = goto_pos(init, dur = 3.)

#     input("[Enter] to go through trajectory")
#     # buffers, time_elapsed = poppy.track_trajectory(uppered, overshoot=1.)
#     buffers, time_elapsed = poppy.track_trajectory(uppered_mirror, overshoot=1.)

#     time.sleep(.25) # catch up end pos
#     end_pos = poppy.remap_low_to_high(poppy.get_present_positions())
#     end_dict = poppy.dict_from(end_pos)
#     end_diff = max([abs(uppered[-1][-1][name] - actu) for (name, actu) in end_diff.items()])

#     return buffers, time_elapsed, end_pos, end_diff



# for

#     msg = "end diff = %f | " % end_diff
#     msg += "[s] for success, [f] for fail, append [r] to abort row, [g] to abort grid: "

#     result = input(msg % (abs_x, abs_y, actu["abs_x"], actu["abs_y"], max_diff))







# def check_ranges(abs_grid, nominal_position):

#     msg = "(%d, %d) vs (%.2f,%.2f) <= %.2f | [s] for success, [f] for fail, append [r] to abort row, [g] to abort grid: "

#     targs, actus, succs = [], [], []
#     try:
#         for abs_row in abs_grid:
#             for i, (abs_x, abs_y) in enumerate(abs_row):
    
#                 targ = set_upper(nominal_position, abs_x, abs_y)
#                 actu = goto_pos(targ, dur = (3. if i == 0 else .5))
#                 max_diff = max([abs(targ[n] - actu[n]) for n in targ.keys()])
#                 result = input(msg % (abs_x, abs_y, actu["abs_x"], actu["abs_y"], max_diff))
            
#                 if result[0] in "sf":
#                     targs.append(targ)
#                     actus.append(actu)
            
#                 if result[0] == "s": succs.append(True)
#                 if result[0] == "f": succs.append(False)
            
#                 if ("r" in result) or ("g" in result): break
#             if "g" in result: break
        
#         msg = "(%d, %d) vs (%.2f,%.2f) <= %.2f | %s"
#         for targ, actu, succ in zip(targs, actus, succs):
#             max_diff = max([abs(targ[n] - actu[n]) for n in targ.keys()])
#             print(msg % (targ['abs_x'], targ['abs_y'], actu['abs_x'], actu['abs_y'], max_diff, ("" if succ else "X")))

#     except Exception as e:
#         print(repr(e))
#     finally:
#         return targs, actus, succs

# # overwrite results if already saved
# try:
#     with open("hand_tune_results.pkl","rb") as f:
#         (targets, actuals, success) = pk.load(f)
# except:
#     targets, actuals, success = {}, {}, {}

# input("[Enter] to enable torque")
# poppy.enable_torques()

# # empirically check range of lean angles

# # # nominal torso parameters from poindexter/hand.py (in degrees)
# # init_abs_y = 10 # initial forward torso lean angle
# # shift_abs_x = 30 # shift waypoint lateral torso lean angle

# input("[Enter] to start check range (hold up)")

# # # (0,9) looks the best, smallest max_diff and middle of success range
# # abs_grid = [[(0, abs_y) for abs_y in range(0, 16)]]
# # targets["init"], actuals["init"], success["init"] = check_ranges(abs_grid, hand["init"])

# # # # diagonals (9,9) -> (18,18) look best
# # abs_grid = [[(abs_x, abs_y) for abs_y in range(6,19,3)] for abs_x in range(0,25,3)]
# # targets["shift"], actuals["shift"], success["shift"] = check_ranges(abs_grid, hand["shift"])

# # # # ~(24,0) seems best
# # # abs_grid = [[(abs_x, abs_y) for abs_y in range(-6,10,5)] for abs_x in range(0,25,6)] # coarse
# # abs_grid = [[(abs_x, abs_y) for abs_y in range(-2,11,2)] for abs_x in range(18,27,2)] # fine
# # targets["lift"], actuals["lift"], success["lift"] = check_ranges(abs_grid, hand["lift"])

# # # (28,0) might be best but none are good
# abs_grid = [[(abs_x, abs_y) for abs_y in range(-6,7,3)] for abs_x in range(24,33,2)]
# targets["plant"], actuals["plant"], success["plant"] = check_ranges(abs_grid, hand["plant"])

# with open("hand_tune_results.pkl","wb") as f:
#     pk.dump((targets, actuals, success), f)

# input("[Enter] to comply (hold up)")
# poppy.disable_torques()

# print("closing poppy...")
# poppy.close()

