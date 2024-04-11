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

# load hand data
waypoints = ('init', 'shift', 'lift', 'plant', 'mirror')
hand = {}
for suffix in waypoints:
    with open("hand_" + suffix + ".pkl", "rb") as f:
        hand[suffix] = pk.load(f)

# configure transitions
transition_sampling = 20
transition_duration = 1.

def setup_transitions(init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, mirror=True):

    # set uppers
    uppers = dict()

    uppers['init'] = set_upper(hand['init'], init_abs_x, init_abs_y)
    uppers['shift'] = set_upper(hand['shift'], (init_abs_x + swing_abs_x)/2, (init_abs_y + swing_abs_y)/2)
    uppers['lift'] = set_upper(hand['lift'], swing_abs_x, swing_abs_y)
    uppers['plant'] = set_upper(hand['plant'], (init_abs_x + swing_abs_x)/2, (init_abs_y + swing_abs_y)/2)
    uppers['mirror'] = set_upper(hand['mirror'], init_abs_x, init_abs_y)

    # mirror everything if requested
    if mirror:
        for suffix in waypoints:
            uppers[suffix] = get_mirror(uppers[suffix])

    # build transition trajectories
    steps = np.arange(1, transition_sampling+1)
    sample_weights = (np.cos(np.pi * steps / transition_sampling) + 1.)/2.
    step_duration = transition_duration / transition_sampling    
    
    transitions = []
    for wp in range(len(waypoints)-1):
        start = uppers[waypoints[wp]]
        end = uppers[waypoints[wp+1]]
        trajectory = []
    
        for weight in sample_weights:
            target = {
                name: weight * start[name] + (1. - weight) * end[name]
                for name in start}
            trajectory.append((step_duration, target))
    
        transitions.append(trajectory)

    return transitions

transitions = setup_transitions(init_abs_x = 9, init_abs_y = 0, swing_abs_x = 9, swing_abs_y = 6)

print(hand['init']['r_knee_y'])
print(hand['init']['l_knee_y'])
print(hand['mirror']['r_knee_y'])
print(hand['mirror']['l_knee_y'])

import matplotlib.pyplot as pt
durs = [dur for traj in transitions for (dur,_) in traj]
angs = [targ['r_knee_y'] for traj in transitions for (_, targ) in traj]
pt.plot(np.cumsum(durs), angs, 'k.-')
pt.show()

poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    # OpenCVCamera("poppy-cam", 0, 10),
)

# wrap low-level trajectory for one waypoint
def goto_pos(goal_positions, dur=1.):
    _ = poppy.track_trajectory([(dur, goal_positions)], overshoot=10.)
    time.sleep(.25) # catch up end pos
    end_pos = poppy.remap_low_to_high(poppy.get_present_positions())
    return poppy.dict_from(end_pos)

def dsa():
    poppy.disable_torques()

input("[Enter] to go compliant and init (suspend first)")
poppy.enable_torques()

_ = goto_pos(hand['init'], dur = 3.)

input("[Enter] to go through trajectory")
tracked = []
for traj in transitions:
    buffers, time_elapsed = poppy.track_trajectory(traj, overshoot=1.)
    tracked.append((buffers, time_elapsed))

with open("hand_tune_buffers.pkl", "wb") as f:
    pk.dump(tracked, f)

input("[Enter] to go compliant")
dsa()

print("closing poppy...")
poppy.close()

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

