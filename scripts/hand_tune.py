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

# load hand data
hand = {}
for suffix in ('init', 'shift', 'lift', 'plant', 'mirror'):
    with open("hand_" + suffix + ".pkl", "rb") as f:
        hand[suffix] = pk.load(f)

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

def check_ranges(abs_grid, nominal_position):

    msg = "(%d, %d) vs (%.2f,%.2f) <= %.2f | [s] for success, [f] for fail, append [r] to abort row, [g] to abort grid: "

    targs, actus, succs = [], [], []
    try:
        for abs_row in abs_grid:
            for i, (abs_x, abs_y) in enumerate(abs_row):
    
                targ = set_upper(nominal_position, abs_x, abs_y)
                actu = goto_pos(targ, dur = (3. if i == 0 else .5))
                max_diff = max([abs(targ[n] - actu[n]) for n in targ.keys()])
                result = input(msg % (abs_x, abs_y, actu["abs_x"], actu["abs_y"], max_diff))
            
                if result[0] in "sf":
                    targs.append(targ)
                    actus.append(actu)
            
                if result[0] == "s": succs.append(True)
                if result[0] == "f": succs.append(False)
            
                if ("r" in result) or ("g" in result): break
            if "g" in result: break
        
        msg = "(%d, %d) vs (%.2f,%.2f) <= %.2f | %s"
        for targ, actu, succ in zip(targs, actus, succs):
            max_diff = max([abs(targ[n] - actu[n]) for n in targ.keys()])
            print(msg % (targ['abs_x'], targ['abs_y'], actu['abs_x'], actu['abs_y'], max_diff, ("" if succ else "X")))

    except Exception as e:
        print(repr(e))
    finally:
        return targs, actus, succs

# overwrite results if already saved
try:
    with open("hand_tune_results.pkl","rb") as f:
        (targets, actuals, success) = pk.load(f)
except:
    targets, actuals, success = {}, {}, {}

input("[Enter] to enable torque")
poppy.enable_torques()

# empirically check range of lean angles

# # nominal torso parameters from poindexter/hand.py (in degrees)
# init_abs_y = 10 # initial forward torso lean angle
# shift_abs_x = 30 # shift waypoint lateral torso lean angle

input("[Enter] to start check range (hold up)")

# # (0,9) looks the best, smallest max_diff and middle of success range
# abs_grid = [[(0, abs_y) for abs_y in range(0, 16)]]
# targets["init"], actuals["init"], success["init"] = check_ranges(abs_grid, hand["init"])

# # # diagonals (9,9) -> (18,18) look best
# abs_grid = [[(abs_x, abs_y) for abs_y in range(6,19,3)] for abs_x in range(0,25,3)]
# targets["shift"], actuals["shift"], success["shift"] = check_ranges(abs_grid, hand["shift"])

# # # ~(24,0) seems best
# # abs_grid = [[(abs_x, abs_y) for abs_y in range(-6,10,5)] for abs_x in range(0,25,6)] # coarse
# abs_grid = [[(abs_x, abs_y) for abs_y in range(-2,11,2)] for abs_x in range(18,27,2)] # fine
# targets["lift"], actuals["lift"], success["lift"] = check_ranges(abs_grid, hand["lift"])

# # (28,0) might be best but none are good
abs_grid = [[(abs_x, abs_y) for abs_y in range(-6,7,3)] for abs_x in range(24,33,2)]
targets["plant"], actuals["plant"], success["plant"] = check_ranges(abs_grid, hand["plant"])

with open("hand_tune_results.pkl","wb") as f:
    pk.dump((targets, actuals, success), f)

input("[Enter] to comply (hold up)")
poppy.disable_torques()

print("closing poppy...")
poppy.close()

