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

poppy = pw.PoppyWrapper(PH())

# PID tuning
K_p, K_i, K_d = 8.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

# lean_angle = 8.
# sway_angle = 0.
timestep = .25
l_hip_y_0 = -3

zero_angles = {name: 0. for name in poppy.motor_names}
zero_angles["l_hip_y"] = l_hip_y_0

input("[Enter] to enable torques")
poppy.enable_torques()

input("[Enter] to goto init (suspend first)")
init_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = 8.)
_ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

# for bend_angle in (20,):
    # print("Trying bend angle", bend_angle)

while True:

    """
    lean_angle, bend1, bend2, bend3, tilt, stretch, which mean respectively:
    lean angle: torso forward, whole time
    bend1: both legs bent at first
    bend2: lesser bend in left leg for pressure
    bend3: greater bend in left leg for lift
    tilt: r_hip_y rotation during bend2 to keep stance foot flat
    stretch: l_hip_y outward stretch during bend3 (but actually negative is better after adding tilt)

    successful left swing: 12 10 4 15 10 -7 (12/28 3:08)
    successful right swing: 12 10 4 15 8 -7 (12/28 4:01)
    but in both cases, swing foot tends to land slightly behind its starting point
    """

    angs = input("Enter <lean> <bend1 both> <bend2 push> <bend3 lift> <tilt right> <stretch left>: ")
    try:
        lean_angle, bend1, bend2, bend3, tilt, stretch = tuple(map(float, angs.split()))
    except:
        print("invalid format")
        continue

    input("[Enter] to bend knees")
    bent1_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = lean_angle)
    bent1_angles.update({
        "r_ankle_y": -bend1, "l_ankle_y": -bend1,
        "r_knee_y": 2*bend1, "l_knee_y": 2*bend1,
        "r_hip_y": -bend1, "l_hip_y": -bend1 + l_hip_y_0,
    })
    _ = poppy.track_trajectory([(2., bent1_angles)], overshoot=1.)
    
    input("[Enter] to quickly step (get ready with strap)")
    # straighten for pressure through swing foot
    bent2_angles = dict(bent1_angles)
    bent2_angles.update({
        "l_ankle_y": -bend2,
        "l_knee_y": 2*bend2,
        "l_hip_y": -bend2 + l_hip_y_0,
        "r_hip_x": -tilt, # keep stance foot flat
    })

    # flex to lift swing foot
    bent3_angles = dict(bent2_angles)
    bent3_angles.update({
        "l_ankle_y": -bend3,
        "l_knee_y": 2*bend3,
        "l_hip_y": -bend3 + l_hip_y_0,
        "l_hip_x": stretch, # counteract gravity pulling foot inwards
        "r_hip_x": -tilt,
    })

    # back to plant
    bent4_angles = dict(bent1_angles)

    traj = [
        (timestep, bent2_angles),
        (timestep, bent3_angles),
        (timestep, bent4_angles),
    ]

    # code to mirror trajectory
    for t, (_, angle_dict) in enumerate(traj):
        # don't overwrite original
        angle_dict = dict(angle_dict)
        # undo asymmetrical hip offset
        angle_dict["l_hip_y"] -= l_hip_y_0
        # mirror angles
        angle_dict = poppy.get_mirror_angles(angle_dict)
        # redo asymmetrical hip offset
        angle_dict["l_hip_y"] += l_hip_y_0
        # save new waypoint
        traj[t] = (timestep, angle_dict)

    buffers, elapsed = poppy.track_trajectory(traj, overshoot=1.)

    with open("empirical_traj.pkl", "wb") as f: pk.dump(traj, f)
    with open("empirical_bufs.pkl", "wb") as f: pk.dump((buffers, elapsed, poppy.motor_names), f)

    qr = input("[Enter] to continue, r to rest, q to abort")

    if qr in "qr":

        input("[Enter] to go compliant (hold strap first)")
        poppy.disable_torques()

        if qr == "q": break
        # else qr == "r"

        input("[Enter] to enable torques")
        poppy.enable_torques()
        
        input("[Enter] to goto init (suspend first)")
        init_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = 8.)
        _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

    
# reset PID
K_p, K_i, K_d = 4.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

# print("don't forget poppy.close()")
print("closing poppy...")
poppy.close()
print("closed.")

