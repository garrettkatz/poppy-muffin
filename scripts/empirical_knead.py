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

    angs = input("Enter <lean> <bend1 both> <bend2 push> <bend3 lift> <stretch1 left> <stretch2 left>: ")
    try:
        lean_angle, bend1, bend2, bend3, stretch1, stretch2 = tuple(map(float, angs.split()))
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
    bent2_angles = dict(bent1_angles)
    bent2_angles.update({
        "l_ankle_y": -bend2,
        "l_knee_y": 2*bend2,
        "l_hip_y": -bend2 + l_hip_y_0,
    })

    bent3_angles = dict(bent2_angles)
    bent3_angles.update({
        "l_ankle_y": -bend3,
        "l_knee_y": 2*bend3,
        "l_hip_y": -bend3 + l_hip_y_0,
        "l_hip_x": stretch1, # counteract gravity pulling foot inwards
    })

    bent4_angles = dict(bent1_angles)
    bent4_angles["l_hip_x"] = stretch2 # counteract gravity pulling foot inwards

    traj = [
        (timestep, bent2_angles),
        (timestep, bent3_angles),
        (timestep, bent4_angles),
    ]
    buffers, elapsed = poppy.track_trajectory(traj, overshoot=1.)

    with open("empirical_traj.pkl", "wb") as f: pk.dump(traj, f)
    with open("empirical_bufs.pkl", "wb") as f: pk.dump((buffers, elapsed, poppy.motor_names), f)

    # # input("[Enter] to re-init")
    # # back to init
    # _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)

    q = input("[Enter] to continue, q to abort")
    if q == "q": break
    

input("[Enter] to go compliant (hold strap first)")
poppy.disable_torques()

# reset PID
K_p, K_i, K_d = 4.0, 0.0, 0.0
for m in poppy.motors:
    if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

# print("don't forget poppy.close()")
print("closing poppy...")
poppy.close()
print("closed.")

