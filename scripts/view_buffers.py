import sys
import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

#pt.rcParams["text.usetex"] = True
pt.rcParams['font.family'] = 'serif'

traj_name = sys.argv[1]
bufs_name = sys.argv[2]

# # load planned trajectory
with open(traj_name, "rb") as f:
    trajectory = pk.load(f, encoding='latin1')

## load the data
with open(bufs_name, "rb") as f:
    (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

if len(sys.argv) > 3:
    ctrl_name = sys.argv[3]
    with open(ctrl_name, "rb") as f:
        control_adjustments = pk.load(f, encoding="latin1")
    control_adjustments = np.array(control_adjustments)

print("Average time elapsed between buffer measurements:")
elapsed = np.array(elapsed)
print((elapsed[1:] - elapsed[:-1]).mean())

# print(all_motor_names)
# print(len(all_motor_names))
# input()

durations, waypoints = zip(*trajectory)
schedule = np.array(durations).cumsum()
motor_names = list(waypoints[0].keys())
planned = np.array([[waypoint.get(name, 0.) for name in all_motor_names] for waypoint in waypoints])

# with open('opt_traj_frames.pkl', "rb") as f:
#     frames = pk.load(f, encoding='latin1')

actuals = buffers['position']
targets = buffers['target']
# motor_idx = [all_motor_names.index(name) for name in motor_names]
# motor_idx = np.flatnonzero(np.fabs(planned).max(axis=0) > .5)
# motor_idx = [all_motor_names.index(name) for name in ("l_hip_y", "l_knee_y", "l_ankle_y", "l_hip_x", "r_hip_x")]
motor_idx = [all_motor_names.index(name) for name in ("r_hip_y", "r_knee_y", "r_ankle_y", "l_hip_x", "r_hip_x")] # mirrored
print(motor_idx)

if len(sys.argv) > 3:
    pt.subplot(2,1,1)
# pt.plot(elapsed, targets[:, motor_idx], linestyle='--', color='r', marker='+', label='Target')
# pt.plot(schedule, planned, linestyle='-', color='b', marker='o', label='Planned')
# pt.plot(elapsed, actuals[:, motor_idx], linestyle='-', marker='+', color='k', label='Actual')
# for m, name in enumerate(motor_names):
#     pt.text(elapsed[0], actuals[0, motor_idx[m]], name, color='b')
for m in motor_idx:
    name = all_motor_names[m]
    pt.plot(schedule, planned[:,m], linestyle='-', marker='.', label=f'{name} [planned]')
    pt.plot(elapsed, actuals[:,m], linestyle='-', marker='+', label=f'{name} [actual]')
    pt.text(elapsed[0], actuals[0, m], name, color='b')
pt.ylabel('Joint Angles (deg)')
pt.legend()

if len(sys.argv) > 3:
    pt.subplot(2,1,2)
    for m in motor_idx:
        name = all_motor_names[m]
        #pt.plot(schedule, planned[:,m], linestyle='-', marker='.', label=f'{name} [planned]')
        #pt.plot(schedule, planned[:,m] + control_adjustments[:,m], linestyle='-', marker='.', label=f'{name} [control]')
        pt.plot(schedule, control_adjustments[:,m], linestyle='-', marker='.', label=f'{name} [control adjustment]')
        pt.ylabel('Control Adjustments (deg)')

pt.xlabel("Time Elapsed (sec)")

pt.tight_layout()
pt.show()

frames = buffers["images"]
## frames[i]: camera image at ith waypoint
#     camera image has dimensions (rows, columns, channels)
#     channels are in reverse order (b,g,r), pixel values have type uint8

# skip None frames (was throttled to fps)
print(f"{len(frames)} frames before")
frames = [frame for frame in frames if frame is not None]
print(f"{len(frames)} frames after")

# total number of frames and dimensions of frames
rows, cols = frames[0].shape[:2]
print("%d frames with dimensions (%d, %d)" % (len(frames), rows, cols))

pt.ion()
pt.show()
for i in range(len(frames)):
    # if i == 10: break
    start = time.perf_counter()

    # convert image to matplotlib format
    img = frames[i][:,:,[2,1,0]].astype(float)/255.

    # initialize image the very first time
    if i == 0:
        im = pt.imshow(img)
    # thereafter just overwrite the image data for fast code
    else:
        im.set_data(img)
        pt.gcf().canvas.draw()

    # render animation at fixed rate
    pt.pause(0.1)

    # # render animation at same speed as original motions
    # animation_time = time.perf_counter() - start
    # if animation_time < elapsed[i]:
    #     pt.pause(elapsed[i] - animation_time)

input('.')
pt.close()


