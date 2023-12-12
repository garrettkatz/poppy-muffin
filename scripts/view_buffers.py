import sys
import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

pt.rcParams["text.usetex"] = True
pt.rcParams['font.family'] = 'serif'

traj_name = sys.argv[1]
bufs_name = sys.argv[2]

# # load planned trajectory
with open(traj_name, "rb") as f:
    trajectory = pk.load(f)

durations, waypoints = zip(*trajectory)
schedule = np.array(durations).cumsum()
motor_names = list(waypoints[0].keys())
planned = np.array([[waypoint[name] for name in motor_names] for waypoint in waypoints])

## load the data
with open(bufs_name, "rb") as f:
    (buffers, elapsed, all_motor_names) = pk.load(f, encoding='latin1')

# with open('opt_traj_frames.pkl', "rb") as f:
#     frames = pk.load(f, encoding='latin1')

actuals = buffers['position']
targets = buffers['target']
motor_idx = [all_motor_names.index(name) for name in motor_names]

pt.subplot(2,1,1)
pt.plot(schedule, planned, linestyle='-', color='b', marker='+', label='Planned')
pt.plot(elapsed, targets[:, motor_idx], linestyle='--', color='r', marker='+', label='Target')
pt.plot(elapsed, actuals[:, motor_idx], linestyle='-', marker='+', color='k', label='Actual')
for m, name in enumerate(motor_names):
    pt.text(elapsed[0], actuals[0, motor_idx[m]], name, color='b')
pt.ylabel('Joint Angles (deg)')
pt.legend()

pt.subplot(2,1,2)
pt.plot(elapsed, actuals - targets, 'k-')
for m, name in enumerate(motor_names):
    pt.text(elapsed[0], (actuals-targets)[0, motor_idx[m]], name, color='b')
pt.ylabel('Actual - Target Angles (deg)')
pt.xlabel("Time Elapsed (sec)")
pt.tight_layout()
pt.show()

## frames[i][t]: camera image at timestep t of ith waypoint
#     camera image has dimensions (rows, columns, channels)
#     channels are in reverse order (b,g,r), pixel values have type uint8

pt.ion()
pt.show()
for i in range(len(frames)):
    if i == 10: break
    start = time.perf_counter()
    elapsed = bufs[i][2]
    for t in range(len(frames[i])):
        print(i,t)
        # convert image to matplotlib format
        img = frames[i][t][:,:,[2,1,0]].astype(float)/255.

        # initialize image the very first time
        if i == t == 0:
            im = pt.imshow(img)
        # thereafter just overwrite the image data for fast code
        else:
            im.set_data(img)
            pt.gcf().canvas.draw()

        pt.pause(0.01)

        # # render animation at same speed as original motions
        # animation_time = time.perf_counter() - start
        # if animation_time < elapsed[t]:
        #     pt.pause(elapsed[t] - animation_time)

pt.close()

# total number of frames and dimensions of frames
total = 0
for i in range(len(frames)):
    total += len(frames[i])
rows, cols = frames[i][0].shape[:2]

print("%d frames with dimensions (%d, %d)" % (total, rows, cols))


