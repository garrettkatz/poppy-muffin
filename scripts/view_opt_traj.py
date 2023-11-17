import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

pt.rcParams["text.usetex"] = True
pt.rcParams['font.family'] = 'serif'

## load the data
with open('opt_traj_result.pkl', "rb") as f:
    (motor_names, bufs) = pk.load(f, encoding='latin1')
with open('opt_traj_frames.pkl', "rb") as f:
    frames = pk.load(f, encoding='latin1')

# load planned trajectory
traj_file = "poppy_opt_traj.pkl" # sys.argv[1]
with open(traj_file, "rb") as f:
    trajectory = pk.load(f)

## bufs[i]: buffer data for ith waypoint of trajectory
# bufs[i] is a tuple (flag, buffers, elapsed)
# flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
# buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
# buffers['target'][t, j]: target angle of jth joint at timestep t of motion
# elapsed[t]: elapsed time since start of motion at timestep t of motion

flags, buffers, elapsed = zip(*bufs)
actuals = np.concatenate([buffers[i]['position'] for i in range(len(buffers))], axis=0)
targets = np.concatenate([buffers[i]['target'] for i in range(len(buffers))], axis=0)
# accumulate elapsed time over multiple waypoints
for i in range(1, len(elapsed)):
    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
elapsed = np.concatenate(elapsed)

planned = []
for i, (_, angles) in enumerate(trajectory[1:]):
    angs = [angles[name] for name in motor_names]
    planned.append(np.array([angs]*len(buffers[i]['position'])))
planned = np.concatenate(planned, axis=0)

pt.subplot(2,1,1)
pt.plot(elapsed, planned, linestyle='-', color='b', label='Planned')
pt.plot(elapsed, targets, linestyle='--', color='r', label='Target')
pt.plot(elapsed, actuals, linestyle='-', marker='+', color='k', label='Actual')
for m, name in enumerate(motor_names):
    pt.text(elapsed[0], actuals[0][m], name, color='b')
pt.ylabel('Joint Angles (deg)')

pt.subplot(2,1,2)
pt.plot(elapsed, actuals - targets, 'k-')
for m, name in enumerate(motor_names):
    pt.text(elapsed[0], (actuals-targets)[0][m], name, color='b')
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


