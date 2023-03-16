import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

pt.rcParams["text.usetex"] = True
pt.rcParams['font.family'] = 'serif'

num_reps = 30

## load all the data
results, bufs, frames = {}, {}, {}
for rep in range(num_reps):
    with open('walk_samples/results_%d.pkl' % (-rep-1), "rb") as f:
        (motor_names, results[rep], bufs[rep]) = pk.load(f, encoding='latin1')

    with open('walk_samples/frames_%d.pkl' % (-rep-1), "rb") as f:
        frames[rep] = pk.load(f, encoding='latin1')

## results[rep]: the number of successful transitions before a fall (4 is a complete step without falling)
results = np.array(list(results.values()))

pt.figure(figsize=(4,2))
pt.barh(
    list(range(5)),
    [(results == k).sum() for k in range(5)],
    height=.8, fc=(.5,)*3, ec='k', align='center')
pt.xlabel('Frequency')
pt.ylabel('Successful Transitions')
pt.yticks(list(range(5)))
pt.tight_layout()
pt.show()

## bufs[rep][k][i]: buffer data for ith waypoint of kth trajectory in repetition rep
    # bufs[rep][k][i] is a tuple (flag, buffers, elapsed)
    # flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
    # buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
    # buffers['target'][t, j]: target angle of jth joint at timestep t of motion
    # elapsed[t]: elapsed time since start of motion at timestep t of motion

rep = np.random.randint(num_reps)
k = 0 # transition from initial to push angles
flags, buffers, elapsed = zip(*bufs[rep][k])
actuals = np.concatenate([buffers[i]['position'] for i in range(len(buffers))], axis=0)
targets = np.concatenate([buffers[i]['target'] for i in range(len(buffers))], axis=0)
# accumulate elapsed time over multiple waypoints
for i in range(1, len(elapsed)):
    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
elapsed = np.concatenate(elapsed)

pt.plot(elapsed, targets, linestyle='-', color=(.5,)*3, label='Target')
pt.plot(elapsed, actuals, linestyle='-', marker='+', color='k', label='Actual')
pt.ylabel('Joint Angles (deg)')
pt.xlabel("Time Elapsed (sec)")
pt.tight_layout()
pt.show()

## frames[rep][k][i][t]: camera image at timestep t of ith waypoint of kth trajectory in repetition rep
#     camera image has dimensions (rows, columns, channels)
#     channels are in reverse order (b,g,r), pixel values have type uint8

pt.ion()
pt.show()
for k in range(len(frames[rep])):
    for i in range(len(frames[rep][k])):
        start = time.perf_counter()
        elapsed = bufs[rep][k][i][2]
        for t in range(len(elapsed)):
            # convert image to matplotlib format
            img = frames[rep][k][i][t][:,:,[2,1,0]].astype(float)/255.

            # initialize image the very first time
            if k == i == t == 0:
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

    # ignore data after falls
    if k == results[rep]: break

pt.close()

# total number of frames and dimensions of frames
total = 0
for rep in range(num_reps):
    for k in range(len(frames[rep])):
        for i in range(len(frames[rep][k])):
            total += len(frames[rep][k][i])
rows, cols = frames[rep][k][i][0].shape[:2]

print("%d frames with dimensions (%d, %d)" % (total, rows, cols))
