import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

num_reps = 3
num_samples = 10

results = np.empty((num_reps, num_samples))
for rep in range(num_reps):
    for sample in range(num_samples):
    
        with open('walk_samples/rep%d/results_%d.pkl' % (rep, sample), "rb") as f:
            (motor_names, result, bufs) = pk.load(f, encoding='latin1')

        results[rep, sample] = result

print(results.T)
        
for t, transition in enumerate(bufs):
    pt.subplot(1, len(bufs), t+1)
    offset = 0
    for (success, buffers, time_elapsed) in transition:
        if not success: continue
        pt.plot(offset + time_elapsed, buffers['position'], 'k-')
        pt.plot([offset, offset], [buffers['position'].min(), buffers['position'].max()], 'k:')
        offset += time_elapsed[-1]
pt.show()

