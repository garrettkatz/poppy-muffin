import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

with open("search_inflect.pkl","rb") as f: results, motor_names = pk.load(f, encoding='latin1')

params, data = zip(*results[0].items())
params = np.array(params)
success = np.array([d[0] for d in data])
error = np.empty(len(success))
for d in range(len(data)):
    actuals, elapsed, planned, timepoints = data[d][1:]
    error[d] = np.mean(np.fabs(actuals[-1] - planned[-1]))

print(f"{len(params)} samples")

good_params = params[success,:]
good_errors = error[success]
bad_params = params[~success,:]
bad_errors = error[~success]

print(np.sort(np.concatenate((good_params, good_errors[:,None]), axis=1), axis=0))
print(np.sort(np.concatenate((bad_params, bad_errors[:,None]), axis=1), axis=0))

actuals, elapsed, planned, timepoints = data[-1][1:]
print(actuals.shape)
# motor_idx = [all_motor_names.index(name) for name in motor_names]
motor_idx = np.flatnonzero(np.fabs(planned).max(axis=0) > .1)
print(motor_idx)

for m in motor_idx:
    name = motor_names[m]
    # pt.plot(schedule, planned[:,m], linestyle='-', marker='.', label=f'{name} [planned]')
    pt.plot(timepoints, planned[:,m], linestyle='-', marker='.', label=f'{name} [planned]')
    pt.plot(elapsed, actuals[:,m], linestyle='-', marker='+', label=f'{name} [actual]')
    pt.text(elapsed[0], actuals[0, m], name, color='b')
pt.ylabel('Joint Angles (deg)')
pt.xlabel("Time Elapsed (sec)")
pt.legend()

pt.tight_layout()
pt.show()

