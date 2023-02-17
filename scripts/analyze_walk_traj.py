import matplotlib.pyplot as pt
import numpy as np
import pickle as pk

with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)
# with open('walk_buffers.real.pkl', 'rb') as f: (motor_names, buffers, durations) = pk.load(f)
with open('walk_buffers.real.nohang.pkl', 'rb') as f: (motor_names, buffers, durations) = pk.load(f)

bufkeys = ("position", "speed", "load", "voltage", "temperature")

target_durations, target_angles = [], []
split_durations = np.zeros(len(trajs))
for t,traj in enumerate(trajs):
    durs, angles = zip(*traj)
    angles = np.array([[a[n] for n in motor_names] for a in angles])
    target_durations.extend(durs)
    target_angles.append(angles)
    split_durations[t] += sum(durs)
target_angles = np.concatenate(target_angles, axis=0)
target_durations = np.cumsum(target_durations)
split_durations = np.cumsum(split_durations)

actual_angles = np.array(buffers['position'])
actual_loads = np.array(buffers['load'])
actual_temperatures = np.array(buffers['temperature'])
actual_durations = np.cumsum(durations)

colors = 'bgrcmyk'
k = 0
for j in range(len(motor_names)):
    if np.fabs(target_angles[:,j]).max() > 5:
        pt.subplot(3,1,1)
        pt.plot(target_durations, target_angles[:,j], colors[k % len(colors)] + '+:')
        pt.plot(actual_durations, actual_angles[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
        pt.subplot(3,1,2)
        pt.plot(actual_durations, actual_loads[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
        pt.subplot(3,1,3)
        pt.plot(actual_durations, actual_temperatures[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
        k += 1
    # else:
    #     pt.plot(target_durations, target_angles[:,j], 'k+:')
    #     pt.plot(actual_durations, actual_angles[:,j], 'k.-')
for sp in range(1,4):
    pt.subplot(3,1,sp)
    if sp == 1:
        for dur in split_durations:
            pt.plot([dur, dur], [target_angles.min(), target_angles.max()], 'k:')
    pt.legend(loc='upper left')
    pt.xlabel('time')
    pt.ylabel('value')

pt.tight_layout()
pt.show()



