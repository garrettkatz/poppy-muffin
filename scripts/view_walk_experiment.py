import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

pt.rcParams["text.usetex"] = True
pt.rcParams['font.family'] = 'serif'

num_perturbed = 30
num_orig = 30

## original params (copied from poppy-simulations/ergo/pybullet/tasks/walking/experiment_trajectories.py

# angle from vertical axis to flat leg in initial stance
init_flat = .02*np.pi
# angle for abs_y joint in initial stance
init_abs_y = np.pi/16
# angle from swing leg to vertical axis in shift stance
shift_swing = .05*np.pi
# angle of torso towards support leg in shift stance
shift_torso = np.pi/5
# angle from vertical axis to flat leg in push stance
push_flat = -.00*np.pi
# angle from swing leg to vertical axis in push stance
push_swing = -.10*np.pi

orig_params = np.array([init_flat, init_abs_y, shift_swing, shift_torso, push_flat, push_swing])

# perturbations to params
with open(f'walk_samples/exp_traj_data.pkl', "rb") as f: sample_data = pk.load(f)
perturbations, _, _ = zip(*sample_data)
perturbations = np.array(perturbations)

results = {'perturbed': np.empty((num_perturbed)), 'hand-tuned': np.empty(num_orig)}
mads = {'perturbed': np.empty((num_perturbed)), 'hand-tuned': np.empty(num_orig)}
return_mads = {'perturbed': np.empty((num_perturbed)), 'hand-tuned': np.empty(num_orig)}

for key in ('perturbed', 'hand-tuned'):

    for sample in range(num_perturbed):
        fnum = sample if key == 'perturbed' else -(sample+1)

        with open('walk_samples/results_%d.pkl' % fnum, "rb") as f:
            (motor_names, result, bufs) = pk.load(f, encoding='latin1')
        results[key][sample] = result
    
        mad_sum = 0
        total_steps = 0
    
        # process transitions after move to init and until failure
        for t, transition in enumerate(bufs[1:][:result+1]):
            for (success, buffers, time_elapsed) in transition:

                # long-duration trajectory points will have high mad in beginning
                # mad_sum += np.fabs(buffers['position'] - buffers['target']).mean(axis=1).sum()
                # total_steps += len(time_elapsed)

                mad_sum += np.fabs(buffers['position'] - buffers['target']).mean(axis=1)[-1]
                total_steps += 1
    
        mads[key][sample] = mad_sum / total_steps

        _, final_buffer, _ = bufs[-1][-1]
        return_mads[key][sample] = np.fabs(final_buffer['position'][-1] - final_buffer['target'][-1]).mean()

marks = np.array([0, 1, 2, 3, 4])

print(results['perturbed'].T)
pt.figure(figsize=(4,2))
pt.barh(marks, [(results['perturbed']==k).sum() for k in marks], height=.4, fc='w', ec='k', align='edge', label='Perturbed')
pt.barh(marks-.4, [(results['hand-tuned']==k).sum() for k in marks], height=.4, fc=(.5,)*3, ec='k', align='edge', label='Hand-tuned')
pt.xlabel('Frequency')
pt.ylabel('Successful Transitions')
pt.yticks(marks, marks)
pt.legend()
pt.tight_layout()
pt.savefig('marks.eps')
pt.show()

# fit linear function: result = (perturbation - orig) @ A + b
delta1 = np.concatenate((perturbations - orig_params, np.ones((num_perturbed,1))), axis=1)
Ab, _, _, _ = np.linalg.lstsq(delta1, results['perturbed'], rcond=None)
dots = delta1[:, :-1] @ Ab[:-1]
orig_dot = 0

pt.figure(figsize=(4,2))
pt.plot(dots, results['perturbed'], 'k+', label='Perturbed')
pt.scatter([orig_dot]*len(marks), marks, [(results['hand-tuned']==k).sum()*10 for k in marks], fc='none', ec='k', label='Hand-tuned')

x = [dots.min(), dots.max()]
y = np.poly1d(np.polyfit(dots, results['perturbed'], 1))(x)
pt.plot(x, y, 'k:')

pt.xlabel('$\\langle \\Delta\\Theta, \\mathbf{a}\\rangle$')
pt.ylabel('Successful Transitions')
pt.yticks(marks[1:], marks[1:])
pt.ylim([0, 5.5])
pt.legend()
pt.tight_layout()
pt.savefig('linfit.eps')
pt.show()

# # fit linear function: final_mad = (perturbation - orig) @ W + z
# finished = {key: (results[key] == 4) for key in ('perturbed', 'hand-tuned')}
# Wz, _, _, _ = np.linalg.lstsq(delta1[finished['perturbed']], return_mads['perturbed'][finished['perturbed']], rcond=None)
# dots = delta1[finished['perturbed'], :-1] @ Wz[:-1]
# orig_dot = 0

# pt.figure(figsize=(4,2))
# pt.plot(dots, return_mads['perturbed'][finished['perturbed']], 'k+', label='Perturbed')
# pt.plot([orig_dot]*finished['hand-tuned'].sum(), return_mads['hand-tuned'][finished['hand-tuned']], 'ko', mfc='none', label='Hand-tuned')

# x = [dots.min(), dots.max()]
# y = np.poly1d(np.polyfit(dots, return_mads['perturbed'][finished['perturbed']], 1))(x)
# pt.plot(x, y, 'k:')

# pt.xlabel('$\\langle \\Delta\\Theta, \\mathbf{w}\\rangle$')
# pt.ylabel('Return MAD')
# pt.legend()
# pt.tight_layout()
# pt.savefig('returnfit.eps')
# pt.show()

# # orthonormal basis
# basis = np.stack((Ab[:-1], Wz[:-1]))
# basis[0] /= np.linalg.norm(basis[0])
# basis[1] -= basis[0] * np.dot(basis[0], basis[1])
# basis[1] /= np.linalg.norm(basis[1])

# proj_A = basis @ Ab[:-1]
# proj_W = basis @ Wz[:-1]
# proj_delta1 = basis @ delta1[:, :-1].T
# mad_max = max(return_mads['perturbed'].max(), return_mads['hand-tuned'].max())
# proj_max = np.fabs(proj_delta1).max()
# print(proj_max)
# colors = return_mads['perturbed']/mad_max
# colors = colors[:,np.newaxis] * np.ones(3)

# pt.plot([0, proj_A[0]], [0, proj_A[1]], 'r-')
# pt.plot([0, -proj_W[0]], [0, -proj_W[1]], 'b-')
# pt.scatter(*proj_delta1[:, finished['perturbed']], marker='+', c=colors[finished['perturbed']])
# pt.scatter(*proj_delta1[:, ~finished['perturbed']], marker='x', c=colors[~finished['perturbed']])

# for k in np.flatnonzero(finished['perturbed']):
#     pt.text(*proj_delta1[:, k], s=str(k))

# k_star = 5
# print("init_flat, init_abs_y, shift_swing, shift_torso, push_flat, push_swing")
# print(orig_params + delta1[k_star,:-1])
# print(orig_params)
# print(return_mads['perturbed'][k_star])
# print(results['perturbed'][k_star])

# pt.xlim([-proj_max, +proj_max])
# pt.ylim([-proj_max, +proj_max])
# # pt.axis('equal')
# pt.show()


# fig = pt.figure(figsize=(6.5, 3), constrained_layout=True)
# gs = fig.add_gridspec(6, 7)
fig = pt.figure(figsize=(4, 6), constrained_layout=True)

# fig.add_subplot(gs[:, :3])
offset = 0
for t, transition in enumerate(bufs[1:]): # skip init buf
    # pt.subplot(1, len(bufs), t+1)
    # offset = 0
    for (success, buffers, time_elapsed) in transition:
        if not success: continue

        pt.subplot(2,1,1)
        pt.plot(offset + time_elapsed, buffers['position'], 'k-')

        if t == 1:
            pt.subplot(2,1,2)
            pt.plot(offset + time_elapsed, buffers['position'], 'k-')
            pt.plot(offset + time_elapsed, buffers['target'], 'k:')
            pt.xlabel('Time elapsed (s)')

        offset += time_elapsed[-1]

    # pt.subplot(2,1,1)
    # pt.plot([offset, offset], [buffers['position'].min(), buffers['position'].max()], 'k:')

pt.gcf().supylabel('Joint Positions (deg)')
pt.subplot(2,1,1)

# pt.subplot(2,1,2)
# # fig.add_subplot(gs[:, 3:])
# transition = bufs[2]
# for (success, buffers, time_elapsed) in transition:
#     if not success: continue

#     pt.plot(offset + time_elapsed, buffers['position'], 'k-')
#     pt.plot(offset + time_elapsed, buffers['target'], 'k:')
#     # pt.plot([offset + time_elapsed[-1]]*buffers['target'].shape[1], buffers['target'][-1], 'ko')
#     pt.plot([offset, offset], [buffers['position'].min(), buffers['position'].max()], 'k:')

#     offset += time_elapsed[-1]

# pt.show()

pt.savefig('buffers.eps')
pt.show()

