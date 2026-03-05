import os, glob
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import stitch_preanalysis as sp

## nominal trajectory and representative run
if False:

    # extract run metadata
    run_filepaths = sp.get_run_filepaths()
    traj_names, bufs_names, stdevs, nfsteps = zip(*run_filepaths)
    stdevs = np.array(stdevs)
    nfsteps = np.array(nfsteps)

    # representative sample
    idx = np.random.choice(np.flatnonzero((stdevs=="0.0") & (nfsteps==6)))

    # # load planned trajectory
    with open(traj_names[idx], "rb") as f:
        trajectory = pk.load(f, encoding='latin1')
    
    ## load the data
    with open(bufs_names[idx], "rb") as f:
        (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

    # plot (similar to view_buffers.py)
    durations, waypoints = zip(*trajectory)
    # durations = [0]+list(durations)
    # waypoints = waypoints[-1:] + waypoints
    schedule = np.array(durations).cumsum()
    planned = np.array([[waypoint.get(name, 0.) for name in all_motor_names] for waypoint in waypoints])
    
    actuals = buffers['position']
    # motor_idx = [all_motor_names.index(name) for name in ("r_hip_y", "r_knee_y", "r_ankle_y", "r_hip_x")] # mirrored
    motor_idx = [all_motor_names.index(name) for name in ("l_hip_y", "r_knee_y", "r_ankle_y", "r_hip_x")] # mirrored

    pt.figure(figsize=(11,4))

    color = "rgbk"
    for m,c in zip(motor_idx,color):
        name = all_motor_names[m]
        #pt.plot(schedule, planned[:,m], linestyle='-', marker='o')
        pt.step(schedule, planned[:,m], where='pre', linestyle='-', marker='o', color=c)
        pt.plot(elapsed, actuals[:,m], ':', alpha=.9, color=c)
        pt.text(elapsed[-1]+.5, actuals[-1, m], name, color=c, fontsize=14)
    pt.xlim([-1,16])
    pt.xlabel("Time (sec)", fontsize=16)
    pt.ylabel('Joint Angles (deg)', fontsize=16)
    # pt.legend()
    pt.tight_layout()
    pt.savefig("nominal.pdf")
    pt.show()

## open-loop dataset divergence from success average
if False:

    run_filepaths = sp.get_run_filepaths()
    num_interp = 200
    total_duration = 13.5 # (1.25 + 4*0.25) * 6
    timepoints = np.linspace(0, total_duration, num_interp)
    
    positions = np.empty((len(run_filepaths), num_interp, 25))
    last_steps = np.empty(len(run_filepaths), dtype=int)
    
    for r, (_, buf_file, _, numsteps) in enumerate(run_filepaths):
    
        last_steps[r] = numsteps
    
        # load the data
        with open(buf_file, "rb") as f:
            (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')
    
        # interpolate positions
        for j in range(25):
            positions[r,:,j] = np.interp(timepoints, elapsed, buffers["position"][:,j])
    
        # save the initial target (same for all runs)
        # initial target is the *last* waypoint of every cycle (and the command send *before the first* waypoint of every cycle)
        init_target = buffers["target"][-1]
    
    # average success run
    mean_run = positions[last_steps==6].mean(axis=0)
    
    # distances to mean run
    distances = np.linalg.norm(positions - mean_run, axis=-1)
    
    # plot
    pt.figure(figsize=(8,4))
    
    pt.subplot(1,2,1)
    for d, dist in enumerate(distances):
        if last_steps[d] == 6:
            pt.plot(timepoints, dist, '-', color='b', alpha=.05)
        else:
            prefall = (timepoints <= last_steps[d] * 2.25)
            pt.plot(timepoints[prefall], dist[prefall], '-', color='r', alpha=.05)
            distances[d][~prefall] = np.nan
    
    pt.plot(timepoints, distances[last_steps==6].mean(axis=0), 'b-', label='success')
    pt.plot(timepoints, np.nanmean(distances[last_steps<6], axis=0), 'r-', label='fall')
    pt.legend()
    pt.xlabel("Time (sec)")
    pt.ylabel("Distance (deg)")
    pt.title("Success Distance")
    
    pt.subplot(1,2,2)
    
    # cycle boundaries
    idx = np.linspace(0, num_interp-1, 4).astype(int)
    returns = positions[:,idx,:]
    deviance = np.linalg.norm(returns - init_target, axis=-1)
    print(returns.shape, init_target.shape, deviance.shape)
    
    for r, ls in enumerate(last_steps):
        deviance[r, ls//2+1:] = np.nan
        pt.plot(deviance[r], '-', marker=('o' if ls==6 else 'x'), color=('b' if ls==6 else 'r'), alpha=.05)
    
    pt.plot(np.nanmean(deviance[last_steps==6],axis=0), 'b-', label='success')
    pt.plot(np.nanmean(deviance[last_steps!=6],axis=0), 'r-', label='fall')
    
    pt.legend()
    pt.xlabel("Footstep Cycle")
    pt.xticks(range(4), range(4))
    #pt.ylabel("Distance (deg)")
    pt.title("Return Distance")
    
    pt.tight_layout()
    pt.savefig("open_loop_observations.pdf")
    pt.show()

## eps, cost, lqr stability
if True:

    fig = pt.figure(figsize=(8,3))

    for e, (eps, marker, color) in enumerate(zip([0., -1., -2.], "o^x", "rgb")):

        with open(f"costcvx_ni2_eps{eps}.pkl","rb") as f: C = pk.load(f)
        with open(f"lqr_ni2_eps{eps}.pkl","rb") as f: (K, max_eigs, max_eigs_open) = pk.load(f)
        max_eigs = np.array([max_eigs[n] for n in sorted(max_eigs.keys())])
        max_eigs_open = np.array([max_eigs_open[n] for n in sorted(max_eigs_open.keys())])

        # for n in range(len(max_eigs)): print(f"{ni=}, {n=}: eig prod = {np.prod(max_eigs[:n+1])} (vs {np.prod(max_eigs_open[:n+1])} open)")

        pt.subplot(1,3,1)
        if e==0: pt.plot([np.prod(max_eigs_open[:n+1]) for n in range(10)], 'k-', label="Open-loop")
        pt.plot([np.prod(max_eigs[:n+1]) for n in range(10)], marker+color+'-', label=r"$\epsilon$" + f"={eps}")
        pt.legend()
        pt.yscale("log")
        pt.ylabel(r"$\Lambda_n$")
        pt.xlabel("Waypoint Index $n$")

        pt.subplot(1,3,2)
        for n,C_n in enumerate(C):
            eigs = np.linalg.eigvals(C[n])
            # pt.plot([n], [eigs.min()], marker+'k', alpha=.1)
            pt.plot(n+.3*e + .3*np.random.rand(len(eigs)), eigs, marker+color, alpha=.05)
        min_eigs = [np.linalg.eigvals(C_n).min() for C_n in C]
        pt.plot(min_eigs, marker+color+'-', label=r"$\epsilon$" + f"={eps}")
        pt.legend()
        pt.ylabel(r"$C_n$ eigenvalues")
        pt.xlabel("Waypoint Index $n$")

    # show classification for eps=-2
    pt.subplot(1,3,3)

    # load in data
    run_filepaths = sp.get_run_filepaths()
    with open(f"lqr_chunks_ni2.pkl","rb") as f: (obs, cmd) = pk.load(f)

    _, _, stdevs, nfsteps = zip(*run_filepaths)
    stdevs = np.array(stdevs)
    nfsteps = np.array(nfsteps)

    # flatten data
    X = obs.reshape(len(run_filepaths), 31, -1)
    A = cmd.reshape(len(run_filepaths), 30, -1)
    dim_X = X.shape[-1]
    dim_A = A.shape[-1]
    dim_XA = dim_X + dim_A

    # pool across footstep cycle
    X_pool = np.concatenate([
        X[:,0:10],
        X[:,10:20],
        X[:,20:30],
    ], axis=0)
    A_pool = np.concatenate([
        A[:,0:10],
        A[:,10:20],
        A[:,20:30],
    ], axis=0)
    stdevs_pool = np.concatenate([stdevs]*3)
    success_pool = np.concatenate([
        (nfsteps >= 2),
        (nfsteps >= 4),
        (nfsteps == 6),
    ])

    # get nominal trajectory from pooled cycles (average within noiseless success)
    X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)
    A0 = A_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

    # duplicate back to full data length for residuals
    X0 = np.concatenate([X0]*3 + [X0[:1]], axis=0)
    A0 = np.concatenate([A0]*3, axis=0)
    dX = X - X0
    dA = A - A0

    # get total costs on each run
    costs = np.empty(len(dX))
    for r, (dx, da) in enumerate(zip(dX, dA)):
        n_F = 5 * nfsteps[r]

        dxa = np.concatenate([dx[:n_F], da[:n_F]], axis=-1)

        costs[r] = dx[n_F] @ C[n_F % 10][:dim_X, :dim_X] @ dx[n_F]
        for n in range(n_F):
            costs[r] += dxa[n].T @ C[n % 10] @ dxa[n]

        costs[r] /= (n_F + 1)

    print(f"{eps=}: success cost <= {costs[nfsteps==6].max():.3f}, {costs[nfsteps<6].min():.3f} <= fall cost")

    pt.plot(.25 + .5*np.random.rand((nfsteps==6).sum()), costs[nfsteps==6], 'k.', alpha=.1)
    pt.plot(1.25+.5*np.random.rand((nfsteps<6).sum()), costs[nfsteps<6], 'k.', alpha=.1)
    pt.violinplot([costs[nfsteps==6], costs[nfsteps<6]], positions=[.5, 1.5], widths=0.5, showextrema=False)
    pt.xticks([0.5,1.5],["Success","Fall"])
    pt.xlabel("Run Class")
    pt.ylabel("Cost Per Unit Time")

    pt.tight_layout()
    pt.savefig("stability.pdf")
    pt.show()

