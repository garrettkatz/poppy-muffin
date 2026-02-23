import os, glob
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from motion_features import get_motion_features

def get_run_filepaths():
    # collect all run filenames
    run_filepaths = [] # each element is (traj_name, buf_name, stdev, num_successes)
    
    path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", "*stdev*")
    
    folders = glob.glob(path)
    folder_stdevs = []
    for folder in folders:
        stdev = folder[folder.index("stdev")+5:]
        traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
        for traj_file in traj_files:
            # filename format is traj_<n>_<num_success>.pkl for the nth episode
            num_successes = int(traj_file[traj_file.rfind("_")+1:-4])
            buf_file = traj_file.replace("traj", "bufs")
            run_filepaths.append( (traj_file, buf_file, stdev, num_successes) )

    return run_filepaths

def interpolate_arrays():

        with open("stitch_arrays.pkl","rb") as f:
            (obs_t, obs_p, obs_v, traj_t, traj_p, nfstep, stdevs) = pk.load(f)
    
        # get maximum duration
        max_duration = max(t.max() for t in obs_t)
        print(f"{max_duration=}")
    
        # linearly interpolate at evenly spaced timepoints
        num_timepoints = 13*100 # roughly 13 seconds, 100Hz
        timepoints = np.linspace(0, max_duration, num_timepoints)
        obs_ip = np.empty((len(obs_t), num_timepoints, 25)) # 25 joints
        obs_iv = np.empty((len(obs_t), num_timepoints, 2)) # 2 vision features
        for r, (t, p, v) in enumerate(zip(obs_t, obs_p, obs_v)):
            # print(f"interpolating {r} of {len(obs_t)}...")
            for j in range(25):
                obs_ip[r,:,j] = np.interp(timepoints, t, p[:,j])
            for j in range(2):
                obs_iv[r,:,j] = np.interp(timepoints, t, v[:,j])

        # wrap trajectories in arrays
        traj_t = np.array(traj_t[0]) # same durations for all trajectories
        traj_p = np.array(traj_p) # (runs, 30, 25)

        return timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs

if __name__ == "__main__":

    do_arrays = False
    view_array = False
    do_chunk = False
    view_chunk = False
    do_nearest = False
    do_centered = False
    do_nearest_full = False
    do_stability = False
    do_clone = False
    do_returns = True
    do_lindyn = False
    
    if do_arrays:
    
        run_filepaths = get_run_filepaths()
        # run_filepaths = run_filepaths[:10] # for quick tests
        print(f"{len(run_filepaths)} runs total")
        
        # use observation windows of previous footstep as policy input, and planned trajectory of next footstep as action
        # chunk the data around each (prev, next) footstep boundary, up until a fall or end of run (zero-based indexing)
        obs_t = [] # obs_t[r] timepoints of buffer observations for run r
        obs_p = [] # obs_p[r] joint positions of buffer observations for run r
        obs_v = [] # obs_v[r] visual motion features of buffer observations for run r
        traj_t = [] # traj_t[r] timepoints of trajectory for run r
        traj_p = [] # traj_p[r] joint positions of trajectory for run r
        nfstep = [] # num successful steps for run r
        stdevs = [] # noise stdev for run r
    
        for r, (traj_file, buf_file, stdev, num_successes) in enumerate(run_filepaths):
            print(f"arrays for {r} of {len(run_filepaths)}")
           
            # load planned trajectory
            with open(traj_file, "rb") as f:
                trajectory = pk.load(f, encoding='latin1')
            
            # load the data
            with open(buf_file, "rb") as f:
                (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')
                elapsed = np.array(elapsed)
                waytime_points = np.array(waytime_points)
          
            assert len(trajectory) == len(waytime_points) == 30 # 6 footsteps, 5 waypoints each
    
            # package trajectory into array
            durs, angs = zip(*trajectory)
            angs = np.array([[ang.get(name, 0.) for name in all_motor_names] for ang in angs])

            # extract motion features
            imgs = [frame[:,:,[2,1,0]].astype(float)/255. for frame in buffers["images"] if frame is not None]
            dxs, dys = [], []
            for (img1, img2) in zip(imgs[:-1], imgs[1:]):
                (dx, dy, _, _), _ = get_motion_features(img1, img2)
                dxs.append(dx)
                dys.append(dy)
            viz = []
            dx = dy = 0.
            for f, frame in enumerate(buffers["images"]):
                # exclude very first frame since you wouldn't be able to calculate motion yet at that point
                if f>0 and frame is not None:# and len(dxs)>0:
                    dx = dxs.pop(0)
                    dy = dys.pop(0)
                viz.append((dx,dy))
            viz = np.array(viz)
    
            obs_t.append(elapsed)
            obs_p.append(buffers["position"])
            obs_v.append(viz)
            traj_t.append(waytime_points)
            traj_p.append(angs)
            nfstep.append(num_successes)
            stdevs.append(stdev)
        
        # save array data
        nfstep = np.array(nfstep)
        stdevs = np.array(stdevs)
        with open("stitch_arrays.pkl","wb") as f:
            pk.dump((obs_t, obs_p, obs_v, traj_t, traj_p, nfstep, stdevs), f)

    if view_array:
        with open("stitch_arrays.pkl","rb") as f:
            (obs_t, obs_p, obs_v, traj_t, traj_p, nfstep, stdevs) = pk.load(f)

        r = 100 # noiseless success

        print(f"stdev={stdevs[r]}, {nfstep[r]} successful footsteps")
        pt.subplot(3,1,1)
        pt.step(traj_t[r], traj_p[r], where='post')
        pt.ylabel("Plan")
        pt.xlim([-1, obs_t[r][-1]+1])

        pt.subplot(3,1,2)
        pt.ylabel("Actual Joints")
        pt.plot(obs_t[r], obs_p[r])
        pt.xlim([-1, obs_t[r][-1]+1])

        pt.subplot(3,1,3)
        pt.ylabel("Vision")
        pt.plot(obs_t[r], obs_v[r])
        pt.xlim([-1, obs_t[r][-1]+1])
        pt.xlabel("Time (sec)")
        pt.show()
        
    
    if do_chunk:
    
        run_filepaths = get_run_filepaths()    
        print(f"{len(run_filepaths)} runs total")
        
        # use observation windows of previous footstep as policy input, and planned trajectory of next footstep as action
        # chunk the data around each (prev, next) footstep boundary, up until a fall or end of run (zero-based indexing)
        chunks = [] # elements are [prev footstep obs, next footstep traj] = [(timepoints, joint obs), (durations, waypoints)]
        falls = [] # True if next footstep is fall, False otherwise
        stdevs = [] # noise stdev for the episode this came from
    
        for (traj_file, buf_file, stdev) in run_filepaths:
        
            # filename format is traj_<n>_<num_success>.pkl for the nth episode
            num_successes = int(traj_file[traj_file.rfind("_")+1:-4])
        
            # load planned trajectory
            with open(traj_file, "rb") as f:
                trajectory = pk.load(f, encoding='latin1')
            
            # load the data
            with open(buf_file, "rb") as f:
                (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')
                elapsed = np.array(elapsed)
        
            print(traj_file)
            print(buf_file)
            print(repr(stdev))
            print(num_successes)
            print(len(trajectory))
            print(len(buffers["position"]))
        
            assert len(trajectory) == 30 # 6 footsteps, 5 waypoints each
        
            last_footstep = min(5, num_successes) # can only go up to footstep boundary (4|5) if all successful
            for next_footstep in range(1, last_footstep+1):
        
                # setup the index of waypoint at the boundary
                wp_idx = 5*next_footstep
        
                # get plan for next footstep
                next_footstep_traj = trajectory[wp_idx:wp_idx+5]
                durs, angs = zip(*next_footstep_traj)
                durs = np.array(durs)
                angs = np.array([[ang.get(name, 0.) for name in all_motor_names] for ang in angs])
        
                # get joint observations from previous footstep
                buf_idx = (waytime_points[wp_idx-5] <= elapsed) & (elapsed <= waytime_points[wp_idx])
                tpts = elapsed[buf_idx]
                jpos = buffers["position"][buf_idx]
        
                chunks.append( [(tpts, jpos), (durs, angs)] )
                falls.append((next_footstep == last_footstep) and (num_successes < 6))
                stdevs.append(stdev)
        
                print(f"   fall={falls[-1]}, boundary (prev={next_footstep-1}|next={next_footstep}) of {num_successes}: {len(jpos)} obs window")
    
        # save chunk data
        falls = np.array(falls)
        stdevs = np.array(stdevs)
        with open("stitch_chunks.pkl","wb") as f: pk.dump((chunks, falls, stdevs), f)
    
    if view_chunk:

        # load chunk data
        with open("stitch_chunks.pkl","rb") as f: (chunks, falls, stdevs) = pk.load(f)
    
        pt.figure(figsize=(8,3))
        colors = 'bgrcmyk' * 4
    
        # # first chunk
        # (tpts, jobs), (durs, angs) = chunks[0]
        # for j,c in enumerate(colors[:25]):
        #     pt.plot(tpts, jobs[:,j], c+'-')
        #     pt.plot(tpts[-1] + np.cumsum(durs), angs[:,j], c+'o:')
        # pt.show()
    
        # random success/fall chunks
        s_idx = np.random.choice(np.flatnonzero(falls == False))
        f_idx = np.random.choice(np.flatnonzero(falls == True))
    
        for i, idx in enumerate([s_idx, f_idx]):
            pt.subplot(1,2,i+1)
    
            (tpts, jobs), (durs, angs) = chunks[idx]
            for j,c in enumerate(colors[:25]):
                pt.plot(tpts, jobs[:,j], c+'-')
                pt.plot(tpts[-1] + np.cumsum(durs), angs[:,j], c+'o:')
    
            pt.ylabel("Angle (deg)")
            pt.xlabel("Time (s)")
            pt.title(["Success","Fall"][i])
    
        pt.tight_layout()
        pt.savefig("example_chunks.eps")
        pt.show()
        
    if do_nearest:

        # load chunk data
        with open("stitch_chunks.pkl","rb") as f: (chunks, falls, stdevs) = pk.load(f)
    
        fig = pt.figure(figsize=(8,3))
    
        # do once for falls and once for stands in stdev0
        for sp, fall in enumerate((False, True)): # following code assumes True comes second, don't change
    
            # track nearest-neighbor distances
            nn_dists = []
    
            # track examples where a fall's nearest neighbor is a success and both are in stdev 0 (same action)?
            if fall: neg_idxs = []
    
            idxs0 = np.flatnonzero((stdevs == "0.0") & (falls == fall))
            for m, idx in enumerate(idxs0):
                (_, X_m), _ = chunks[idx]
                print(f"noiseless chunk {m} of {len(idxs0)} ({fall=})")
    
                # find nearest neighbor in all chunks
                # for now, just average element-wise joint difference truncated to equal length
                # (TODO: compare different-length observations based on timepoints)
                dists = []
                for c, ((_, X), _) in enumerate(chunks):
    
                    # skip self, not a neighbor
                    if c == idx:
                        dists.append(np.inf)
                        continue
    
                    # get distance
                    trunc = min(len(X_m), len(X))
                    dist = np.fabs(X_m[-trunc:] - X[-trunc:]).mean()
                    dists.append(dist)
    
                # nearest neighbor index
                dists = np.array(dists)
                n_idx = np.argmin(dists)
                nn_dists.append(dists[n_idx])
    
                # distribution of distances
                print(f"dists ~ {np.mean(dists[dists<np.inf])} +/- {np.std(dists[dists<np.inf])} >= {dists[n_idx]}")
                
                # save negative examples
                if fall and stdevs[n_idx] == "0.0" and falls[n_idx] == False:
                    print("found negative")
                    neg_idxs.append((idx, n_idx))
                elif fall:
                    print(f"found positive, stdev={stdevs[n_idx]}, fall={falls[n_idx]}")
    
            # plot nn_dist distribution
            pt.subplot(1,2,sp+1)
            pt.hist(nn_dists, edgecolor='k',facecolor='w')
            pt.title(f"Fall={fall}")
    
        print(f"{len(neg_idxs)} of {len(idxs0)} noiseless falls are negative examples")
    
        fig.supxlabel("Whole-data nearest-neighbor distance (MAD) from noiseless chunks")
        fig.supylabel("Count")
        pt.tight_layout()
        pt.savefig("nn_dists.eps")
        pt.show()
    
    if do_centered:
    
        with open("stitch_arrays.pkl","rb") as f:
            (obs_t, obs_p, obs_v, traj_t, traj_p, nfstep, stdevs) = pk.load(f)
    
        # get union of all buffer timepoints
        timepoints = set()
        durations = []
        for t in obs_t:
            timepoints.update(t)
            durations.append(t.max())
        print(f"{len(timepoints)} timepoints total")
    
        # pt.hist(durations)
        # pt.show()
    
        # linearly interpolate at evenly spaced timepoints
        num_timepoints = 13*100 # roughly 13 seconds, 100Hz
        timepoints = np.linspace(0, max(durations), num_timepoints)
        histories = np.empty((len(obs_t), num_timepoints, 25)) # 25 joints
        for r, (t, p) in enumerate(zip(obs_t, obs_p)):
            print(f"interpolating {r} of {len(durations)}...")
            for j in range(25):
                histories[r,:,j] = np.interp(timepoints, t, p[:,j])
    
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)
    
        mean_traj = histories[noiseless & successes].mean(axis=0)
        traj_dist = np.fabs(histories - mean_traj[None]).mean(axis=2)
    
        pt.plot(timepoints, traj_dist[noiseless & successes].T, '--', color=(.8,.8,1.))
        pt.plot(timepoints, traj_dist[noiseless & ~successes].T, '--', color=(1.,.8,.8))
        pt.plot(timepoints, traj_dist[~noiseless & successes].T, ':', color=(.8,.8,1.))
        pt.plot(timepoints, traj_dist[~noiseless & ~successes].T, ':', color=(1.,.8,.8))
        pt.plot(timepoints, traj_dist[noiseless & successes].mean(axis=0), '--', color='b', label="noiseless success")
        pt.plot(timepoints, traj_dist[noiseless & ~successes].mean(axis=0), '--', color='r', label="noiseless failure")
        pt.plot(timepoints, traj_dist[~noiseless & successes].mean(axis=0), ':', color='b', label="noisy success")
        pt.plot(timepoints, traj_dist[~noiseless & ~successes].mean(axis=0), ':', color='r', label="noisy failure")
        pt.xlabel("time")
        pt.ylabel("MAD from noiseless success mean")
        pt.legend()
        pt.show()

    if do_nearest_full:

        timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs = interpolate_arrays()
    
        # see if there is any waypoint within noiseless failures at which time the nearest success neighbor is noisy
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)
        for f_idx in np.flatnonzero(noiseless & ~successes):

            # check all waypoints up to fall
            fall_n = 5*nfstep[f_idx]
            for w, wp_time in enumerate(traj_t[1:fall_n]):

                # timesteps up to current waypoint
                t_mask = (timepoints <= wp_time)

                # check all successes
                s_idxs = np.flatnonzero(successes)
                p_dists = []
                v_dists = []
                for s_idx in s_idxs:

                    # joint history distance
                    p_dist = np.fabs(obs_ip[f_idx, t_mask] - obs_ip[s_idx, t_mask]).mean()
                    # vision history distance
                    v_dist = np.fabs(obs_iv[f_idx, t_mask] - obs_iv[s_idx, t_mask]).mean()

                    p_dists.append(p_dist)
                    v_dists.append(v_dist)

                # nearest joint neighbor for current waypoint
                p_nn_idx = s_idxs[np.argmin(p_dists)]
                v_nn_idx = s_idxs[np.argmin(v_dists)]

                if (not noiseless[p_nn_idx]) or (not noiseless[v_nn_idx]):
                    print(f"noiseless failure {f_idx}, waypoint {w+1} has noisy success nearest neighbor!! ({t_mask.sum()} timepts)")
                else:
                    print(f"noiseless failure {f_idx}, waypoint {w+1}")

    if do_stability:
        # see how many noisy success actions bring observations closer to noiseless mean
        
        timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs = interpolate_arrays()
    
        # calculate noiseless mean
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)

        obs_cp = obs_ip[noiseless & successes].mean(axis=0)
        obs_cv = obs_iv[noiseless & successes].mean(axis=0)

        # check each waypoint in each noisy success
        contracted = []
        # for idx in np.flatnonzero(~noiseless & successes):
        #     print(f"noisy success {idx}")
        # for idx in np.flatnonzero(noiseless & successes):
        #     print(f"noiseless success {idx}")
        for idx in np.flatnonzero(~noiseless & ~successes):
            print(f"noisy fail {idx}")
        # for idx in np.flatnonzero(noiseless & ~successes):
        #     print(f"noiseless fail {idx}")

            # check all waypoints
            # for n in range(1, len(traj_t)):
            for n in range(1, 5*nfstep[idx]):
                prev_mask = (timepoints <= traj_t[n-1])
                next_mask = (timepoints <= traj_t[n])

                # prev_dist = np.fabs(obs_ip[idx, prev_mask] - obs_cp[prev_mask]).mean()
                # next_dist = np.fabs(obs_ip[idx, next_mask] - obs_cp[next_mask]).mean()

                prev_dist = np.fabs(obs_iv[idx, prev_mask] - obs_cv[prev_mask]).mean()
                next_dist = np.fabs(obs_iv[idx, next_mask] - obs_cv[next_mask]).mean()

                contracted.append(prev_dist > next_dist)
                if prev_dist > next_dist:
                    print(f" waypoint {n} reduced distance")

        print(f"{np.mean(contracted)} portion contracted")

        # pt.plot(timepoints, obs_cp)
        # pt.show()

    if do_clone:
        # clone successful behavior, check conditioning and policy output on failed observations

        timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs = interpolate_arrays()
        num_runs = len(nfstep)
    
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)
        print(f"{successes.sum()} successes")

        # noiseless trajectory
        traj_0 = traj_p[np.flatnonzero(noiseless)[0]]
        # traj_0 = traj_p[0] # just when testing 10 runs

        s_res = []
        f_res = []
        a_res = []
        W_norm = []
        for n, wp_time in enumerate(traj_t):

            # timesteps up to current waypoint
            t_mask = (timepoints <= wp_time)
            # if n == 0:
            #     print(wp_time)
            #     print(timepoints[:10])
            #     print(t_mask[:10])
            #     print(obs_iv[0,0])
            #     input('.')

            # collect policy inputs up to current waypoint
            X_p = obs_ip[:, t_mask].reshape(num_runs, -1) # positions
            X_v = obs_iv[:, t_mask].reshape(num_runs, -1) # vision
            X_c = traj_p[:, :n].reshape(num_runs, -1) # commands so far
            X_b = np.ones((num_runs, 1)) # for bias
            X = np.concatenate([X_p, X_v, X_c, X_b], axis=1) # (num_runs, -1)

            # collect next waypoint commands
            A = traj_p[:, n] # (num_runs, 25)

            # best linear fit to success data
            W = np.linalg.lstsq(X[successes], A[successes], rcond=None)[0]
            W_norm.append((W**2).sum()**.5)

            # measure residual on successes
            s_res.append( np.fabs(X[successes] @ W - A[successes]).mean() )

            # measure residual on failures that haven't ended yet
            f_mask = ~successes & (n < 5*nfstep)
            f_res.append( np.fabs(X[f_mask] @ W - A[f_mask]).mean() )

            # measure largest distance from noiseless trajectory over all runs, timesteps, joints
            # print((X @ W).shape, traj_0[n].shape, (X @ W - traj_0[n]).shape)
            # input('.')
            a_res.append( np.fabs(X @ W - traj_0[n]).max() )

            print(f"waypoint {n}: {X[successes].shape}*{W.shape} = {A[successes].shape}, {W_norm[-1]=}, {s_res[-1]=}, {f_res[-1]=}, {a_res[-1]=}")

        pt.figure(figsize=(8,5))
        pt.plot(np.arange(30), s_res, 'bo-', label="success residual")
        pt.plot(np.arange(30), f_res, 'rx-', label="failure residual")
        pt.plot(np.arange(30), a_res, 'g^-', label="nominal residual")
        pt.plot(np.arange(30), W_norm, 'k-', label="W Fro norm")
        pt.yscale("log")
        pt.xlabel("waypoint")
        pt.legend()
        pt.tight_layout()
        pt.savefig("linclone.eps")
        pt.show()
        
    if do_returns:

        timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs = interpolate_arrays()
        num_runs = len(nfstep)
    
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)
        print(f"{successes.sum()} successes")

        # noiseless trajectory
        traj_0 = traj_p[np.flatnonzero(noiseless)[0]]

        # average starting joints
        central_0 = obs_ip[:,0].mean(axis=0)
        print(f"Variation in starting joints ~ {np.fabs(obs_ip[:,0] - central_0).mean()} <= {np.fabs(obs_ip[:,0] - central_0).max()}")

        # target initial joints (same at very end of trajectory, not perturbed)
        init_angs = traj_p[0,-1]
        print(f"Average starting from target initial joints ~ {np.fabs(obs_ip[:,0] - init_angs).mean()} <= {np.fabs(obs_ip[:,0] - init_angs).max()}")

        # plot each trajectory's distance to its starting point after each cycle
        cycle_times = traj_t[[10, 20]]
        cycle_timesteps = [0] + [max(np.flatnonzero(timepoints < t)) for t in cycle_times] + [len(timepoints)-1]
        print("cycle timesteps", cycle_timesteps)

        pt.figure(figsize=(10,5))

        pt.subplot(1,2,1)
        for r in range(num_runs):
            hi = (nfstep[r] // 2) + 1
            dists = np.fabs(obs_ip[r, cycle_timesteps[:hi]] - obs_ip[r, 0]).mean(axis=-1)
            c = 'r' if nfstep[r] < 6 else ('g' if noiseless[r] else 'b')
            pt.plot(timepoints[cycle_timesteps[:hi]], dists, c+'o-', alpha=.25)
        pt.ylabel("Mean joint delta from initial observation (deg)")

        pt.subplot(1,2,2)
        for r in range(num_runs):
            hi = (nfstep[r] // 2) + 1
            dists = np.fabs(obs_ip[r, cycle_timesteps[:hi]] - init_angs).mean(axis=-1)
            c = 'r' if nfstep[r] < 6 else ('g' if noiseless[r] else 'b')
            pt.plot(timepoints[cycle_timesteps[:hi]], dists, c+'o-', alpha=.25)
        pt.ylabel("Mean joint delta from initial command (deg)")

        pt.gcf().suptitle("Footstep cycle return distance")
        pt.gcf().supxlabel("Time (sec)")
        pt.tight_layout()
        pt.savefig("return.pdf")
        pt.show()
            

        # for
        # obs_ip[noiseless & success, ::10

        # # distances to nominal at very beginning
        # nom_0 = obs_ip[noiseless & successes,0].mean(axis=0)
        # print(nom_0)

        # print(f"noiseless succ <= {np.fabs(obs_ip[noiseless & successes, 0] - nom_0).max(axis=1).mean()}")
        # print(f"noisy succ <= {np.fabs(obs_ip[~noiseless & successes, 0] - nom_0).max(axis=1).mean()}")
        # print(f"noiseless fall <= {np.fabs(obs_ip[noiseless & ~successes, 0] - nom_0).max(axis=1).mean()}")
        # print(f"noisy fall <= {np.fabs(obs_ip[~noiseless & ~successes, 0] - nom_0).max(axis=1).mean()}")

    if do_lindyn:
        # see if per-time linear models can fit the cycle transition dynamics

        timepoints, obs_ip, obs_iv, traj_t, traj_p, nfstep, stdevs = interpolate_arrays()
        num_runs = len(nfstep)
    
        noiseless = (stdevs == "0.0")
        successes = (nfstep == 6)
        print(f"{successes.sum()} successes")

        # noiseless trajectory
        traj_0 = traj_p[np.flatnonzero(noiseless)[0]]

        # time steps at cycle boundaries
        cycle_times = traj_t[[10, 20]]
        cycle_timesteps = [0] + [max(np.flatnonzero(timepoints < t)) for t in cycle_times] + [len(timepoints)-1]
        print("cycle timesteps", cycle_timesteps)

        # collect transition data from all successful cycles
        X0, A, X1 = [], [], []
        for r in range(num_runs):
            # last successful cycle in this run
            hi = (nfstep[r] // 2) + 1
            for c in range(hi-1):
                X0.append(obs_ip[r, cycle_timesteps[c]].flatten())
                A.append(traj_p[r,10*c:10*(c+1)].flatten())
                X1.append(obs_ip[r, cycle_timesteps[c+1]].flatten())

        # try fitting the linear model
        X0 = np.stack(X0)
        A = np.stack(A)
        X1 = np.stack(X1)
        X0A = np.concatenate([X0, A, np.ones((len(X0),1))], axis=1)

        W = np.linalg.lstsq(X0A, X1, rcond=None)[0]
        print(f"{X0A.shape}*{W.shape} ~ {X1.shape}")

        print(f"W Fro = {(W**2).sum()**.5}")
        print(f"res ~ {np.fabs(X0A @ W - X1).max(axis=1).mean()} <= {np.fabs(X0A @ W - X1).max()}")

