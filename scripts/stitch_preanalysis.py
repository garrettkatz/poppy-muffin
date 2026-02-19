import os, glob
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

do_chunk = False
view_chunk = False
do_nearest = True

if do_chunk:

    # collect all run filenames
    run_filepaths = [] # each element is (traj_name, buf_name, stdev)
    
    path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", "*stdev*")
    
    folders = glob.glob(path)
    folder_stdevs = []
    for folder in folders:
        stdev = folder[folder.index("stdev")+5:]
        traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
        for traj_file in traj_files:
            buf_file = traj_file.replace("traj", "bufs")
            run_filepaths.append( (traj_file, buf_file, stdev) )
    
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

# load chunk data
with open("stitch_chunks.pkl","rb") as f: (chunks, falls, stdevs) = pk.load(f)

if view_chunk:

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
