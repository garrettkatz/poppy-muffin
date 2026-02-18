import os, glob
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

do_chunk = False
view_chunk = True

if do_chunk:

    # collect all run filenames
    run_filepaths = [] # each element is (traj_name, buf_name)
    
    path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", "*stdev*")
    
    folders = glob.glob(path)
    for folder in folders:
        traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
        for traj_file in traj_files:
            buf_file = traj_file.replace("traj", "bufs")
            run_filepaths.append( (traj_file, buf_file) )
    
    print(f"{len(run_filepaths)} runs total")
    
    # use observation windows of previous footstep as policy input, and planned trajectory of next footstep as action
    # chunk the data around each (prev, next) footstep boundary, up until a fall or end of run (zero-based indexing)
    chunks = [] # elements are [prev footstep obs, next footstep traj] = [(timepoints, joint obs), (durations, waypoints)]
    labels = [] # True if next footstep is fall, False otherwise

    for (traj_file, buf_file) in run_filepaths:
    
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
            labels.append((next_footstep == last_footstep) and (num_successes < 6))
    
            print(f"   fall={labels[-1]}, boundary (prev={next_footstep-1}|next={next_footstep}) of {num_successes}: {len(jpos)} obs window")

    # save chunk data
    labels = np.array(labels)
    with open("stitch_chunks.pkl","wb") as f: pk.dump((chunks, labels), f)

# load chunk data
with open("stitch_chunks.pkl","rb") as f: (chunks, labels) = pk.load(f)

if view_chunk:

    colors = 'bgrcmyk' * 4

    # # first chunk
    # (tpts, jobs), (durs, angs) = chunks[0]
    # for j,c in enumerate(colors[:25]):
    #     pt.plot(tpts, jobs[:,j], c+'-')
    #     pt.plot(tpts[-1] + np.cumsum(durs), angs[:,j], c+'o:')
    # pt.show()

    # random success/fall chunks
    s_idx = np.random.choice(np.flatnonzero(labels == False))
    f_idx = np.random.choice(np.flatnonzero(labels == True))

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
    pt.show()
    
