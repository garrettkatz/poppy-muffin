import os, glob
import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

# do once for stdev=0 and then once for all
fig = pt.figure(figsize=(8,3))

for i, suffix in enumerate(["0.0", "*"]):

    # histogram number of successful steps in the stdev 0 walk data
    num_success_data = []
    path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", "*stdev" + suffix)
    folders = glob.glob(path)
    buf_files = []
    for folder in folders:
        traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
        for traj_file in traj_files:
            # filename format is traj_<n>_<num_success>.pkl for the nth episode
            num_successes = int(traj_file[traj_file.rfind("_")+1:-4])
            num_success_data.append(num_successes)
    
            # save buf file paths for loading later
            buf_file = traj_file.replace("traj", "bufs")
            buf_files.append( buf_file )
    
    print(f"stdev{suffix}: {len(buf_files)} runs")
   
    pt.subplot(1,2,i+1)
    counts, bins, bars = pt.hist(num_success_data, bins=np.arange(8)-.5, edgecolor='k', facecolor='none') # 6 is the maximum number of footsteps, include the bin [5.5,6.5]
    
    percentages = [f'{(c/len(num_success_data) * 100):.1f}%' for c in counts]
    pt.bar_label(bars, labels=percentages, label_type='center', color='k', fontweight='bold')
    pt.title("Noiseless" if suffix == "0.0" else "All")
    
fig.supxlabel("Number of successful steps before fall")
fig.supylabel("Count")

pt.tight_layout()
pt.savefig("controllability.eps")
pt.show()
