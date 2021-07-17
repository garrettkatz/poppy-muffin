import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

num_bases, max_levels = 7, 3
# result_file = "results_compare_%d_%d.pkl" % (num_bases, max_levels)
result_file = "nvm_rvm_compare_%d_%d.pkl" % (num_bases, max_levels)
with open(result_file, "rb") as f: all_results = pk.load(f)

block_counts = sorted(all_results.keys())
for num_blocks in block_counts:
    num_reps = len(all_results[num_blocks])
    for rep in range(num_reps):
        print("%d blocks, rep %d of %d:" % (num_blocks, rep, num_reps))
        print(" am:", all_results[num_blocks][rep][0])
        print(" nvm:", all_results[num_blocks][rep][1])
        nvm_size = all_results[num_blocks][rep][2]
        # print(" size:")
        # for sz in nvm_size: print(sz)
        print(" nvm size: %d" % nvm_size[2])

# reorganize by metric
# metrics[metric][num_blocks][am/nvm][rep]
metrics = {}
for m,met in enumerate(["ticks","time","sym","spa"]):
    metrics[met] = {}
    for num_blocks in block_counts:
        metrics[met][num_blocks] = {}
        for v,mach in enumerate(["rvm","nvm"]):
            metrics[met][num_blocks][mach] = []
            for rep in all_results[num_blocks].values():
                metrics[met][num_blocks][mach].append(rep[v][m])

# sym violin plot
fig = pt.figure(figsize=(10,2.5))
pt.subplot(1,2,1)
block_counts = list(sorted(metrics["sym"].keys()))
positions = np.arange(len(block_counts))
handles = [0,0]
for m,mach in enumerate(["rvm", "nvm"]):
    x = [-np.array(metrics["sym"][num_blocks][mach]) for num_blocks in block_counts]
    parts = pt.violinplot(x, positions=2*positions + m, widths=.9)
    shade = [.75, .25][m]
    for pc in parts["bodies"]:
        pc.set_facecolor((shade,)*3)
        handles[m] = pc
        pc.set_edgecolor('k')
        pc.set_alpha(1)
    for key in ["cmins","cmaxes","cbars"]:
        parts[key].set_edgecolor('k')
        parts[key].set_alpha(1)
pt.ylabel("Symbolic distance", fontsize=12)
# pt.yticks([])
pt.xticks(.5 + 2*positions, block_counts)
pt.xlabel("Number of blocks in problem instance", fontsize=12)
pt.yticks(range(len(block_counts)+1))
pt.legend(handles, ["RVM", "NVM"], loc='upper left')
pt.title("(A)", fontsize=12)

# mp plot
with open("fcase_data.pkl", "rb") as f: (mp, sym) = pk.load(f)
mp = mp[10:]
ax = pt.subplot(1,2,2)
ax2 = ax.twinx()
h2 = ax2.plot(np.cumsum(mp), 'k--', label="cumulative")
h1 = ax.plot(mp, 'k-', label="Movement penalty")
ax.set_xlabel("Simulation steps", fontsize=12)
ax.set_ylabel("Movement penalty", fontsize=12)
ax2.set_ylabel("Cumulative", fontsize=12)
pt.title("(B)", fontsize=12)

pt.tight_layout()
pt.savefig("sym_mp.pdf")
pt.show()
pt.close()



