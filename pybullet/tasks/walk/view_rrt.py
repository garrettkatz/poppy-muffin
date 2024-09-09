import matplotlib.pyplot as pt
import numpy as np

# the five performance metrics for each sample
# progress is base displacement along walking direction (negative y-axis)
# lateral is base displacement perpendicular to walking direction (x axis)
# spin is angular displacement in base orientation
# velocity is base speed at the end of the motion
# jerk is the non-smoothness of the joint trajectory
objective_names = ("progress", "lateral", "spin", "velocity", "jerk")

# duration of each time step in trajectories
time_step = 0.2

# load the data
npz = np.load("rrt_trajs.npz")
trajs = npz["trajs"] # trajs[n,t,i]: joint angle i at timestep t in episode k
falls = npz["falls"] # falls[n] = True if episode k was a fall
objectives = npz["objectives"] # objectives[n,:]: various performance metrics for episode k

# normalize all jerk values so maximum is 1.0
objectives[:,-1] *= 1. / objectives[:,-1].max()

# print label imbalance
print(f"{falls.sum()} falls + {(~falls).sum()} successes = {len(falls)} trajs")

# filter to high quality objective values
filt = (objectives[:,1] < .005) & (objectives[:,2] < .01) & (objectives[:,3] < .002)
trajs = trajs[filt]
objectives = objectives[filt]
falls = falls[filt]

# filter out the dominated samples
dominated = np.empty(len(trajs), dtype=bool)
for n in range(len(trajs)):
    dominated[n] = (objectives < objectives[n]).all(axis=1).any()
trajs = trajs[~dominated]
objectives = objectives[~dominated]
falls = falls[~dominated]

# sort by objectives lexicographically
lex = np.lexsort(objectives.T[::-1])

# view histogram of objectives
fig, axs = pt.subplots(1, len(objective_names), figsize=(len(objective_names)*3, 3))
for i, (name, ax) in enumerate(zip(objective_names, axs)):
    ax.hist(objectives[:,i])
    ax.set_xlabel(name)
axs[0].set_ylabel("Frequency")
pt.tight_layout()

# view jerk-progress trade-offs
pt.figure()
pt.plot(objectives[:,0], objectives[:,4], 'k.')
pt.xlabel("progress")
pt.ylabel("jerk")

# view one of the best trajectories and a random one
pt.figure()
pt.subplot(1,2,1)
pt.title("Best")
pt.plot(time_step * np.arange(trajs.shape[1]), trajs[lex[0]])
pt.xlabel("Time")
pt.ylabel("Angle (rad)")
pt.subplot(1,2,2)
pt.title("Random")
pt.plot(time_step * np.arange(trajs.shape[1]), trajs[np.random.choice(lex)])
pt.xlabel("Time")
pt.show()

