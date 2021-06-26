import torch as tr
import sys, time
import pickle as pk
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv, random_thing_below
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize
from restack import compute_spatial_reward, compute_symbolic_reward, random_problem_instance

def run_machine(machine, goal_thing_below, reset_dict):

    goal_thing_above = machine.env.invert(goal_thing_below)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"

    start = time.perf_counter()
    memorize_env(machine, goal_thing_above)
    machine.reset(reset_dict)
    ticks = machine.run()
    running_time = time.perf_counter() - start    

    sym_reward = compute_symbolic_reward(machine.env, goal_thing_below)
    spa_reward = compute_spatial_reward(machine.env, goal_thing_below)
    
    return ticks, running_time, sym_reward, spa_reward

def run_trial(num_bases, num_blocks, max_levels):

    env = BlocksWorldEnv(show=False)

    # rejection sample non-trivial instance
    thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)

    am = make_abstract_machine(env, num_bases, max_levels)
    nvm = virtualize(am)

    am_results = run_machine(am, goal_thing_below, {"jnt": "rest"})
        
    env.reset()
    env.load_blocks(thing_below, num_bases)

    nvm_results = run_machine(nvm, goal_thing_below, {"jnt": tr.tensor(am.ik["rest"]).float()})

    env.close()
    
    return am_results, nvm_results, nvm.size(), thing_below, goal_thing_below

if __name__ == "__main__":

    num_bases, max_levels = 7, 3
    result_file = "results_compare_%d_%d.pkl" % (num_bases, max_levels)

    # 7 bases, 7 blocks ~ 12 seconds
    # <12s * 5 block counts * reps = time
    # reps = time / 60s = time minutes (1 minute per rep)
    run_exp = False
    if run_exp:
        num_reps = 60 * 5
        block_counts = list(range(3, 8))
        all_results = {num_blocks: {} for num_blocks in block_counts}
        for rep in range(num_reps):
            for num_blocks in block_counts:
    
                am_results, nvm_results, nvm_size = run_trial(num_bases, num_blocks, max_levels)
                all_results[num_blocks][rep] = am_results, nvm_results, nvm_size
    
                with open(result_file, "wb") as f: pk.dump(all_results, f)
    
                print("%d blocks, rep %d of %d:" % (num_blocks, rep, num_reps))
                print(" am:", all_results[num_blocks][rep][0])
                print(" nvm:", all_results[num_blocks][rep][1])
                # print(" size:")
                # for sz in nvm_size: print(sz) 
                print(" nvm size: %d" % nvm_size[2])

    plt_exp = True
    if plt_exp:

        import numpy as np
        import matplotlib.pyplot as pt
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

        # tick scatterplot
        fig = pt.figure(figsize=(6,2.35))
        gs = fig.add_gridspec(1,3)
        # pt.subplot(1,2,1)
        fig.add_subplot(gs[0,0])
        for n,num_blocks in enumerate(metrics["ticks"].keys()):
            x = metrics["ticks"][num_blocks]["rvm"]
            y = metrics["ticks"][num_blocks]["nvm"]
            # shade = [.4, .3, .2, .1, 0][n]
            shade = 0
            pt.scatter(x, y, ec=(shade,)*3, fc="none", marker="o")
        pt.xticks([600, 900, 1200])
        pt.yticks([600, 900, 1200])
        pt.xlabel("RVM ticks", fontsize=12)
        pt.ylabel("NVM ticks", fontsize=12)

        # tick boxplot
        fig.add_subplot(gs[0,1:])
        # pt.subplot(1,2,2)
        x = [metrics["ticks"][num_blocks]["nvm"] for num_blocks in metrics["ticks"]]
        positions = list(sorted(metrics["ticks"].keys()))
        pt.boxplot(x, positions=positions, medianprops={"c": "k"})
        # pt.ylabel("Ticks")
        pt.yticks([])
        pt.xlabel("Blocks", fontsize=12)
        
        pt.tight_layout()
        pt.savefig("tick_compare.eps")
        # pt.show()
        pt.close()

        # clock rates
        for mach in ["rvm", "nvm"]:
            runtimes = [rep for num_blocks in range(3,8)
                for rep in metrics["time"][num_blocks][mach]]
            numticks = [rep for num_blocks in range(3,8)
                for rep in metrics["ticks"][num_blocks][mach]]
            print("%s clock rate: %f Hz" % (mach, np.mean(np.array(numticks)/np.array(runtimes))))

        # runtime scatterplot
        fig = pt.figure(figsize=(6,2.35))
        gs = fig.add_gridspec(1,5)
        fig.add_subplot(gs[0,:2])
        for n,num_blocks in enumerate(reversed(sorted(metrics["time"].keys()))):
            x = metrics["time"][num_blocks]["rvm"]
            y = metrics["time"][num_blocks]["nvm"]
            shade = np.linspace(0, .7, 5)[n]
            # shade = 0
            pt.scatter(x, y, fc=(shade,)*3, ec="none", marker="o", label=str(num_blocks))
        # pt.xticks([600, 900, 1200])
        # pt.yticks([600, 900, 1200])
        pt.legend()
        pt.xlabel("RVM runtime (s)", fontsize=12)
        pt.ylabel("NVM runtime (s)", fontsize=12)

        # runtime histogram
        fig.add_subplot(gs[0,2:])
        for m,mach in enumerate(["rvm","nvm"]):
            x = [rep for num_blocks in metrics["time"]
                    for rep in metrics["time"][num_blocks][mach]]
            pt.hist(x, ec="k", fc=["w",(.5,)*3][m], bins=np.linspace(0,10,20), label=mach.upper())
        pt.ylabel("Frequency", fontsize=12)
        pt.xlabel("Runtime (s)", fontsize=12)
        pt.legend()
        
        pt.tight_layout()
        pt.savefig("time_compare.eps")
        # pt.show()
        pt.close()

        # sym box plot
        fig = pt.figure(figsize=(6,2.35))
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
        
        pt.tight_layout()
        # pt.savefig("sym_compare.eps")
        # pt.savefig("sym_compare.png")
        pt.savefig("sym_compare.pdf")
        # pt.show()
        pt.close()

        # # spa box plot
        # fig = pt.figure(figsize=(6,2.35))
        # block_counts = list(sorted(metrics["spa"].keys()))
        # positions = np.arange(len(block_counts))
        # handles = [0,0]
        # for m,mach in enumerate(["rvm", "nvm"]):
        #     x = [-np.array(metrics["spa"][num_blocks][mach]) for num_blocks in block_counts]
        #     parts = pt.violinplot(x, positions=2*positions + m, widths=.9)
        #     shade = [.75, .25][m]
        #     for pc in parts["bodies"]:
        #         pc.set_facecolor((shade,)*3)
        #         handles[m] = pc
        #         pc.set_edgecolor('k')
        #         pc.set_alpha(1)
        #     for key in ["cmins","cmaxes","cbars"]:
        #         parts[key].set_edgecolor('k')
        #         parts[key].set_alpha(1)
        # pt.ylabel("Spatial distance", fontsize=12)
        # # pt.yticks([])
        # pt.xticks(.5 + 2*positions, block_counts)
        # pt.xlabel("Number of blocks in problem instance", fontsize=12)
        # pt.yticks(range(len(block_counts)+1))
        # pt.legend(handles, ["RVM", "NVM"], loc='upper left')
        
        # pt.tight_layout()
        # # pt.savefig("spa_compare.eps")
        # pt.savefig("spa_compare.png")
        # # pt.show()
        # pt.close()

        # spa scatterplot
        fig = pt.figure(figsize=(3,3.25))
        gs = fig.add_gridspec(1,1)
        fig.add_subplot(gs[0,0])
        for n,num_blocks in enumerate(reversed(sorted(metrics["spa"].keys()))):
            x = -np.array(metrics["spa"][num_blocks]["rvm"])
            y = -np.array(metrics["spa"][num_blocks]["nvm"])
            shade = np.linspace(0, .7, 5)[n]
            # shade = 0
            pt.scatter(x, y, fc=(shade,)*3, ec="none", marker="o", label=str(num_blocks))
        pt.xlim([-.2, 8.5])
        pt.ylim([-.2, 6.5])
        pt.xticks(np.arange(7))
        pt.yticks(np.arange(7))
        pt.legend(loc='lower right')
        pt.xlabel("RVM spatial distance", fontsize=12)
        pt.ylabel("NVM spatial distance", fontsize=12)
        # pt.axis("equal")
        pt.tight_layout()
        pt.savefig("spa_compare.eps")
        pt.savefig("spa_compare.png")
        pt.show()
        pt.close()

        # # scatter plots
        # pt.figure(figsize=(3,8))
        # for m, metric in enumerate(["Ticks", "Runtime (s)", "Symbolic reward", "Spatial reward"]):
        #     pt.subplot(4,1,m+1)
        #     for num_blocks in reversed(block_counts):
        #         x = [am_result[m] for am_result, nvm_result, nvm_size in all_results[num_blocks].values()]
        #         y = [nvm_result[m] for am_result, nvm_result, nvm_size in all_results[num_blocks].values()]
        #         if m == 2:
        #             x = 0.1*np.random.rand(len(x)) + x
        #             y = 0.1*np.random.rand(len(y)) + y
        #         pt.plot(x, y, '.', label="%d blocks" % num_blocks)
        #     if m == 3: pt.xlabel("VM")
        #     pt.ylabel("NVM")
        #     if m == 0: pt.legend(loc="upper left")
        #     pt.title(metric)

        # pt.tight_layout()
        # pt.show()
