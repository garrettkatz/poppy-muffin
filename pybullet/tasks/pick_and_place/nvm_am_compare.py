import torch as tr
import sys, time
import pickle as pk
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv, random_thing_below
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize
from restack import compute_spatial_reward, compute_symbolic_reward

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
    while True:
        thing_below = random_thing_below(num_blocks, max_levels, num_bases)
        goal_thing_below = random_thing_below(num_blocks, max_levels, num_bases)
        env.load_blocks(thing_below, num_bases)    
        if compute_symbolic_reward(env, goal_thing_below) < 0: break
        env.reset()

    am = make_abstract_machine(env, num_bases, max_levels)
    nvm = virtualize(am)

    am_results = run_machine(am, goal_thing_below, {"jnt": "rest"})
        
    env.reset()
    env.load_blocks(thing_below, num_bases)

    nvm_results = run_machine(nvm, goal_thing_below, {"jnt": tr.tensor(am.ik["rest"]).float()})

    env.close()
    
    return am_results, nvm_results, nvm.size()

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
        
        # scatter plots
        pt.figure(figsize=(3,8))
        for m, metric in enumerate(["Ticks", "Runtime (s)", "Symbolic reward", "Spatial reward"]):
            pt.subplot(4,1,m+1)
            for num_blocks in reversed(block_counts):
                x = [am_result[m] for am_result, nvm_result, nvm_size in all_results[num_blocks].values()]
                y = [nvm_result[m] for am_result, nvm_result, nvm_size in all_results[num_blocks].values()]
                if m == 2:
                    x = 0.1*np.random.rand(len(x)) + x
                    y = 0.1*np.random.rand(len(y)) + y
                pt.plot(x, y, '.', label="%d blocks" % num_blocks)
            if m == 3: pt.xlabel("VM")
            pt.ylabel("NVM")
            if m == 0: pt.legend(loc="upper left")
            pt.title(metric)

        pt.tight_layout()
        pt.show()
