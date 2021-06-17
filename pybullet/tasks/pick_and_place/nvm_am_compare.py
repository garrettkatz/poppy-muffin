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

    num_bases, max_levels = 4, 3
    result_file = "results_compare_%d_%d.pkl" % (num_bases, max_levels)

    run_exp = True
    if run_exp:
        all_results = {}
        num_reps = 2
        block_counts = [3] # list(range(3, 8))
        for num_blocks in block_counts:
    
            all_results[num_blocks] = {}
            for rep in range(num_reps):
    
                am_results, nvm_results, nvm_size = run_trial(num_bases, num_blocks, max_levels)
                all_results[num_blocks][rep] = am_results, nvm_results, nvm_size
    
                with open(result_file, "wb") as f: pk.dump(all_results, f)
    
                print("%d blocks, rep %d of %d:" % (num_blocks, rep, num_reps))
                print(" am:", all_results[num_blocks][rep][0])
                print(" nvm:", all_results[num_blocks][rep][1])
                # print(" size:")
                # for sz in nvm_size: print(sz) 
                print(" nvm size: %d" % nvm_size[0])

    plt_exp = True
    if plt_exp:

        with open(result_file, "rb") as f: all_results = pk.load(f)
    
        for num_blocks in sorted(all_results.keys()):
            num_reps = len(all_results[num_blocks])
            for rep in range(num_reps):
                print("%d blocks, rep %d of %d:" % (num_blocks, rep, num_reps))
                print(" am:", all_results[num_blocks][rep][0])
                print(" nvm:", all_results[num_blocks][rep][1])
                # print(" size:")
                # for sz in nvm_size: print(sz) 
                print(" nvm size: %d" % nvm_size[0])


