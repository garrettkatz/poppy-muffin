import torch as tr
import numpy as np
import pickle as pk
import sys
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import compute_symbolic_reward, random_problem_instance
from abstract_machine import make_abstract_machine, memorize_env
from nvm_am_compare import run_trial, run_machine
from nvm import virtualize
import neural_virtual_machine as nv

def find_failure_case(num_bases, num_blocks, max_levels):
    while True:
        # print("trying..")
        env = BlocksWorldEnv(show=False)
        thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)    
        am = make_abstract_machine(env, num_bases, max_levels)
        am_results = run_machine(am, goal_thing_below, {"jnt": "rest"})
        env.close()
        ticks, running_time, sym_reward, spa_reward = am_results        
        if sym_reward <= -2: break
        # print(sym_reward)
    return thing_below, goal_thing_below, sym_reward

if __name__ == "__main__":

    num_bases, num_blocks, max_levels = 5, 5, 3
    
    find_new = False
    if find_new:
        thing_below, goal_thing_below, _ = find_failure_case(num_bases, num_blocks, max_levels)    
        print(thing_below)
        print(goal_thing_below)
    
    else:
        # one failure case:
        thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
        goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
    
    run_case = True
    if run_case:
        class Tracker:
            def __init__(self, goal_thing_below):
                self.mp = []
                self.sym = []
                self.goal_thing_below = goal_thing_below
            def reset(self):
                self.mp = []
                self.sym = []
            def step_hook(self, env, action):
                self.mp.append(env.movement_penalty())
                self.sym.append(compute_symbolic_reward(env, self.goal_thing_below))
        # load
        tracker = Tracker(goal_thing_below)
        env = BlocksWorldEnv(show=False, step_hook = tracker.step_hook)
        env.load_blocks(thing_below)
        # run rvm
        rvm = make_abstract_machine(env, num_bases, max_levels, gen_regs=["r0", "r1"])
        nvm = virtualize(rvm, nv.default_activator)
        # run
        goal_thing_above = env.invert(goal_thing_below)
        for key, val in goal_thing_above.items():
            if val == "none": goal_thing_above[key] = "nil"
        memorize_env(rvm, goal_thing_above)
        rvm.reset({"jnt": "rest"})
        rvm.mount("main")
        while True:
            done = rvm.tick()
            if done: break
        # run_machine(rvm, goal_thing_below, reset_dict={"jnt": "rest"})
        num_time_steps = rvm.tick_counter
        env.close()
        # save
        with open("fcase_batched_data.pkl", "wb") as f: pk.dump((tracker.mp, tracker.sym), f)
        
        # run batched nvm
        nvm.mount("main")
        nvm.registers["jnt"].content = tr.tensor(rvm.ik["rest"]).float()

        W_init = {name: {0: nvm.net.batchify_weights(conn.W)} for name, conn in nvm.connections.items()}
        v_init = {name: {0: nvm.net.batchify_activities(reg.content)} for name, reg in nvm.registers.items()}

        memorize_env(nvm, goal_thing_above)
        for name in ["obj","loc","goal"]:
            W_init[name][0] = nvm.net.batchify_weights(nvm.connections[name].W)

        # test batching
        for W in W_init.values(): W[0] = W[0].repeat(8,1,1)
        for v in v_init.values(): v[0] = v[0].repeat(8,1,1)

        W, v = nvm.net.run(W_init, v_init, num_time_steps)

        env = BlocksWorldEnv(show=True, step_hook = tracker.step_hook)
        env.load_blocks(thing_below)
        tar = nvm.registers["tar"]
        for t in range(2,num_time_steps):
            if tar.decode(v["tar"][t-2][0,:,0]) != tar.decode(v["tar"][t-1][0,:,0]):
                env.goto_position(v["jnt"][t][0,:,0].detach().numpy())

    plot_case = False
    if plot_case:
    
        with open("fcase_batched_data.pkl", "rb") as f: (mp, sym) = pk.load(f)
        
        print("Sym reward = %f" % sym[-1])
        max_mp = max(mp)
        
        import matplotlib.pyplot as pt
        # import matplotlib.axes as ax
        pt.figure(figsize=(5,2))
        ax = pt.gca()
        ax2 = ax.twinx()
        # ax = pt.subplot(2,1,1)
        # ax2 = pt.subplot(2,1,2)
        h2 = ax2.plot(np.cumsum(mp), 'k--', label="cumulative")
        h1 = ax.plot(mp, 'k-', label="Movement penalty")
        ax.set_xlabel("Simulation steps")
        ax.set_ylabel("Movement penalty")
        ax2.set_ylabel("Cumulative")
        # pt.legend([h1, h2], ["Movement penalty", "Cumulative"])
        pt.tight_layout()
        pt.savefig("movement_penalties_batched.eps")
        pt.show()
