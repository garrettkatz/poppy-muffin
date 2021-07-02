import numpy as np
import pickle as pk
import sys
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import compute_symbolic_reward, random_problem_instance
from abstract_machine import make_abstract_machine, memorize_env
from nvm_am_compare import run_trial, run_machine
import block_stacking_problem as bp

def find_failure_case(env, domain, sym_cutoff=-1):
    while True:
        # print("trying..")
        problem = domain.random_problem_instance()
        env.reset()
        env.load_blocks(problem.thing_below)
        am = make_abstract_machine(env, domain.num_bases, domain.max_levels)
        am_results = run_machine(am, problem.goal_thing_below, {"jnt": "rest"})
        ticks, running_time, sym_reward, spa_reward = am_results        
        if sym_reward <= sym_cutoff: break
        # print(sym_reward)
    env.reset()
    return problem, sym_reward

if __name__ == "__main__":

    num_bases, num_blocks, max_levels = 5, 5, 3
    domain = bp.BlockStackingDomain(num_bases, num_blocks, max_levels)
    
    find_new = True
    if find_new:
        env = BlocksWorldEnv(show=False)
        problem, _ = find_failure_case(env, domain)
        env.close()
        thing_below = problem.thing_below
        goal_thing_below = problem.goal_thing_below
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
        env = BlocksWorldEnv(show=True, step_hook = tracker.step_hook, timestep=1./200)
        env.load_blocks(thing_below)    
        # run rvm
        rvm = make_abstract_machine(env, num_bases, max_levels)
        # run
        goal_thing_above = env.invert(goal_thing_below)
        for key, val in goal_thing_above.items():
            if val == "none": goal_thing_above[key] = "nil"
        memorize_env(rvm, goal_thing_above)
        rvm.reset({"jnt": "rest"})
        rvm.mount("main")
        while True:
            done = rvm.tick()
            if rvm.registers["jnt"].content != rvm.registers["jnt"].old_content:
                position = rvm.ik[rvm.registers["jnt"].content]
                # input('.')
                rvm.env.goto_position(position, speed=1.5)
            if done: break
        # run_machine(rvm, goal_thing_below, reset_dict={"jnt": "rest"})
        # save
        with open("fcase_data.pkl", "wb") as f: pk.dump((tracker.mp, tracker.sym), f)

    plot_case = True
    if plot_case:
    
        with open("fcase_data.pkl", "rb") as f: (mp, sym) = pk.load(f)
        
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
        pt.savefig("movement_penalties.eps")
        pt.show()
