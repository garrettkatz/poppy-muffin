import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from abstract_machine import make_abstract_machine, memorize_env
from restack import invert, compute_spatial_reward, compute_symbolic_reward, random_problem_instance
from practice_failure_case import PenaltyTracker, calc_reward
from nvm import virtualize

def run_episode(env, thing_below, goal_thing_below, goal_thing_above, nvm, init_regs, init_conns, penalty_tracker, sigma=0):
    
    # reload blocks
    env.reset()
    env.load_blocks(thing_below)

    # reset nvm, input new env, mount main program
    nvm.reset_state(init_regs, init_conns)
    memorize_env(nvm, goal_thing_above)
    nvm.mount("main")

    log_prob = 0.0 # accumulate over episode
    log_probs, rewards = [], []

    dbg = False
    if dbg: nvm.dbg()
    target_changed = False
    while True:
        done = nvm.tick() # reliable if core is not trained
        if dbg: nvm.dbg()
        # if nvm.tick_counter % 100 == 0: print("     tick %d" % nvm.tick_counter)
        if target_changed:
            mu = nvm.registers["jnt"].content
            if sigma > 0:
                dist = tr.distributions.normal.Normal(mu, sigma)
                position = dist.sample()
                log_probs.append(dist.log_prob(position).sum()) # multivariate white noise
                log_prob += log_probs[-1]
            else:
                position = mu

            penalty_tracker.reset()
            # nvm.dbg()
            # print("       pos:", position.detach().numpy())
            nvm.env.goto_position(position.detach().numpy())
            rewards.append(-penalty_tracker.penalty)
            # print("net penalty: %.5f" % penalty_tracker.penalty)
            # input('...')

        tar = nvm.registers["tar"]
        # decode has some robustness to noise even if tar connections are trained
        target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
        if done: break
    
    if len(rewards) == 0: # target never changed
        mu = nvm.registers["jnt"].content
        dist = tr.distributions.normal.Normal(mu, 0.001)
        log_probs.append(dist.log_prob(mu).sum()) # multivariate white noise
        rewards = [-10]
    
    sym_reward = compute_symbolic_reward(nvm.env, goal_thing_below)
    spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
    end_reward = calc_reward(sym_reward, spa_reward)
    rewards[-1] += end_reward
    
    return end_reward, log_prob, rewards, log_probs

if __name__ == "__main__":

    max_levels = 3
    num_blocks = 5
    num_bases = 5
    
    for rep in range(10):
    
        penalty_tracker = PenaltyTracker()
        env = BlocksWorldEnv(show=False, step_hook = penalty_tracker.step_hook)
        
        thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)
        goal_thing_above = invert(goal_thing_below, num_blocks, num_bases)
        for key, val in goal_thing_above.items():
            if val == "none": goal_thing_above[key] = "nil"
        
        σ1 = tr.tensor(1.).tanh()
        def σ(v): return tr.tanh(v) / σ1
        
        # set up rvm and virtualize
        rvm = make_abstract_machine(env, num_bases, max_levels)
        memorize_env(rvm, goal_thing_above)
        rvm.reset({"jnt": "rest"})
        rvm.mount("main")
        
        nvm = virtualize(rvm, σ)
        init_regs, init_conns = nvm.get_state()
        init_regs["jnt"] = tr.tensor(rvm.ik["rest"]).float()
        
        init_sym, _, init_rewards, _ = run_episode(env, thing_below, goal_thing_below, goal_thing_above, nvm, init_regs, init_conns, penalty_tracker, sigma=0)
        
        learning_rate = .000075
        rep = 4
        with open("stack_trained/pfc_%f_state_%d.pkl" % (learning_rate, rep), "rb") as f: (trained_regs, trained_conns) = pk.load(f)
        
        trained_sym, _, trained_rewards, _ = run_episode(env, thing_below, goal_thing_below, goal_thing_above, nvm, trained_regs, trained_conns, penalty_tracker, sigma=0)
        
        print("sym (init, trained): %f[%f], %f[%f]" % (init_sym, sum(init_rewards), trained_sym, sum(trained_rewards)))
        
        env.close()

