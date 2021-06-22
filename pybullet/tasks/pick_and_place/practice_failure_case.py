import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from abstract_machine import make_abstract_machine, memorize_env
from restack import invert, compute_spatial_reward, compute_symbolic_reward
from nvm import virtualize

def calc_reward(sym_reward, spa_reward):
    # reward = sym_reward + 0.1*spa_reward
    # reward = sym_reward
    reward = spa_reward
    return reward

def run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns, sigma=0):

    # reload blocks
    env.reset()
    env.load_blocks(thing_below)

    # reset nvm, input new env, mount main program
    nvm.reset_state(init_regs, init_conns)
    memorize_env(nvm, goal_thing_above)
    nvm.mount("main")

    log_prob = 0.0 # accumulate over episode

    dbg = False
    if dbg: nvm.dbg()
    target_changed = True
    while True:
        done = nvm.tick() # reliable if core is not trained
        if dbg: nvm.dbg()
        if nvm.tick_counter % 100 == 0: print("     tick %d" % nvm.tick_counter)
        if target_changed:
            mu = nvm.registers["jnt"].content
            if sigma > 0:
                dist = tr.distributions.normal.Normal(mu, sigma)
                position = dist.sample()
                log_probs = dist.log_prob(position)
                log_prob += log_probs.sum() # multivariate white noise
            else:
                position = mu
            nvm.env.goto_position(position.detach().numpy())
        tar = nvm.registers["tar"]
        # decode has some robustness to noise even if tar connections are trained
        target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
        if done: break
    
    sym_reward = compute_symbolic_reward(nvm.env, goal_thing_below)
    spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
    reward = calc_reward(sym_reward, spa_reward)
    
    return reward, log_prob

if __name__ == "__main__":

    num_repetitions = 1
    num_episodes = 2
    num_epochs = 3
    
    run_exp = True
    showresults = True
    # tr.autograd.set_detect_anomaly(True)

    sigma = 0.001 # stdev in random angular sampling (radians)

    # one failure case:
    max_levels = 3
    num_blocks = 5
    num_bases = 5
    thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
    goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
    goal_thing_above = invert(goal_thing_below, num_blocks, num_bases)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"

    if run_exp:

        results = []
        for rep in range(num_repetitions):
            start_rep = time.perf_counter()
            results.append([])
        
            env = BlocksWorldEnv(show=False)
            env.load_blocks(thing_below)
        
            # set up rvm and virtualize
            rvm = make_abstract_machine(env, num_bases, max_levels)
            memorize_env(rvm, goal_thing_above)
            rvm.reset({"jnt": "rest"})
            rvm.mount("main")
        
            nvm = virtualize(rvm)
            init_regs, init_conns = nvm.get_state()
            init_regs["jnt"] = tr.tensor(rvm.ik["rest"]).float()

            # set up trainable connections
            # trainable = ["ik", "to", "tc", "pc", "pc", "right", "above", "base"]
            # trainable = ["ik", "to", "tc", "pc", "pc"]
            trainable = ["ik"]
            conn_params = {name: init_conns[name] for name in trainable}
            for p in conn_params.values(): p.requires_grad_()
            opt = tr.optim.Adam(conn_params.values(), lr=0.00001)
            
            # save original values for comparison
            orig_conns = {name: init_conns[name].detach().clone() for name in trainable}
            
            for epoch in range(num_epochs):
                start_epoch = time.perf_counter()
                epoch_rewards = []
                epoch_baselines = []
        
                for episode in range(num_episodes):
                    start_episode = time.perf_counter()
            
                    baseline = 0

                    if episode == 0: # noiseless first episode
                        reward, log_prob = run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns, sigma=0)
                    else:                    
                        reward, log_prob = run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns, sigma)
                        loss = - (reward - baseline) * log_prob
                        loss.backward()
    
                    epoch_rewards.append(reward)
                    epoch_baselines.append(baseline)
                    episode_time = time.perf_counter() - start_episode
                    print("    %d,%d,%d: r = %f, b = %f, lp= %f, took %fs" % (
                        rep, epoch, episode, reward, baseline, log_prob, episode_time))
    
                # update params based on episodes
                opt.step()
                opt.zero_grad()
                
                delta = max((orig_conns[name] - conn_params[name]).abs().max() for name in trainable).item()
                avg_reward = np.mean(epoch_rewards)
                std_reward = np.std(epoch_rewards)
                
                results[-1].append((epoch_rewards, epoch_baselines, delta))
                with open("pfc.pkl","wb") as f: pk.dump(results, f)
                epoch_time = time.perf_counter() - start_epoch
                print(" %d,%d: R = %f (+/- %f), dW = %f" % (rep, epoch, avg_reward, std_reward, delta))
                print(" %d,%d took %fs" % (rep, epoch, epoch_time))

            env.close()
            rep_time = time.perf_counter() - start_rep
            print("%d took %fs" % (rep, rep_time))
    
    if showresults:
        import matplotlib.pyplot as pt

        with open("pfc.pkl","rb") as f: results = pk.load(f)
