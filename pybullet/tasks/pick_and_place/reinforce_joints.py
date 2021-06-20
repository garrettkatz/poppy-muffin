import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import random_problem_instance, compute_spatial_reward, compute_symbolic_reward
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize

def calc_reward(sym_reward, spa_reward):
    reward = sym_reward + 0.1*spa_reward
    # reward = spa_reward
    return reward

def rvm_baseline(env, thing_below, goal_thing_above, rvm):

    start = time.perf_counter()

    # reload blocks
    env.reset()
    env.load_blocks(thing_below)

    # reset rvm, input new env, mount main program
    rvm.env = env
    memorize_env(rvm, goal_thing_above)
    rvm.reset({"jnt": "rest"})
    rvm.mount("main")

    # run
    ticks = rvm.run()
    running_time = time.perf_counter() - start    

    sym_reward = compute_symbolic_reward(env, goal_thing_below)
    spa_reward = compute_spatial_reward(env, goal_thing_below)
    reward = calc_reward(sym_reward, spa_reward)
    
    return running_time, reward

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
        done = nvm.tick()
        if dbg: nvm.dbg()
        # if nvm.tick_counter % 100 == 0: print("     tick %d" % nvm.tick_counter)
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
        target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
        if done: break
    
    sym_reward = compute_symbolic_reward(nvm.env, goal_thing_below)
    spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
    reward = calc_reward(sym_reward, spa_reward)
    
    return reward, log_prob
    

# ~ 11s per episode on lightop, ~6 on outpost
# 12 hours = 8*60*60s = 6s * 30epis * epochs * 3reps
# epochs = 12*60*60 / (6*30*3) = 80

if __name__ == "__main__":

    num_blocks, max_levels, num_bases = 6, 3, 6
    num_repetitions = 1
    num_episodes = 2
    num_epochs = 3
    
    run_exp = False
    showresults = True
    # tr.autograd.set_detect_anomaly(True)

    if run_exp:
    
        results = []
        for rep in range(num_repetitions):
            start_rep = time.perf_counter()
            results.append([])

            env = BlocksWorldEnv(show=False)
            # placehold blocks for nvm init
            env.load_blocks({"b%d" % n: "t%d" % n for n in range(num_bases)})
        
            rvm = make_abstract_machine(env, num_bases, max_levels)
            nvm = virtualize(rvm)
            # print(nvm.size())
            # input('.')
            init_regs, init_conns = nvm.get_state()
            orig_ik_W = init_conns["ik"].clone()
            init_regs["jnt"] = tr.tensor(rvm.ik["rest"]).float()
        
            # set up trainable connections
            conn_params = {
                name: init_conns[name]
                # for name in ["ik", "to", "tc", "pc", "pc"]
                for name in ["ik"]
            }
            for p in conn_params.values(): p.requires_grad_()
            opt = tr.optim.Adam(conn_params.values(), lr=0.00001)

            # # one failure case in all episodes:
            # thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
            # goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
            # goal_thing_above = nvm.env.invert(goal_thing_below)
            # for key, val in goal_thing_above.items():
            #     if val == "none": goal_thing_above[key] = "nil"
        
            for epoch in range(num_epochs):
                start_epoch = time.perf_counter()
                epoch_rewards = []
                epoch_baselines = []

                for episode in range(num_episodes):
                    start_episode = time.perf_counter()
            
                    # different problem instance each episode
                    env.reset()
                    thing_below, goal_thing_below = random_problem_instance( # loads thing_below
                        env, num_blocks, max_levels, num_bases)
                    goal_thing_above = env.invert(goal_thing_below)
                    for key, val in goal_thing_above.items():
                        if val == "none": goal_thing_above[key] = "nil"
                    
                    # run twice, once noiseless, to compute baseline
                    sigma = 0.001 # stdev in random angular sampling (radians)
                    log_prob = 0.0 # accumulate over episode
                    
                    # # noiseless
                    # start_noiseless = time.perf_counter()
                    # with tr.no_grad():
                    #     baseline, _ = run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns)
                    # noiseless_time = time.perf_counter() - start_noiseless
                    # print("     noiseless took %fs" % noiseless_time)
                    noiseless_time, baseline = rvm_baseline(env, thing_below, goal_thing_above, rvm)                    
                    print("     noiseless took %fs" % noiseless_time)

                    # noisy
                    start_noisy = time.perf_counter()
                    reward, log_prob = run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns, sigma)
                    noisy_time = time.perf_counter() - start_noisy
                    print("     noisy took %fs" % noisy_time)

                    # for noise in [False, True]:

                    #     # reload blocks
                    #     env.reset()
                    #     env.load_blocks(thing_below)
                    
                    #     # reset nvm, input new env, mount main program
                    #     nvm.reset_state(init_regs, init_conns)
                    #     memorize_env(nvm, goal_thing_above)
                    #     nvm.mount("main")

                    #     dbg = False
                    #     if dbg: nvm.dbg()
                    #     target_changed = True
                    #     while True:
                    #         done = nvm.tick()
                    #         if dbg: nvm.dbg()
                    #         # if nvm.tick_counter % 100 == 0: print("     tick %d" % nvm.tick_counter)
                    #         if target_changed:
                    #             mu = nvm.registers["jnt"].content
                    #             if noise:
                    #                 dist = tr.distributions.normal.Normal(mu, sigma)
                    #                 position = dist.sample()
                    #                 log_probs = dist.log_prob(position)
                    #                 log_prob += log_probs.sum() # multivariate white noise
                    #             else:
                    #                 position = mu
                    #             nvm.env.goto_position(position.detach().numpy())
                    #         tar = nvm.registers["tar"]
                    #         target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
                    #         if done: break
                        
                    #     # calculate reward (second noisy iteration overwrites first noiseless one)
                        
                    #     spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
                    #     # sym_reward = compute_symbolic_reward(nvm.env, goal_thing_below)
                    #     # reward = sym_reward + 0.5*spa_reward
                    #     reward = spa_reward
                        
                    #     if not noise:
                    #         baseline = reward
                    #         print("     noiseless done...")

                    # accumulate gradient
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
                
                delta = (orig_ik_W - conn_params["ik"]).abs().max()
                avg_reward = np.mean(epoch_rewards)
                std_reward = np.std(epoch_rewards)
                
                results[-1].append((epoch_rewards, epoch_baselines, delta.item()))
                with open("rj.pkl","wb") as f: pk.dump(results, f)
                epoch_time = time.perf_counter() - start_epoch
                print(" %d,%d: R = %f (+/- %f), dW = %f" % (rep, epoch, avg_reward, std_reward, delta))
                print(" %d,%d took %fs" % (rep, epoch, epoch_time))

            env.close()
            rep_time = time.perf_counter() - start_rep
            print("%d took %fs" % (rep, rep_time))
    
    if showresults:
        import matplotlib.pyplot as pt

        with open("rj.pkl","rb") as f: results = pk.load(f)
        # with open("rj_1e-5_spa.pkl","rb") as f: results = [pk.load(f)]
        # with open("rj_1e-5_sym_spa_.01.pkl","rb") as f: results = [pk.load(f)]
        # with open("rj_1e-5_3rep.pkl","rb") as f: results = pk.load(f)
        
        num_repetitions = len(results)
        for rep in range(num_repetitions):
            epoch_rewards, epoch_baselines, deltas = zip(*results[rep])
            num_epochs = len(results[rep])
        
            print("rep %d LC:" % rep)
            for r,rewards in enumerate(epoch_rewards): print(r, rewards)
            
            pt.subplot(num_repetitions, 1, rep+1)
            pt.plot([np.mean(rewards) for rewards in epoch_rewards], 'k-')
            # pt.plot([rewards[0] for rewards in epoch_rewards], 'b-')
            pt.plot([np.mean(baselines) for baselines in epoch_baselines], 'b-')
            x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rewards[r]])
            pt.plot(x, y, 'k.')
            x, y = zip(*[(r,baseline) for r in range(num_epochs) for baseline in epoch_baselines[r]])
            pt.plot(np.array(x)+.5, y, 'b.')
    
            # pt.plot(np.log(-np.array(rewards)))
            # pt.ylabel("log(-R)")
            
            # trend line
            window = 3
            avg_rewards = np.mean(epoch_rewards, axis=1)
            trend = np.zeros(len(avg_rewards) - window+1)
            for w in range(window):
                trend += avg_rewards[w:w+len(trend)]
            trend /= window
            # for r in range(len(avg_rewards)-window):
            #     avg_rewards[r] += avg_rewards[r+1:r+window].sum()
            #     avg_rewards[r] /= window
            # pt.plot(np.arange(window//2, len(avg_rewards)-(window//2)), avg_rewards, 'ro-')
            pt.plot(np.arange(len(trend)) + window//2, trend, 'ro-')
            # pt.ylim([-2, 0])

        pt.show()
    
