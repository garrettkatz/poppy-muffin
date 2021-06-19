import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import random_problem_instance, compute_spatial_reward, compute_symbolic_reward
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize

# ~ 11s per episode
# 12 hours = 8*60*60s = 11s * 30epis * epochs * 3reps
# epochs = 12*60*60 / (11*30*3) = 

if __name__ == "__main__":

    num_blocks, max_levels, num_bases = 6, 3, 6
    num_repetitions = 2
    num_episodes = 2
    num_epochs = 2
    
    run_exp = False
    showresults = True

    if run_exp:
    
        results = []
        for rep in range(num_repetitions):
            start_rep = time.perf_counter()
            results.append([])

            env = BlocksWorldEnv(show=False)
            env.load_blocks({"b%d" % n: "t%d" % n for n in range(num_bases)}) # placehold for nvm init
        
            # tr.autograd.set_detect_anomaly(True)
        
            am = make_abstract_machine(env, num_bases, max_levels)
            nvm = virtualize(am)
            # print(nvm.size())
            # input('.')
        
            # set up trainable connections
            conn_params = {
                name: nvm.connections[name].W
                # for name in ["ik", "to", "tc", "pc", "pc"]
                for name in ["ik"]
            }
            for p in conn_params.values(): p.requires_grad_()
            opt = tr.optim.Adam(conn_params.values(), lr=0.00001)
            
            orig_ik = nvm.connections["ik"].W.clone()
        
            # # one failure case in all episodes:
            # thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
            # goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
            # goal_thing_above = nvm.env.invert(goal_thing_below)
            # for key, val in goal_thing_above.items():
            #     if val == "none": goal_thing_above[key] = "nil"
        
            lc = []
            for epoch in range(num_epochs):
                start_epoch = time.perf_counter()
                epoch_rewards = []

                for episode in range(num_episodes):
                    start_episode = time.perf_counter()
            
                    # different problem instance each episode
                    env.reset()
                    thing_below, goal_thing_below = random_problem_instance(
                        env, num_blocks, max_levels, num_bases)
                    goal_thing_above = env.invert(goal_thing_below)
                    for key, val in goal_thing_above.items():
                        if val == "none": goal_thing_above[key] = "nil"
            
                    # reset trainable parameters
                    for name, param in conn_params.items():
                        nvm.connections[name].W = param
                    
                    # input env
                    memorize_env(nvm, goal_thing_above)
                
                    reset_dict = {"jnt": tr.tensor(am.ik["rest"]).float()}
                    nvm.reset(reset_dict)
                
                    sigma = 0.001 # radians
                    log_prob = 0.0 # accumulate over episode
                
                    dbg = False
                    nvm.mount("main")
                    if dbg: nvm.dbg()
                    target_changed = True
                    while True:
                        done = nvm.tick()
                        if dbg: nvm.dbg()
                        if target_changed:
                            mu = nvm.registers["jnt"].content
                            dist = tr.distributions.normal.Normal(mu, sigma)
                            if episode == 0:
                                position = mu.detach().clone()
                            else:
                                position = dist.sample()
                                log_probs = dist.log_prob(position)
                                log_prob += log_probs.sum() # multivariate white noise
                            nvm.env.goto_position(position.detach().numpy())
                        tar = nvm.registers["tar"]
                        target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
                        if done: break
                    
                    spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
                    # sym_reward = compute_symbolic_reward(nvm.env, goal_thing_below)
                    # reward = sym_reward + 0.5*spa_reward
                    reward = spa_reward
                    epoch_rewards.append(reward)
                    
                    if episode != 0:
                        loss = -reward * log_prob
                        loss.backward()
    
                    episode_time = time.perf_counter() - start_episode
                    print("    %d,%d,%d: r = %f, lp= %f, took %fs" % (
                        rep, epoch, episode, reward, log_prob, episode_time))
    
                # update params based on episodes
                opt.step()
                opt.zero_grad()
                
                delta = (orig_ik - conn_params["ik"]).abs().max()
                avg_reward = np.mean(epoch_rewards)
                std_reward = np.std(epoch_rewards)
                
                results[-1].append((epoch_rewards, delta.item()))
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
        
        num_repetitions = len(results)
        for rep in range(num_repetitions):
            epoch_rewards, deltas = zip(*results[rep])
            num_epochs = len(results[rep])
        
            print("rep %d LC:" % rep)
            for r,rewards in enumerate(epoch_rewards): print(r, rewards)
            
            pt.subplot(num_repetitions, 1, rep+1)
            pt.plot([np.mean(rewards) for rewards in epoch_rewards], 'k-')
            pt.plot([rewards[0] for rewards in epoch_rewards], 'b-')
            x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rewards[r]])    
            pt.plot(x, y, 'k.')
    
            # pt.plot(np.log(-np.array(rewards)))
            # pt.ylabel("log(-R)")
            
            # trend line
            avg_rewards = np.mean(epoch_rewards, axis=1)
            window = 5
            for r in range(len(avg_rewards)-window):
                avg_rewards[r] += avg_rewards[r+1:r+window].sum()
                avg_rewards[r] /= window
            pt.plot(avg_rewards, 'ro-')

        pt.show()
    
