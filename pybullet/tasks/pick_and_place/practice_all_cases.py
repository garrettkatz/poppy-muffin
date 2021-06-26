import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from abstract_machine import make_abstract_machine, memorize_env
from restack import invert, compute_spatial_reward, compute_symbolic_reward, random_problem_instance
from nvm import virtualize

np.set_printoptions(linewidth=1000)

def calc_reward(sym_reward, spa_reward):
    # reward = sym_reward + 0.1*spa_reward
    reward = sym_reward
    # reward = spa_reward
    return reward

class PenaltyTracker:
    def __init__(self):
        self.penalty = 0
    def reset(self):
        self.penalty = 0
    def step_hook(self, env, action):
        mp = env.movement_penalty()
        self.penalty += mp
        # print("penalty: %.5f" % mp)

def run_episode(env, thing_below, goal_thing_below, nvm, init_regs, init_conns, penalty_tracker, sigma=0):
    
    # reload blocks
    env.reset()
    env.load_blocks(thing_below)

    # invert goals for nvm
    goal_thing_above = invert(goal_thing_below, num_blocks=len(thing_below), num_bases=len(env.bases))
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"

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
    
    tr.set_printoptions(precision=8, sci_mode=False, linewidth=1000)
    
    num_repetitions = 5
    num_episodes = 30
    num_epochs = 50
    # num_repetitions = 2
    # num_episodes = 2
    # num_epochs = 2
    
    run_exp = True
    showresults = False
    showenv = False
    showtrained = False
    # tr.autograd.set_detect_anomaly(True)
    
    use_penalties = True
    reject_above = 0

    # learning_rates=[0.00005] # all stack layers trainable
    # learning_rates=[0.0001, 0.000075, 0.00005] # all stack layers trainable
    # learning_rates=[0.0001, 0.00005] # all stack layers trainable
    # learning_rates=[0.001, .0005] # all stack layers trainable
    # learning_rates=[0.00001] # all stack layers trainable
    # trainable = ["ik", "to", "tc", "po", "pc", "right", "above", "base"]

    # learning_rates=[0.00005, 0.00001] # base only trainable, 5 works better than 1
    # trainable = ["ik", "to", "tc", "po", "pc", "base"]

    learning_rates = [0.01] # ik/motor layrs only
    # trainable = ["ik", "to", "tc", "po", "pc"]
    trainable = ["ik"]

    sigma = 0.001 # stdev in random angular sampling (radians)

    # problem size:
    max_levels = 3
    num_blocks = 5
    num_bases = 5

    penalty_tracker = PenaltyTracker()

    σ1 = tr.tensor(1.).tanh()
    def σ(v): return tr.tanh(v) / σ1
    # def σ(v): return v

    if run_exp:
        
        lr_results = {lr: list() for lr in learning_rates}
        for rep in range(num_repetitions):
            for learning_rate in learning_rates:
        
                results = lr_results[learning_rate]
                start_rep = time.perf_counter()
                results.append([])
                
                env = BlocksWorldEnv(show=showenv, step_hook = penalty_tracker.step_hook)
                env.load_blocks({"b%d"%n: "t%d"%n for n in range(num_bases)}) # placeholder for rvm construction
            
                # set up rvm and virtualize
                rvm = make_abstract_machine(env, num_bases, max_levels, gen_regs=["r0","r1"])
                rvm.reset({"jnt": "rest"})
                rvm.mount("main")
    
                nvm = virtualize(rvm, σ)
                init_regs, init_conns = nvm.get_state()
                init_regs["jnt"] = tr.tensor(rvm.ik["rest"]).float()
    
                # set up trainable connections
                conn_params = {name: init_conns[name] for name in trainable}
                for p in conn_params.values(): p.requires_grad_()
                opt = tr.optim.Adam(conn_params.values(), lr=learning_rate)
                
                # save original values for comparison
                orig_conns = {name: init_conns[name].detach().clone() for name in trainable}
                
                # cleanup
                env.close()

                for epoch in range(num_epochs):
                    start_epoch = time.perf_counter()
                    epoch_rewards = []
                    epoch_baselines = []
                    epoch_rtgs = []
    
                    for episode in range(num_episodes):
                        start_episode = time.perf_counter()
                        
                        # random problem instance
                        env = BlocksWorldEnv(show=showenv, step_hook = penalty_tracker.step_hook)
                        thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)
                        nvm.env = env
                
                        baseline = 0
    
                        reward, log_prob, rewards, log_probs = run_episode(
                            env, thing_below, goal_thing_below, nvm, init_regs, init_conns, penalty_tracker, sigma)
                        
                        env.close()

                        if use_penalties:
                            rewards = np.array(rewards)
                            rewards_to_go = np.cumsum(rewards)
                            rewards_to_go = rewards_to_go[-1] - rewards_to_go + rewards
                            if reward <= reject_above:
                                for t in range(len(rewards)):
                                    loss = - (rewards_to_go[t] * log_probs[t])
                                    loss.backward(retain_graph=(t+1 < len(rewards))) # each log_prob[t] shares the graph
                        else:
                            rewards_to_go = [0]
                            if reward <= reject_above:
                                loss = - (reward - baseline) * log_prob
                                loss.backward()
                                                
                        epoch_rewards.append(reward)
                        epoch_baselines.append(baseline)
                        epoch_rtgs.append(rewards_to_go[0])
                        episode_time = time.perf_counter() - start_episode
                        print("    %d,%d,%d: r = %f[%f], b = %f, lp= %f, took %fs" % (
                            rep, epoch, episode, reward, rewards_to_go[0], baseline, log_prob, episode_time))
        
                    # update params based on episodes
                    opt.step()
                    opt.zero_grad()
    
                    delta = {name: (orig_conns[name] - conn_params[name]).abs().max().item() for name in trainable}
                    results[-1].append((epoch_rewards, epoch_baselines, epoch_rtgs, delta))
                    with open("pac_%f.pkl" % learning_rate,"wb") as f: pk.dump(results, f)
    
                    avg_reward = np.mean(epoch_rtgs)
                    std_reward = np.std(epoch_rtgs)
                    print(" %d,%d: R = %f (+/- %f), dW: %s" % (rep, epoch, avg_reward, std_reward, delta))
                    epoch_time = time.perf_counter() - start_epoch
                    print(" %d,%d took %fs" % (rep, epoch, epoch_time))
    
                rep_time = time.perf_counter() - start_rep
                print("%d took %fs" % (rep, rep_time))
                # save model (pkl might need outside epoch loop??)
                with open("pac_%f_state_%d.pkl" % (learning_rate, rep),"wb") as f: pk.dump((init_regs, init_conns), f)
    
    if showresults:
        import os
        import matplotlib.pyplot as pt
        
        for lr, learning_rate in enumerate(learning_rates):

            # with open("pac.pkl","rb") as f: results = pk.load(f)
            fname = "pac_%f.pkl" % learning_rate
            # fname = "stack_trained/pac_%f.pkl" % learning_rate
            # fname = "all_case5/pac_%f.pkl" % learning_rate
            if os.path.exists(fname):
                with open(fname,"rb") as f: results = pk.load(f)
    
            num_repetitions = len(results)
            for rep in range(num_repetitions):
                epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                num_epochs = len(results[rep])
                
                pt.subplot(num_repetitions, len(learning_rates), len(learning_rates)*rep+lr+1)
                if rep == 0: pt.title(str(learning_rate))
                x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rtgs[r]])
                pt.plot(x, y, 'k.')
                # x, y = zip(*[(r,baseline) for r in range(num_epochs) for baseline in epoch_baselines[r]])
                # pt.plot(np.array(x)+.5, y, 'b.')
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'b-')
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_rewards], 'b--')
        
                # pt.plot(np.log(-np.array(rewards)))
                # pt.ylabel("log(-R)")
                
                # trend line
                avg_rewards = np.mean(epoch_rtgs, axis=1)
                window = min(10, len(avg_rewards))
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
                
                # constant line
                pt.plot([0, len(avg_rewards)], avg_rewards[[0, 0]],'g-')

        pt.show()
    
    if showtrained:

        rep = 0
        with open("pac_state_%d.pkl" % rep, "rb") as f: (init_regs, init_conns) = pk.load(f)

        env = BlocksWorldEnv(show=True, step_hook = penalty_tracker.step_hook)
        env.load_blocks(thing_below)
    
        # set up rvm and virtualize
        rvm = make_abstract_machine(env, num_bases, max_levels)
        memorize_env(rvm, goal_thing_above)
        rvm.reset({"jnt": "rest"})
        rvm.mount("main")

        nvm = virtualize(rvm, σ)
        nvm.reset_state(init_regs, init_conns)
        reward, log_prob, rewards, log_probs = run_episode(
            env, thing_below, goal_thing_below, nvm, init_regs, init_conns, penalty_tracker, sigma=0)



