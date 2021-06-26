import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from abstract_machine import make_abstract_machine, memorize_env
from restack import invert, compute_spatial_reward, compute_symbolic_reward
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
    num_epochs = 150
    # num_repetitions = 2
    # num_episodes = 30
    # num_epochs = 30
    
    run_exp = False
    showresults = True
    showenv = False
    showtrained = False
    # tr.autograd.set_detect_anomaly(True)
    
    use_penalties = True
    # learning_rates=[0.0001, 0.00005] # all stack layers trainable
    learning_rates=[0.0001, 0.000075, 0.00005] # all stack layers trainable
    trainable = ["ik", "to", "tc", "po", "pc", "right", "above", "base"]

    # learning_rates=[0.00005, 0.00001] # base only trainable, 5 works better than 1
    # trainable = ["ik", "to", "tc", "po", "pc", "base"]

    # learning_rates = [0.00005] # ik/motor layrs only
    # trainable = ["ik", "to", "tc", "po", "pc"]
    # trainable = ["ik"]

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
                env.load_blocks(thing_below)
            
                # set up rvm and virtualize
                rvm = make_abstract_machine(env, num_bases, max_levels)
                memorize_env(rvm, goal_thing_above)
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
                
                for epoch in range(num_epochs):
                    start_epoch = time.perf_counter()
                    epoch_rewards = []
                    epoch_baselines = []
                    epoch_rtgs = []
    
                    # print("Wik:")
                    # print(conn_params["ik"][:,:8])
            
                    for episode in range(num_episodes):
                        start_episode = time.perf_counter()
                
                        baseline = 0
    
                        if episode == 0: # noiseless first episode
                            reward, log_prob, rewards, log_probs = run_episode(
                                env, thing_below, goal_thing_below, nvm, init_regs, init_conns, penalty_tracker, sigma=0)
                            rewards_to_go = [sum(rewards)]
                        else:                    
                            reward, log_prob, rewards, log_probs = run_episode(
                                env, thing_below, goal_thing_below, nvm, init_regs, init_conns, penalty_tracker, sigma)
                            
                            if use_penalties:
                                rewards = np.array(rewards)
                                rewards_to_go = np.cumsum(rewards)
                                rewards_to_go = rewards_to_go[-1] - rewards_to_go + rewards
                                for t in range(len(rewards)):
                                    loss = - (rewards_to_go[t] * log_probs[t])
                                    loss.backward(retain_graph=(t+1 < len(rewards))) # each log_prob[t] shares the graph
                            else:
                                rewards_to_go = [0]
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
                    with open("pfc_%f.pkl" % learning_rate,"wb") as f: pk.dump(results, f)
    
                    avg_reward = np.mean(epoch_rtgs)
                    std_reward = np.std(epoch_rtgs)
                    print(" %d,%d: R = %f (+/- %f), dW: %s" % (rep, epoch, avg_reward, std_reward, delta))
                    epoch_time = time.perf_counter() - start_epoch
                    print(" %d,%d took %fs" % (rep, epoch, epoch_time))
    
                env.close()
                rep_time = time.perf_counter() - start_rep
                print("%d took %fs" % (rep, rep_time))
                # save model (pkl might need outside epoch loop??)
                with open("pfc_%f_state_%d.pkl" % (learning_rate, rep),"wb") as f: pk.dump((init_regs, init_conns), f)
    
    if showresults:
        import os
        import matplotlib.pyplot as pt
        
        # one representative run
        if False:
            learning_rate = .000075
            with open("stack_trained/pfc_%f.pkl" % learning_rate,"rb") as f: results = pk.load(f)
            rep = 4
            num_epochs = len(results[rep])
            epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
            pt.figure(figsize=(5,1.75))
            # x, y = zip(*[(r+.5,reward) for r in range(num_epochs) for reward in epoch_rewards[r][1:]])
            # pt.plot(x, y, '.', c=(.75, .75, .75))
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'k-')
            # y = [np.mean(rewards[1:]) for rewards in epoch_rtgs]
            # ye = [np.std(rewards[1:]) for rewards in epoch_rtgs]
            # pt.errorbar(np.arange(len(y)), y, yerr=ye, fmt='k-')
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_rewards], 'k--')
            from matplotlib.lines import Line2D
            h1 = Line2D([], [], linestyle="-", color="k")
            h2 = Line2D([], [], linestyle="-", color="k")
            pt.legend([h1, h2], ["avg. reward", "avg. symbolic"], framealpha=1)
            x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rtgs[r][1:]])
            pt.plot(x, y, '.', c=(.75, .75, .75), zorder=-1)
            pt.xlabel("Training epoch")
            pt.ylabel("Performance")
            pt.tight_layout()
            pt.savefig("fail_train_one.pdf")
            # pt.savefig("fail_train_one.eps")
            # pt.savefig("fail_train_one.png")
            pt.show()

            pt.figure(figsize=(5,1.75))
            # pt.plot([d["ik"] for d in deltas], color=(0,)*3, label="ik")
            # pt.plot([d["to"] for d in deltas], color=(.25,)*3, label="to,tc,pc")
            # pt.plot([d["pc"] for d in deltas], color=(.25,)*3)
            # pt.plot([d["tc"] for d in deltas], color=(.25,)*3)
            # pt.plot([d["right"] for d in deltas], color=(.5,)*3, label="right,above,next")
            # pt.plot([d["above"] for d in deltas], color=(.5,)*3)
            # pt.plot([d["base"] for d in deltas], color=(.5,)*3)
            pt.plot([d["ik"] for d in deltas], 'k-', label="ik")
            pt.plot([d["to"] for d in deltas], 'k--', label="to,tc,pc")
            pt.plot([d["pc"] for d in deltas], 'k--')
            pt.plot([d["tc"] for d in deltas], 'k--')
            pt.plot([d["right"] for d in deltas], 'k:', label="right,above,next")
            pt.plot([d["above"] for d in deltas], 'k:')
            pt.plot([d["base"] for d in deltas], 'k:')
            # c = np.linspace(0, .5, len(deltas[0]))
            # # for n,name in enumerate(deltas[0].keys()):
            # for n,name in enumerate(reversed(["right","base","above","ik","to","tc","pc"])):
            # # for n,name in enumerate(["right","base","above","ik","to","tc","po","pc"]):
            #     delt = [d[name] for d in deltas]
            #     pt.plot(delt, label=name, color=(c[n],)*3)
            #     # pt.plot(delt, label=name)
            pt.xlabel("Training epoch")
            pt.ylabel("Weight changes")
            pt.legend(framealpha=1)
            pt.tight_layout()
            pt.savefig("deltas.pdf")
            # pt.savefig("deltas.eps")
            # pt.savefig("deltas.png")
            pt.show()
        
            # different learning rates
            pt.figure(figsize=(5,4))
            for lr, learning_rate in enumerate(learning_rates):
                fname = "stack_trained/pfc_%f.pkl" % learning_rate
                with open(fname,"rb") as f: results = pk.load(f)    
                num_repetitions = len(results)
                pt.subplot(len(learning_rates), 1, lr+1)
                num_epochs = len(results[0])
                syms = np.empty((num_repetitions, num_epochs))
                for rep in range(num_repetitions):
                    epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                    syms[rep,:num_epochs] = np.array([np.mean(rewards[1:]) for rewards in epoch_rewards])
                pt.plot(syms.T, '-', color=(.75, .75, .75))
                pt.plot(syms.mean(axis=0), 'k-')
                pt.xlim([0, 150])
                # pt.ylabel("LR = %s" % str(learning_rate))
                pt.ylabel("%.1e" % learning_rate)
                if lr == 0: pt.title("Symbolic performance with different learning rates")
                if lr+1 == len(learning_rates): pt.xlabel("Training epoch")
            pt.tight_layout()
            pt.savefig("fail_train_many.pdf")
            # pt.savefig("fail_train_many.png")
            # pt.savefig("fail_train_many.eps")
            pt.show()

        for lr, learning_rate in enumerate(learning_rates):

            # with open("pfc.pkl","rb") as f: results = pk.load(f)
            # fname = "stack_trained/pfc_%f.pkl" % learning_rate
            fname = "pfc_%f.pkl" % learning_rate
            if os.path.exists(fname):
                with open(fname,"rb") as f: results = pk.load(f)
    
            num_repetitions = len(results)
            for rep in range(num_repetitions):
                epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                num_epochs = len(results[rep])
            
                # print("rep %d LC:" % rep)
                # for r,rewards in enumerate(epoch_rewards): print(r, rewards)
                
                pt.subplot(num_repetitions, len(learning_rates), len(learning_rates)*rep+lr+1)
                if rep == 0: pt.title(str(learning_rate))
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'k-')
                pt.plot([rewards[0] for rewards in epoch_rtgs], 'b-')
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_rewards], 'k--')
                # pt.plot([np.mean(baselines) for baselines in epoch_baselines], 'b-')
                x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rtgs[r]])
                pt.plot(x, y, 'k.')
                # x, y = zip(*[(r,baseline) for r in range(num_epochs) for baseline in epoch_baselines[r]])
                # pt.plot(np.array(x)+.5, y, 'b.')
        
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
        with open("pfc_state_%d.pkl" % rep, "rb") as f: (init_regs, init_conns) = pk.load(f)

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

