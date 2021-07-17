import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv, MovementPenaltyTracker
from abstract_machine import make_abstract_machine, memorize_env, memorize_problem
from restack import invert, compute_spatial_reward, compute_symbolic_reward
from nvm import virtualize
import neural_virtual_machine as nv
import block_stacking_problem as bp
from failure_case import find_failure_case
import pybullet as pb

np.set_printoptions(linewidth=1000)

def calc_reward(sym_reward, spa_reward):
    # reward = sym_reward + 0.1*spa_reward
    reward = sym_reward
    # reward = spa_reward
    return reward

def run_episodes(problem, nvm, W_init, v_init, num_time_steps, num_episodes, penalty_tracker, sigma):

    memorize_problem(nvm, problem)
    for name in ["obj","loc","goal"]:
        W_init[name][0] = nvm.connections[name].W.unsqueeze(dim=0)

    perf_counter = time.perf_counter()
    # W, v = nvm.net.run(W_init, v_init, num_time_steps)
    nvm.net.clear_ticks()
    for t in range(num_time_steps):
        nvm.net.tick(W_init, v_init)
        # nvm.pullback(t)
        # nvm.dbg()
        # input('.')
    W, v = nvm.net.weights, nvm.net.activities
    print("    NVM run took %fs (%d timesteps)" % (time.perf_counter() - perf_counter, num_time_steps))
    
    perf_counter = time.perf_counter()
    positions, log_probs = tuple({b: list() for b in range(num_episodes)} for _ in [0,1])
    tar = nvm.registers["tar"]
    for t in range(2, num_time_steps):
        if nvm.decode("tar", t-2) != nvm.decode("tar", t-1):
            mu = v["jnt"][t][0,:,0]
            dist = tr.distributions.normal.Normal(mu, sigma)
            for b in range(num_episodes):
                position = dist.sample() if b > 0 else mu # first episode noiseless
                positions[b].append(position)
                log_probs[b].append(dist.log_prob(position).sum()) # multivariate white noise
    for b in range(num_episodes):
        if any([tr.isnan(lp) for lp in log_probs[b]]):
            print(" "*6, log_probs[b])
            for t in range(2, num_time_steps):
                if nvm.decode("tar", t-2, b) != nvm.decode("tar", t-1, b):
                    nvm.pullback(t, b)
                    nvm.dbg()
            #         input('.')
            # input('.')
    print("    log probs took %fs (%d motions)" % (time.perf_counter() - perf_counter, len(positions[0])))

    perf_counter = time.perf_counter()
    # env = BlocksWorldEnv(show=False, step_hook=penalty_tracker.step_hook)
    env = nvm.env
    rewards, sym = [], []
    for b in range(num_episodes):
        rewards.append([])
        env.reset()
        env.load_blocks(problem.thing_below)
        for position in positions[b]:
            penalty_tracker.reset()
            env.goto_position(position.detach().numpy(), speed=1.5)
            rewards[b].append(-penalty_tracker.penalty)
        sym_reward = compute_symbolic_reward(env, problem.goal_thing_below)
        # spa_reward = compute_spatial_reward(env, problem.goal_thing_below)
        # end_reward = calc_reward(sym_reward, spa_reward)
        # rewards[b][-1] += end_reward
        rewards[b][-1] += sym_reward
        sym.append(sym_reward)
        # env.reset()
    # env.close()
    print("    simulation rewards took %fs" % (time.perf_counter() - perf_counter))
            
    perf_counter = time.perf_counter()
    rewards_to_go = []
    for b in range(num_episodes):
        rewards[b] = tr.tensor(rewards[b]).float()
        rtg = tr.cumsum(rewards[b], dim=0)
        rtg = rtg[-1] - rtg + rewards[b]
        rewards_to_go.append(rtg)
    baselines = tr.stack(rewards_to_go[1:]).mean(dim=0) # exclude noiseless
    baselines *= (num_episodes-1) / (num_episodes-2) # de-bias
    baseline = baselines[0]
    loss = tr.sum(tr.stack([
        -((rewards_to_go[b] - baselines) * tr.stack(log_probs[b])).sum() / (num_episodes - 1) / len(positions[0])
        for b in range(1,num_episodes)]))
    # loss = tr.tensor(0.)
    # for b in range(1,num_episodes): # exclude noiseless
    #     loss -= ((rewards_to_go[b] - baselines) * tr.stack(log_probs[b])).sum() / (num_episodes - 1) / len(positions[0])
    #     # loss = - ((rewards_to_go[b] - baselines) * tr.stack(log_probs[b])).sum() / (num_episodes - 1) / len(positions[0])
    #     # loss.backward(retain_graph=(b+1 < len(rewards)))
    loss.backward()
    print("    backprop took %fs" % (time.perf_counter() - perf_counter))

    return sym, rewards_to_go, baseline

def get_rvm_timesteps(rvm, problem, simulate=False, dbg=False):
    # run nvm for time-steps
    rvm.reset({"jnt": "rest"})
    rvm.mount("main") # sets tick counter to 0
    memorize_problem(rvm, problem)
    if dbg: rvm.dbg()
    while True:
        done = rvm.tick()
        if dbg: rvm.dbg()
        if simulate and rvm.registers["jnt"].content != rvm.registers["jnt"].old_content:
            position = rvm.ik[rvm.registers["jnt"].content]
            rvm.env.goto_position(position)
        if done: break
    return rvm.tick_counter

if __name__ == "__main__":
    
    tr.set_printoptions(precision=8, sci_mode=False, linewidth=1000)
    
    # prob_freq = "repetition"
    # prob_freq = "epoch"
    prob_freq = "minibatch"

    if prob_freq in ["repetition","epoch"]:
        num_repetitions = 5
        num_episodes = 32
        num_minibatches = 1
        num_epochs = 100
    if prob_freq == "minibatch":
        num_repetitions = 5
        num_episodes = 16
        num_minibatches = 16
        num_epochs = 64
    # num_repetitions = 1
    # num_episodes = 3
    # num_minibatches = 2
    # num_epochs = 2
    
    sizing = num_repetitions, num_epochs, num_minibatches, num_episodes
    
    run_exp = False
    showresults = True
    # run_exp = True
    # showresults = False
    showenv = False
    showtrained = True
    # tr.autograd.set_detect_anomaly(True)
    
    detach_gates = True
    # detach_gates = False
    # only_fails = True
    only_fails = False

    # learning_rates=[0.000001] # base only trainable
    # trainable = ["ik", "to", "tc", "po", "pc", "base"]

    # learning_rates=[0.0001, 0.00005] # all stack layers trainable
    # learning_rates=[0.0001, 0.000075, 0.00005] # all stack layers trainable
    # learning_rates=[0.00001, 0.0000075, 0.000005] # all stack layers trainable
    learning_rates=[0.0005, 0.0001] # all stack layers trainable
    trainable = ["ik", "to", "tc", "po", "pc", "right", "above", "base"]

    # learning_rates = [0.0005] # ik/motor layrs only
    # trainable = ["ik", "to", "tc", "po", "pc"]
    # # trainable = ["ik"]

    # sigma = 0.001 # stdev in random angular sampling (radians)
    sigma = 0.0174 # stdev in random angular sampling (radians)

    max_levels = 3
    num_blocks = 5
    num_bases = 5
    domain = bp.BlockStackingDomain(num_blocks, num_bases, max_levels)

    # # one failure case:
    # prob_freq = "once"
    # thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
    # goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
    # goal_thing_above = invert(goal_thing_below, num_blocks, num_bases)
    # for key, val in goal_thing_above.items():
    #     if val == "none": goal_thing_above[key] = "nil"
    # problem = bp.BlockStackingProblem(domain, thing_below, goal_thing_below, goal_thing_above)

    tracker_period = 5
    penalty_tracker = MovementPenaltyTracker(period=tracker_period)

    if run_exp:
        
        lr_results = {lr: list() for lr in learning_rates}
        for rep in range(num_repetitions):
            for learning_rate in learning_rates:
                print("Starting lr=%f" % learning_rate)
        
                results = lr_results[learning_rate]
                start_rep = time.perf_counter()
                results.append([])
                
                if prob_freq != "once": problem = domain.random_problem_instance()
                env = BlocksWorldEnv(show=False, step_hook=penalty_tracker.step_hook)
                env.load_blocks(problem.thing_below)
            
                # set up rvm and virtualize
                rvm = make_abstract_machine(env, domain)
                rvm.reset({"jnt": "rest"})
                rvm.mount("main")

                nvm = virtualize(rvm, σ=nv.default_activator, detach_gates=detach_gates)
                nvm.mount("main")
                W_init = {name: {0: nvm.net.batchify_weights(conn.W)} for name, conn in nvm.connections.items()}
                v_init = {name: {0: nvm.net.batchify_activities(reg.content)} for name, reg in nvm.registers.items()}
                v_init["jnt"][0] = nvm.net.batchify_activities(tr.tensor(rvm.ik["rest"]).float())

                # set up trainable connections
                conn_params = {name: W_init[name][0] for name in trainable}
                for p in conn_params.values(): p.requires_grad_()
                
                # set up optimizer, lump minibatch denominator into learning rate
                opt = tr.optim.Adam(conn_params.values(), lr=learning_rate / num_minibatches)
                
                # save original values for comparison
                orig_conns = {name: conn_params[name].detach().clone() for name in trainable}

                if prob_freq == "repetition":
                    if only_fails: problem, _ = find_failure_case(env, domain, sym_cutoff=-2)
                    else: problem = domain.random_problem_instance()
                
                if prob_freq in ["repetition", "once"]:
                    num_time_steps = get_rvm_timesteps(rvm, problem)

                for epoch in range(num_epochs):
                    start_epoch = time.perf_counter()
                    epoch_syms = []
                    epoch_baselines = []
                    epoch_rtgs = []
    
                    # print("Wik:")
                    # print(conn_params["ik"][:,:8])

                    if prob_freq == "epoch":
                        if only_fails: problem, _ = find_failure_case(env, domain)
                        else: problem = domain.random_problem_instance()
                        num_time_steps = get_rvm_timesteps(rvm, problem)

                    for minibatch in range(num_minibatches):
                        start_minibatch = time.perf_counter()
                        
                        if prob_freq == "minibatch":
                            if only_fails: problem, _ = find_failure_case(env, domain)
                            else: problem = domain.random_problem_instance()
                            num_time_steps = get_rvm_timesteps(rvm, problem)

                        sym, rewards_to_go, baseline = run_episodes(
                            problem, nvm, W_init, v_init, num_time_steps, num_episodes, penalty_tracker, sigma)
                        epoch_rtgs.extend([rtg[0].item() for rtg in rewards_to_go])
                        epoch_syms.extend(sym)
                        epoch_baselines.append(baseline)
                        
                        minibatch_time = time.perf_counter() - start_minibatch
                        print("  %d,%d,%d took %fs" % (rep, epoch, minibatch, minibatch_time))
        
                    # update params based on episodes
                    opt.step()
                    opt.zero_grad()

                    delta = {name: (orig_conns[name] - conn_params[name]).abs().max().item() for name in trainable}
                    results[-1].append((epoch_syms, epoch_baselines, epoch_rtgs, delta))
                    with open("pacb_%.2g.pkl" % learning_rate,"wb") as f: pk.dump((sizing, results), f)

                    avg_reward = np.mean(epoch_rtgs)
                    std_reward = np.std(epoch_rtgs)
                    print(" %d,%d: R = %f (+/- %f), dW: %s" % (rep, epoch, avg_reward, std_reward, delta))
                    epoch_time = time.perf_counter() - start_epoch
                    print(" %d,%d took %fs" % (rep, epoch, epoch_time))
    
                rep_time = time.perf_counter() - start_rep
                print("%d took %fs" % (rep, rep_time))
                # save model (pkl might need outside epoch loop??)
                with open("pacb_%.2g_state_%d.pkl" % (learning_rate, rep),"wb") as f: pk.dump((W_init, v_init), f)
    
                env.close()

    if showresults:
        import os
        import matplotlib.pyplot as pt

        for lr, learning_rate in enumerate(learning_rates):

            # with open("pacb.pkl","rb") as f: results = pk.load(f)
            # fname = "stack_trained/pacb_%.2g.pkl" % learning_rate
            fname = "pacb_%.2g.pkl" % learning_rate
            if os.path.exists(fname):
                with open(fname,"rb") as f: sizing, results = pk.load(f)
                # with open(fname,"rb") as f: results = pk.load(f)
    
            num_repetitions, num_epochs, num_minibatches, num_episodes = sizing
            num_repetitions = len(results)
            for rep in range(num_repetitions):
                epoch_syms, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                num_epochs = len(results[rep])
            
                # print("rep %d LC:" % rep)
                # for r,rewards in enumerate(epoch_syms): print(r, rewards)
                
                pt.subplot(num_repetitions, len(learning_rates), len(learning_rates)*rep+lr+1)
                if rep == 0: pt.title(str(learning_rate))
                baselines = [rewards[::num_episodes] for rewards in epoch_rtgs]
                signals = [[rewards[b*num_episodes+e] for b in range(num_minibatches) for e in range(1,num_episodes)] for rewards in epoch_rtgs]
                # pt.plot([np.mean(baselines) for baselines in epoch_baselines], 'b-')
                x, y = zip(*[(r+np.random.rand()*.5,reward)
                    for r in range(num_epochs) for reward in signals[r]])
                print(len(signals[0]))
                print(len(signals[1]))
                pt.plot(x, y, '.', c=(.5,)*3)
                # x, y = zip(*[(r,baseline) for r in range(num_epochs) for baseline in epoch_baselines[r]])
                # pt.plot(np.array(x)+.5, y, 'b.')
        
                # pt.plot(np.log(-np.array(rewards)))
                # pt.ylabel("log(-R)")
                pt.plot([np.mean(rewards) for rewards in signals], 'k-')
                pt.plot([np.mean(rewards) for rewards in baselines], 'b-')
                pt.plot([np.mean(rewards) for rewards in epoch_syms], 'k--')
                
                # trend line
                avg_rewards = np.mean(signals, axis=1)
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
                
                # constant lines
                pt.plot([0, len(avg_rewards)], avg_rewards[[0, 0]],'k-')
                pt.plot([0, len(avg_rewards)], (np.mean(baselines[0]),)*2, 'b-')
                pt.plot([0, len(avg_rewards)], (np.mean(epoch_syms[0]),)*2,'k--')

        pt.show()

        # one representative run
        if True:
            learning_rate = .0005
            # with open("stack_trained/pacb_%.2g.pkl" % learning_rate,"rb") as f: results = pk.load(f)
            with open("pacb_%.2g.pkl" % learning_rate,"rb") as f: sizing, results = pk.load(f)
            rep = 0
            num_epochs = len(results[rep])
            epoch_syms, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])

            pt.figure(figsize=(10, 3))
            
            ax = pt.subplot(1,2,1)
            pt.plot([0, num_epochs],[np.mean(epoch_rtgs[0][1:])]*2, '-', c=(.25,)*3)
            pt.plot([0, num_epochs],[np.mean(epoch_syms[0][1:])]*2, '--', c=(.25,)*3)
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'k-')
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_syms], 'k--')
            pt.legend(["avg. reward", "avg. symbolic"], framealpha=1, fontsize=12, loc="center left")
            x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rtgs[r][1:]])
            pt.scatter(x, y, s=4, c=((.75,)*3,), zorder=-1)
            pt.xlabel("Training iteration", fontsize=12)
            pt.ylabel("Performance", fontsize=12)
            pt.yticks(range(0,-8,-1))
            ax.tick_params(labelright=True, right=True)
            pt.title("(A)")

            pt.subplot(1,2,2)
            pt.plot([d["tc"] for d in deltas], 'k-', label="tc")
            pt.plot([d["ik"] for d in deltas], 'k--', label="ik,to,po,pc")
            pt.plot([d["to"] for d in deltas], 'k--')
            pt.plot([d["po"] for d in deltas], 'k--')
            pt.plot([d["pc"] for d in deltas], 'k--')
            pt.plot([d["above"] for d in deltas], 'k-.', label="above")
            pt.plot([d["right"] for d in deltas], 'k:', label="right,next")
            pt.plot([d["base"] for d in deltas], 'k:')
            # pt.plot([d["to"] for d in deltas[::8]], 'k+-', label="to")
            # pt.plot([d["po"] for d in deltas[::8]], 'k^-', label="po")
            # pt.plot([d["pc"] for d in deltas[::8]], 'kv-', label="pc")
            # pt.plot([d["tc"] for d in deltas[::8]], 'ko-', label="tc")
            # pt.plot([d["ik"] for d in deltas[::8]], 'k+--', label="ik")
            # pt.plot([d["right"] for d in deltas[::8]], 'k^--', label="right")
            # pt.plot([d["above"] for d in deltas[::8]], 'kv--', label="above")
            # pt.plot([d["base"] for d in deltas[::8]], 'ko--', label="base")
            pt.xlabel("Training iteration", fontsize=12)
            pt.ylabel("Weight changes", fontsize=12)
            pt.legend(framealpha=1, fontsize=12, loc="upper left")
            pt.title("(B)")

            pt.tight_layout()
            pt.savefig("pacb_one_rep.pdf")
            pt.show()
            pt.close()
            print(deltas[-1])
            # input('.')

            # different learning rates
            fig = pt.figure(figsize=(10, 2.75))
            gs = fig.add_gridspec(1,3)
            fname = "pacb_%.2g.pkl" % learning_rate
            with open(fname,"rb") as f: sizing, results = pk.load(f)
            num_repetitions = len(results)
            fig.add_subplot(gs[:,:2])
            num_epochs = len(results[0])
            syms = np.nan*np.ones((num_repetitions, num_epochs))
            rews = np.nan*np.ones((num_repetitions, num_epochs))
            for rep in range(num_repetitions):
                epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                # syms[rep,:len(results[rep])] = np.array([np.mean(rewards[1:]) for rewards in epoch_rewards])
                # rews[rep,:len(results[rep])] = np.array([np.mean(rewards[1:]) for rewards in epoch_rtgs])
                syms[rep,:len(results[rep])] = np.array([np.mean(rewards[::num_episodes]) for rewards in epoch_rewards])
                rews[rep,:len(results[rep])] = np.array([np.mean(rewards[::num_episodes]) for rewards in epoch_rtgs])
            pt.plot(rews.T, '-', color=(.75, .75, .75))
            pt.plot(syms.T, '--', color=(.75, .75, .75))
            pt.plot([0, num_epochs], rews.mean(axis=0)[[0,0]], '-', color=(.25, .25, .25))
            pt.plot([0, num_epochs], syms.mean(axis=0)[[0,0]], '--', color=(.25, .25, .25))
            h1 = pt.plot(rews.mean(axis=0), 'k-')
            h2 = pt.plot(syms.mean(axis=0), 'k--')
            # pt.xlim([0, 100])
            # pt.ylim([-4, 0])
            pt.legend([h1[0], h2[0]], ["reward", "symbolic"], framealpha=1, fontsize=12, loc="lower right", ncol=2)
            # pt.title("LR = %.1e" % learning_rate)
            # pt.title("LR = %.2g" % learning_rate, fontsize=12)
            pt.xlabel("Training iteration", fontsize=12)
            pt.ylabel("Performance", fontsize=12)
            if lr == 0: pt.ylabel("Average performance", fontsize=12)
            pt.tick_params(labelright=True, right=True)
            pt.title("(A)")

            fig.add_subplot(gs[:,2])
            x_sym, y_sym = [], []
            x_rew, y_rew = [], []
            x_avg_sym, y_avg_sym = [], []
            x_avg_rew, y_avg_rew = [], []
            for rep in range(num_repetitions):
                epoch_rewards, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                # # x_sym.append(np.mean(epoch_rewards[0][1:]))
                # # y_sym.append(np.mean(epoch_rewards[-1][1:]))
                # # x_rew.append(np.mean(epoch_rtgs[0][1:]))
                # # y_rew.append(np.mean(epoch_rtgs[-1][1:]))
                x_avg_sym.append(np.stack([r[::num_episodes] for r in epoch_rewards[:5]]).mean())
                y_avg_sym.append(np.stack([r[::num_episodes] for r in epoch_rewards[-5:]]).mean())
                x_avg_rew.append(np.stack([r[::num_episodes] for r in epoch_rtgs[:5]]).mean())
                y_avg_rew.append(np.stack([r[::num_episodes] for r in epoch_rtgs[-5:]]).mean())
                # x_sym.extend([np.mean(r[::num_episodes]) for r in epoch_rewards[:5]])
                # y_sym.extend([np.mean(r[::num_episodes]) for r in epoch_rewards[-5:]])
                # x_rew.extend([np.mean(r[::num_episodes]) for r in epoch_rtgs[:5]])
                # y_rew.extend([np.mean(r[::num_episodes]) for r in epoch_rtgs[-5:]])
            # pt.scatter(x_rew, y_rew, marker='o', color=(.75,)*3)
            # pt.scatter(x_sym, y_sym, marker='o', ec=(.75,)*3, fc="none")
            h1 = pt.scatter(x_avg_rew, y_avg_rew, marker='o', color='k')
            h2 = pt.scatter(x_avg_sym, y_avg_sym, marker='o', ec='k', fc="none")
            # pt.plot([min(x_rew)-.1, 0.1], [min(x_rew)-.1, 0.1], ':', c=(.25,)*3)
            # pt.xlim([min(x_rew)-.1, 0.1])
            # pt.ylim([min(x_rew)-.1, 0.1])
            pt.plot([min(x_avg_rew)-.1, 0.1], [min(x_avg_rew)-.1, 0.1], ':', c=(.25,)*3)
            pt.xlim([min(x_avg_rew)-.1, 0.1])
            pt.ylim([min(x_avg_rew)-.1, 0.1])
            pt.xticks([-1.5, -1, -.5, -0])
            pt.yticks([-1.5, -1, -.5, -0])
            pt.legend([h1, h2], ["reward", "symbolic"], loc="lower right", fontsize=12)
            pt.xlabel("First 5 Iterations", fontsize=12)
            pt.ylabel("Last 5 Iterations", fontsize=12)
            pt.title("(B)")

            pt.tight_layout()
            pt.savefig("pacb_all_reps.pdf")
            pt.show()
            pt.close()
            
            improves = np.array([y/x for x,y in zip(x_avg_sym, y_avg_sym)])
            print(improves)
            print(improves.mean())

    if showtrained:
        
        showpb = False
        runpb = False
        tr_file = "show_trained_data.pkl"

        lr = 0.0005
        rep = 4

        class Tracker:
            def __init__(self, period):
                self.period = period
                self.penalties = []
                self.grips = []
                self.joints = []
                self.counter = 0
            def step_hook(self, env, action):
                if self.counter % self.period == 0:
                    mp = env.movement_penalty()
                    self.penalties.append(mp)
                    s_5, s_7 = pb.getLinkStates(env.robot_id, [5, 7])
                    xyz_5, xyz_7 = s_5[0], s_7[0]
                    p_grip = np.array([xyz_5, xyz_7]).mean(axis=0)
                    self.grips.append(p_grip)
                    self.joints.append(env.get_position())
                self.counter += 1
        
        # tracker = Tracker(period=tracker_period)
        tracker = Tracker(period=1)

        if runpb:

            with open("pacb_%.2g_state_%d.pkl" % (lr, rep), "rb") as f: (W_init, v_init) = pk.load(f)
    
            # env = BlocksWorldEnv(show=showpb)
            # problem, _ = find_failure_case(env, domain)
            # print(problem.thing_below)
            # print(problem.goal_thing_below)
            # print(problem.goal_thing_above)
            # env.close()
    
            thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
            goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
            goal_thing_above = domain.invert(goal_thing_below)
            problem = bp.BlockStackingProblem(domain, thing_below, goal_thing_below, goal_thing_above)
    
            env = BlocksWorldEnv(show=showpb, step_hook = tracker.step_hook)
            yaw, pitch, dist, targ = 0, -7, 1.1, (0, 0.75, 0) # got from running blocks_world.py
            pb.resetDebugVisualizerCamera(dist, yaw, pitch, targ)
    
            env.reset()
            env.load_blocks(problem.thing_below)
            if showpb: input('...')
    
            # set up rvm and virtualize
            rvm = make_abstract_machine(env, domain)
            rvm.reset({"jnt": "rest"})
            rvm.mount("main")
            memorize_problem(rvm, problem)
    
            nvm = virtualize(rvm, σ=nv.default_activator, detach_gates=detach_gates)
            nvm.mount("main")
            memorize_problem(nvm, problem)
            for name in ["obj","loc","goal"]:
                W_init[name][0] = nvm.connections[name].W.unsqueeze(dim=0)
    
            while True:
                done = rvm.tick()
                if rvm.registers["jnt"].content != rvm.registers["jnt"].old_content:
                    position = rvm.ik[rvm.registers["jnt"].content]
                    env.goto_position(position, speed=1.5)
                if done: break
            num_time_steps = rvm.tick_counter
            rvm_sym = compute_symbolic_reward(env, problem.goal_thing_below)
            rvm_mps, rvm_joints, rvm_grips = tracker.penalties[10:], tracker.joints, tracker.grips
            tracker.penalties, tracker.joints, tracker.grips = [], [], []
            print(rvm_sym, sum(rvm_mps))
            if showpb: input('...')
    
            env.reset()
            env.load_blocks(problem.thing_below)
            nvm.net.clear_ticks()
            for t in range(num_time_steps):
                nvm.net.tick(W_init, v_init)
                if t > 1 and nvm.decode("tar", t-2) != nvm.decode("tar", t-1):
                    position = nvm.net.activities["jnt"][t][0,:,0]
                    env.goto_position(position.detach().numpy(), speed=1.5)
            nvm_sym = compute_symbolic_reward(env, problem.goal_thing_below)
            nvm_mps, nvm_joints, nvm_grips = tracker.penalties[10:], tracker.joints, tracker.grips
            if showpb: input('...')
            
            tr_res = (rvm_sym, rvm_mps, rvm_joints, rvm_grips, nvm_sym, nvm_mps, nvm_joints, nvm_grips)
            with open(tr_file, "wb") as f: pk.dump(tr_res, f)

        with open(tr_file, "rb") as f: tr_res = pk.load(f)
        rvm_sym, rvm_mps, rvm_joints, rvm_grips, nvm_sym, nvm_mps, nvm_joints, nvm_grips = tr_res
        print(rvm_sym, sum(rvm_mps), nvm_sym, sum(nvm_mps))

        import matplotlib.pyplot as pt
        fig = pt.figure(figsize=(10,5))
        gs = fig.add_gridspec(2,5)

        # pt.subplot(1,3,1)
        # pt.plot(np.array(rvm_joints), '-', c=(.6,)*3)
        # pt.plot(np.array(nvm_joints), 'k-')

        # pt.subplot(1,3,2)
        fig.add_subplot(gs[0,:3])
        env = BlocksWorldEnv(show=False)
        env.bases = ["t%d" % b for b in range(num_bases)]
        env.blocks = ["b%d" % b for b in range(num_blocks)]
        for name in env.bases:
            pos, _ = env.placement_of(name)
            th = np.arctan2(pos[1], pos[0])
            for h in range(max_levels):
                pt.plot(
                    [th-.1, th-.1, th+.1, th+.1, th-.1],
                    [h*.02, h*.02+.02, h*.02+.02, h*.02, h*.02],
                    ':', c=(.75,)*3)
        rvm_grips, nvm_grips = np.array(rvm_grips), np.array(nvm_grips)
        # rvm_grips, nvm_grips = rvm_grips[:300,:], nvm_grips[:300,:]
        rvm_polar = np.stack([np.arctan2(*rvm_grips[:,[1, 0]].T), rvm_grips[:,2]])
        nvm_polar = np.stack([np.arctan2(*nvm_grips[:,[1, 0]].T), nvm_grips[:,2]])
        pt.plot(*rvm_polar, '-', c=(.6,)*3)
        pt.plot(*nvm_polar, 'k-')
        pt.ylim([0, .1])
        pt.ylabel("Height", fontsize=14)
        pt.xlabel("Polar angle", fontsize=14)
        pt.title("(A)", fontsize=14)

        fig.add_subplot(gs[1,:3])
        for name in env.bases:
            pos, _ = env.placement_of(name)
            th = np.arctan2(pos[1], pos[0])
            for h in range(max_levels):
                pt.plot(
                    [th-.1, th-.1, th+.1, th+.1, th-.1],
                    [h*.02, h*.02+.02, h*.02+.02, h*.02, h*.02],
                    ':', c=(.75,)*3)
        pt.plot(*rvm_polar, '-', c=(.6,)*3)
        pt.plot(*nvm_polar, 'k-')
        pt.xlim([-2.05, -1.9])
        pt.ylim([.02, .06])
        pt.ylabel("Height", fontsize=14)
        pt.xlabel("Polar angle", fontsize=14)
        pt.title("(B)", fontsize=14)
        
        fig.add_subplot(gs[:,3:])
        # pt.subplot(1,3,3)
        h1 = pt.plot(rvm_mps, '-', c=(.6,)*3, label="RVM")
        h2 = pt.plot(nvm_mps, 'k-', label="NVM")
        pt.xlabel("Simulation steps", fontsize=14)
        pt.ylabel("Movement penalty", fontsize=14)
        pt.legend([h1[0], h2[0]], ["Untrained", "Trained"], loc="upper right", fontsize=14)
        pt.title("(C)", fontsize=14)
        
        pt.tight_layout()
        pt.savefig("tr_mp.pdf")
        pt.show()
        pt.close()

