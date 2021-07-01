import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from abstract_machine import make_abstract_machine, memorize_env, memorize_problem
from restack import invert, compute_spatial_reward, compute_symbolic_reward
from nvm import virtualize
import neural_virtual_machine as nv
import block_stacking_problem as bp

np.set_printoptions(linewidth=1000)

def calc_reward(sym_reward, spa_reward):
    # reward = sym_reward + 0.1*spa_reward
    reward = sym_reward
    # reward = spa_reward
    return reward

class PenaltyTracker:
    def __init__(self, period):
        self.period = period
        self.reset()
    def reset(self):
        self.penalty = 0
        self.counter = 0
    def step_hook(self, env, action):
        if self.counter % self.period == 0:
            mp = env.movement_penalty()
            self.penalty += mp * self.period
            # print("penalty: %.5f" % mp)
        self.counter += 1

def run_episodes(problem, nvm, W_init, v_init, num_time_steps, num_episodes, penalty_tracker, sigma):

    memorize_problem(nvm, problem)
    for name in ["obj","loc","goal"]:
        W_init[name][0] = nvm.connections[name].W.repeat(num_episodes, 1, 1)

    perf_counter = time.perf_counter()
    # W, v = nvm.net.run(W_init, v_init, num_time_steps)
    nvm.net.clear_ticks()
    for t in range(num_time_steps):
        nvm.net.tick(W_init, v_init)
        # nvm.pullback(t)
        # nvm.dbg()
        # input('.')
    W, v = nvm.net.weights, nvm.net.activities
    print("    NVM run took %fs" % (time.perf_counter() - perf_counter))
    
    perf_counter = time.perf_counter()
    positions, log_probs = {}, {}
    tar = nvm.registers["tar"]
    for b in range(num_episodes):
        positions[b], log_probs[b] = [], []
        for t in range(2, num_time_steps):
            if nvm.decode("tar", t-2, b) != nvm.decode("tar", t-1, b):
                mu = v["jnt"][t][b,:,0]
                dist = tr.distributions.normal.Normal(mu, sigma)
                position = dist.sample() if b > 0 else mu # first episode noiseless
                # position = dist.sample()
                positions[b].append(position)
                log_probs[b].append(dist.log_prob(position).sum()) # multivariate white noise
            if len(positions[b]) > len(positions[0]): break # avoid devolution into moving every step
        if any([tr.isnan(lp) for lp in log_probs[b]]) or len(positions[b]) != len(positions[0]):
            print(" "*6, log_probs[b])
            print(len(positions[b]), len(positions[0]))
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
    baseline = baselines[0]
    loss = tr.tensor(0.)
    for b in range(1,num_episodes): # exclude noiseless
        loss -= ((rewards_to_go[b] - baselines) * tr.stack(log_probs[b])).sum() / (num_episodes - 1)
        # loss = - ((rewards_to_go[b] - baselines) * tr.stack(log_probs[b])).sum() / (num_episodes - 1)
        # loss.backward(retain_graph=(b+1 < len(rewards)))
    loss.backward()
    print("    backprop took %fs" % (time.perf_counter() - perf_counter))

    return sym, rewards_to_go, baseline

if __name__ == "__main__":
    
    tr.set_printoptions(precision=8, sci_mode=False, linewidth=1000)
    
    # num_repetitions = 5
    # num_episodes = 5
    # num_minibatches = 6
    # num_epochs = 100
    num_repetitions = 1
    num_episodes = 3
    num_minibatches = 2
    num_epochs = 2
    
    # run_exp = False
    # showresults = True
    run_exp = True
    showresults = False
    showenv = False
    showtrained = False
    # tr.autograd.set_detect_anomaly(True)
    
    detach_gates = True
    # detach_gates = False

    # # learning_rates=[0.0001, 0.00005] # all stack layers trainable
    # # learning_rates=[0.0001, 0.000075, 0.00005] # all stack layers trainable
    # # learning_rates=[0.00001, 0.0000075, 0.000005] # all stack layers trainable
    # learning_rates=[0.00005] # all stack layers trainable
    # trainable = ["ik", "to", "tc", "po", "pc", "right", "above", "base"]

    # learning_rates=[0.00005, 0.00001] # base only trainable, 5 works better than 1
    # trainable = ["ik", "to", "tc", "po", "pc", "base"]

    learning_rates = [0.0001] # ik/motor layrs only
    trainable = ["ik", "to", "tc", "po", "pc"]
    # trainable = ["ik"]

    sigma = 0.001 # stdev in random angular sampling (radians)

    max_levels = 3
    num_blocks = 5
    num_bases = 5
    domain = bp.BlockStackingDomain(num_blocks, num_bases, max_levels)

    # # one failure case:
    # thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
    # goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}
    # goal_thing_above = invert(goal_thing_below, num_blocks, num_bases)
    # for key, val in goal_thing_above.items():
    #     if val == "none": goal_thing_above[key] = "nil"
    # problem = bp.BlockStackingProblem(domain, thing_below, goal_thing_below, goal_thing_above)

    penalty_tracker = PenaltyTracker(period=5)

    if run_exp:
        
        lr_results = {lr: list() for lr in learning_rates}
        for rep in range(num_repetitions):
            for learning_rate in learning_rates:
                print("Starting lr=%f" % learning_rate)
        
                results = lr_results[learning_rate]
                start_rep = time.perf_counter()
                results.append([])
                
                problem = domain.random_problem_instance()
                env = BlocksWorldEnv(show=False, step_hook=penalty_tracker.step_hook)
                env.load_blocks(problem.thing_below)
            
                # set up rvm and virtualize
                rvm = make_abstract_machine(env, num_bases, max_levels)
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
                
                # run nvm for time-steps
                memorize_env(rvm, problem.goal_thing_above)
                num_time_steps = rvm.run()
                # env.close()
                
                for epoch in range(num_epochs):
                    start_epoch = time.perf_counter()
                    epoch_syms = []
                    epoch_baselines = []
                    epoch_rtgs = []
    
                    # print("Wik:")
                    # print(conn_params["ik"][:,:8])
                    
                    for minibatch in range(num_minibatches):
                        start_minibatch = time.perf_counter()
                        
                        problem = domain.random_problem_instance()
                    
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
                    with open("pacb_%.2g.pkl" % learning_rate,"wb") as f: pk.dump(results, f)

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
        
        # one representative run
        if False:
            learning_rate = .000075
            # with open("stack_trained/pacb_%.2g.pkl" % learning_rate,"rb") as f: results = pk.load(f)
            with open("pacb_%.2g.pkl" % learning_rate,"rb") as f: results = pk.load(f)
            rep = 0
            num_epochs = len(results[rep])
            epoch_syms, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
            pt.figure(figsize=(5,1.75))
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'k-')
            pt.plot([np.mean(rewards[1:]) for rewards in epoch_syms], 'k--')
            from matplotlib.lines import Line2D
            h1 = Line2D([], [], linestyle="-", color="k")
            h2 = Line2D([], [], linestyle="-", color="k")
            pt.legend([h1, h2], ["avg. reward", "avg. symbolic"], framealpha=1)
            x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rtgs[r][1:]])
            pt.plot(x, y, '.', c=(.75, .75, .75), zorder=-1)
            pt.xlabel("Training epoch")
            pt.ylabel("Performance")
            pt.tight_layout()
            pt.savefig("pacb_one.pdf")
            pt.show()

            pt.figure(figsize=(5,1.75))
            pt.plot([d["ik"] for d in deltas], 'k-', label="ik")
            pt.plot([d["to"] for d in deltas], 'k--', label="to,tc,pc")
            pt.plot([d["pc"] for d in deltas], 'k--')
            pt.plot([d["tc"] for d in deltas], 'k--')
            pt.plot([d["right"] for d in deltas], 'k:', label="right,above,next")
            pt.plot([d["above"] for d in deltas], 'k:')
            pt.plot([d["base"] for d in deltas], 'k:')
            pt.xlabel("Training epoch")
            pt.ylabel("Weight changes")
            pt.legend(framealpha=1)
            pt.tight_layout()
            pt.savefig("pacb_deltas.pdf")
            pt.show()
        
            # different learning rates
            pt.figure(figsize=(5,4))
            for lr, learning_rate in enumerate(learning_rates):
                # fname = "stack_trained/pacb_%.2g.pkl" % learning_rate
                fname = "pacb_%.2g.pkl" % learning_rate
                with open(fname,"rb") as f: results = pk.load(f)    
                num_repetitions = len(results)
                pt.subplot(len(learning_rates), 1, lr+1)
                num_epochs = len(results[0])
                syms = np.empty((num_repetitions, num_epochs))
                for rep in range(num_repetitions):
                    epoch_syms, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                    syms[rep,:num_epochs] = np.array([np.mean(rewards[1:]) for rewards in epoch_syms])
                pt.plot(syms.T, '-', color=(.75, .75, .75))
                pt.plot(syms.mean(axis=0), 'k-')
                # pt.xlim([0, 150])
                # pt.ylabel("LR = %s" % str(learning_rate))
                pt.ylabel("%.1e" % learning_rate)
                if lr == 0: pt.title("Symbolic performance with different learning rates")
                if lr+1 == len(learning_rates): pt.xlabel("Training epoch")
            pt.tight_layout()
            pt.savefig("pabc_many.pdf")
            pt.show()

        for lr, learning_rate in enumerate(learning_rates):

            # with open("pacb.pkl","rb") as f: results = pk.load(f)
            # fname = "stack_trained/pacb_%.2g.pkl" % learning_rate
            fname = "pacb_%.2g.pkl" % learning_rate
            if os.path.exists(fname):
                with open(fname,"rb") as f: results = pk.load(f)
    
            num_repetitions = len(results)
            for rep in range(num_repetitions):
                epoch_syms, epoch_baselines, epoch_rtgs, deltas = zip(*results[rep])
                num_epochs = len(results[rep])
            
                # print("rep %d LC:" % rep)
                # for r,rewards in enumerate(epoch_syms): print(r, rewards)
                
                pt.subplot(num_repetitions, len(learning_rates), len(learning_rates)*rep+lr+1)
                if rep == 0: pt.title(str(learning_rate))
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_rtgs], 'k-')
                pt.plot([rewards[0] for rewards in epoch_rtgs], 'b-')
                pt.plot([np.mean(rewards[1:]) for rewards in epoch_syms], 'k--')
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
        with open("pacb_state_%d.pkl" % rep, "rb") as f: (init_regs, init_conns) = pk.load(f)

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

