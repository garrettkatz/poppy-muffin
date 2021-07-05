import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv, MovementPenaltyTracker
from abstract_machine import make_abstract_machine, memorize_problem
from nvm import virtualize
import neural_virtual_machine as nv
import block_stacking_problem as bp
from restack import compute_symbolic_reward
from failure_case import find_failure_case

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

def run_nvm(nvm, batch_time_steps, W_init, v_init, dbg=False):
    nvm.net.clear_ticks()
    for t in range(max(batch_time_steps)):
        nvm.net.tick(W_init, v_init)
        if dbg:
            nvm.pullback(t)
            nvm.dbg()
            input('.')
    return nvm.net.weights, nvm.net.activities

if __name__ == "__main__":
    
    tr.set_printoptions(precision=8, sci_mode=False, linewidth=1000)
    
    showresults = True
    run_exp = False

    results_file = "erl.pkl"

    detach_gates = True
    sigma = 0.0174 # stdev in random angular sampling (radians)

    batch_size = 1
    num_episodes = 8
    num_batch_iters = 3
    
    num_descent_iters = 3
    descent_error_tol = 6*(sigma/10)**2
    descent_lr = 1.0
    optimizer = tr.optim.SGD
    # descent_lr = 0.001
    # optimizer = tr.optim.Adam

    max_levels = 3
    num_blocks = 5
    num_bases = 5
    
    # prob_freq = "batch"
    prob_freq = "once"
    only_fails = True
    
    if run_exp:

        domain = bp.BlockStackingDomain(num_blocks, num_bases, max_levels)
        mp_tracker = MovementPenaltyTracker(period=5)
        env = BlocksWorldEnv(show=False, step_hook=mp_tracker.step_hook)
    
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
        inputable = ("obj","loc","goal")
        # trainable = ["ik", "to", "tc", "po", "pc", "right", "above", "base"]
        trainable = tuple(set(nvm.connections.keys()) - set(nvm.net.plastic_connections + inputable))
        train_params = {name: W_init[name][0] for name in trainable}
        for p in train_params.values(): p.requires_grad_()
    
        # save original values for comparison
        orig_params = {name: train_params[name].clone().detach() for name in trainable}
        
        # size up for problem instance batches
        for name in inputable: W_init[name][0] = tr.zeros((batch_size,) + W_init[name][0].shape[1:])
        
        if prob_freq == "once":
            if only_fails:
                problems, sym_rewards = zip(*[find_failure_case(env, domain, sym_cutoff=-2) for b in range(batch_size)])
                print(" sym rewards: %s" % str(sym_rewards))
            else: problems = [domain.random_problem_instance() for b in range(batch_size)]

        results = []
        for batch_iter in range(num_batch_iters):
            batch_iter_counter = time.perf_counter()
    
            # setup weights for problem instances
            if prob_freq == "batch":
                if only_fails:
                    problems, sym_rewards = zip(*[find_failure_case(env, domain, sym_cutoff=-2) for b in range(batch_size)])
                    print(" sym rewards: %s" % str(sym_rewards))
                else: problems = [domain.random_problem_instance() for b in range(batch_size)]

            batch_weights = {name: list() for name in inputable}
            batch_time_steps = []
            for b, problem in enumerate(problems):
                batch_time_steps.append(get_rvm_timesteps(rvm, problem, simulate=False, dbg=False))
                memorize_problem(nvm, problem)
                for name in inputable: batch_weights[name].append(nvm.connections[name].W)
            for name in inputable:
                W_init[name][0] = tr.stack(batch_weights[name]).clone().detach()
        
            # run nvm on instances
            perf_counter = time.perf_counter()
            with tr.no_grad():
                W, v = run_nvm(nvm, batch_time_steps, W_init, v_init)
            joint_output, time_index = {}, {}
            for b, num_time_steps in enumerate(batch_time_steps):
                joint_output[b], time_index[b] = [], []
                for t in range(2, num_time_steps):
                    if nvm.decode("tar", t-2, b) != nvm.decode("tar", t-1, b):
                        joint_output[b].append(nvm.net.activities["jnt"][t][b,:,0])
                        time_index[b].append(t)
            print("    NVM run took %fs" % (time.perf_counter() - perf_counter))
        
            # generate random nearby actions
            perf_counter = time.perf_counter()
            positions = {b: {e: list() for e in range(num_episodes)} for b in range(batch_size)}
            for b in range(batch_size):
                for k, mu in enumerate(joint_output[b]):
                    dist = tr.distributions.normal.Normal(mu, sigma)
                    for e in range(num_episodes):
                        position = dist.sample() if e > 0 else mu # noiseless first episode
                        positions[b][e].append(position)
            num_motions = [len(positions[b][0]) for b in range(batch_size)]
            print("    actions took %fs (%d-%d motions)" % (time.perf_counter() - perf_counter, min(num_motions), max(num_motions)))
        
            # simulate to get rewards
            perf_counter = time.perf_counter()
            rewards, sym = tuple(np.zeros((batch_size, num_episodes)) for _ in [0,1])
            for b, problem in enumerate(problems):
                for e in range(num_episodes):
                    env.reset()
                    env.load_blocks(problem.thing_below)
                    for position in positions[b][e]:
                        mp_tracker.reset()
                        env.goto_position(position.detach().numpy(), speed=1.5)
                        rewards[b,e] -= mp_tracker.penalty
                    sym[b,e] = compute_symbolic_reward(env, problem.goal_thing_below)
                    rewards[b,e] += sym[b,e]
                    # print("      %d,%d: %f" % (b,e,rewards[b,e]))
            avg_reward = rewards[:,0].mean() # noiseless episodes
            opt_index = rewards.argmax(axis=1)
            print("    simulation rewards took %fs" % (time.perf_counter() - perf_counter))
            print("    %d problems with better noisy episodes" % (opt_index > 0).sum())
            
            final_batch = batch_iter+1 == num_batch_iters
            all_opt = (opt_index == 0).all()
            if final_batch or all_opt:
                print("   batch iter %d took %fs, avg reward = %f" % (batch_iter, time.perf_counter() - batch_iter_counter, avg_reward))
                results.append((avg_reward, rewards, {}, 0, []))
                with open(results_file, "wb") as f: pk.dump(results, f)
            if final_batch: break
            if all_opt: continue
            
            # descent on error from optimal
            descent_counter = time.perf_counter()
            targets = [tr.stack(positions[b][opt_index[b]]).clone().detach() for b in range(batch_size)]
            W_start = {name: W_init[name][0].clone().detach() for name in trainable}
            opt = optimizer(train_params.values(), lr=descent_lr)
            
            descent_log = []
            for descent_iter in range(num_descent_iters):
                descent_iter_counter = time.perf_counter()
    
                run_counter = time.perf_counter()
                W, v = run_nvm(nvm, batch_time_steps, W_init, v_init)
                outputs = [
                    tr.stack([v["jnt"][t][b,:,0] for t in time_index[b]])
                    for b in range(batch_size)]
                # print("      %d NVM run took %f" % (descent_iter, time.perf_counter() - run_counter))
                
                objective_counter = time.perf_counter()
                objective = tr.sum(tr.stack([tr.sum((outputs[b] - targets[b])**2) for b in range(batch_size)]))
                objective /= sum(batch_time_steps)
                # print("      %d L=%f vs %f, took %f" % (descent_iter, objective, descent_error_tol, time.perf_counter() - objective_counter))
    
                backward_counter = time.perf_counter()
                objective.backward()
                # print("      %d L grad took %f" % (descent_iter, time.perf_counter() - backward_counter))
            
                update_counter = time.perf_counter()
                grad_sq_norm = tr.sum(tr.stack([tr.tensor(0.) if p.grad is None else tr.sum(p.grad**2) for p in train_params.values()]))
                descent_log.append((objective.item(), grad_sq_norm.item()))
                if objective > descent_error_tol:

                    if all([p.grad is None for p in train_params.values()]): raise ValueError("All grads none!")
                    # for name, p in train_params.items():
                    #     if p.grad is None: continue
                    #     p.data -= p.grad * descent_lr
                    #     p.grad *= 0
                    opt.step()
                    opt.zero_grad()
                    print("      %d update took %f" % (descent_iter, time.perf_counter() - update_counter))
            
                print("     descent iter %d took %fs, L = %f, |grad L|**2 = %f" % (
                    descent_iter, time.perf_counter() - descent_iter_counter, objective, grad_sq_norm))
                
                if objective <= descent_error_tol: break
                
            sq_delta = tr.sum(tr.stack([tr.sum((W_init[name][0] - W_start[name][0])**2) for name in trainable]))
            print("    descent took %fs, |delta W|**2 = %f, |v-th|**2 = %f" % (
                time.perf_counter() - descent_counter, sq_delta, objective))
                
            print("   batch iter %d took %fs, avg reward = %f" % (batch_iter, time.perf_counter() - batch_iter_counter, avg_reward))
            delta = {name: (orig_params[name] - train_params[name]).abs().max().item() for name in trainable}
            print("    delta from 0 = %f" % max(delta.values()))
            
            results.append((avg_reward, rewards, delta, sq_delta, descent_log))
            with open(results_file, "wb") as f: pk.dump(results, f)
    
    if showresults:
        import matplotlib.pyplot as pt
        with open(results_file, "rb") as f: results = pk.load(f)
        x_avg_rewards, y_avg_rewards = [], []
        x_rewards, y_rewards = [], []
        x_descent, y_descent = [], []
        for avg_reward, rewards, delta, sq_delta, descent_log in results:
            x_avg_rewards.append(len(x_descent))
            y_avg_rewards.append(avg_reward)
            x_rewards += list((np.random.rand(*rewards.shape)*.5 + len(x_descent)).flat)
            y_rewards += list(rewards.flat)
            for objective, grad_sq_norm in descent_log:
                x_descent.append(len(x_descent))
                # y_descent.append(objective)
                y_descent.append(grad_sq_norm)
        pt.subplot(2,1,1)
        pt.plot(x_rewards, y_rewards, '.', c=(.75,)*3)
        pt.plot(x_avg_rewards, y_avg_rewards, 'ko-')
        pt.xlim([-1, x_avg_rewards[-1]+1])
        pt.subplot(2,1,2)
        pt.plot(x_descent, y_descent, 'k-')
        pt.xlim([-1, x_avg_rewards[-1]+1])
        pt.show()
        
