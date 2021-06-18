import pickle as pk
import numpy as np
import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import random_problem_instance, compute_spatial_reward
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize

if __name__ == "__main__":

    num_blocks, max_levels, num_bases = 6, 3, 6
    num_episodes = 30
    num_epochs = 50
    
    run_exp = True
    if run_exp:
        results = []

        env = BlocksWorldEnv(show=False)
        env.load_blocks({"b%d" % n: "t%d" % n for n in range(num_bases)}) # placehold for nvm init
    
        # tr.autograd.set_detect_anomaly(True)
    
        am = make_abstract_machine(env, num_bases, max_levels)
        nvm = virtualize(am)
        # print(nvm.size())
        # input('.')
    
        # set up trainable connections
        conn_params = {
            "ik": nvm.connections["ik"].W,
        }
        for p in conn_params.values(): p.requires_grad_()
        opt = tr.optim.Adam(conn_params.values(), lr=0.00001)
        
        orig_ik = nvm.connections["ik"].W.clone()
    
        # env.reset()
        # thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)

        # failure case:
        thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
        goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}

        # thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
        # thing_below.update({"b1": "b0", "b2": "b3"})    
        # goal_thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
        # goal_thing_below.update({"b1": "b2", "b2": "b0"})
    
        goal_thing_above = nvm.env.invert(goal_thing_below)
        for key, val in goal_thing_above.items():
            if val == "none": goal_thing_above[key] = "nil"
    
        lc = []
        for epoch in range(num_epochs):
            epoch_rewards = []
            for episode in range(num_episodes):
        
                env.reset()
                env.load_blocks(thing_below, num_bases)
        
                # reset trainable parameters
                for name, param in conn_params.items():
                    nvm.connections[name].W = param
                
                # input env
                memorize_env(nvm, goal_thing_above)
            
                reset_dict = {"jnt": tr.tensor(am.ik["rest"]).float()}
                # reset_dict.update({"obj": nvm.registers["obj"].encode("t0")}) # stack all
                # reset_dict.update({"r0": nvm.registers["r0"].encode("b0")}) # stack on
                nvm.reset(reset_dict)
            
                sigma = 0.001 # radians
                log_prob = 0.0 # accumulate over episode
            
                dbg = False
                nvm.mount("main")
                # nvm.mount("stack_all")
                # nvm.mount("stack_on")
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
                epoch_rewards.append(spa_reward)
                
                if episode != 0:
                    loss = -spa_reward * log_prob
                    loss.backward()

                print("  %d(%d): r = %f, lp= %f" % (epoch, episode, spa_reward, log_prob))

            # update params based on episodes
            opt.step()
            opt.zero_grad()
            
            delta = (orig_ik - conn_params["ik"]).abs().max()
            avg_reward = np.mean(epoch_rewards)
            std_reward = np.std(epoch_rewards)
            
            results.append((epoch_rewards, delta.item()))
            print("%d: R = %f (+/- %f), dW = %f" % (epoch, avg_reward, std_reward, delta))
            with open("rj.pkl","wb") as f: pk.dump(results, f)
    
    showresults = False
    if showresults:
        with open("rj.pkl","rb") as f: results = pk.load(f)
        epoch_rewards, deltas = zip(*results)
        num_epochs = len(results)
    
        print("LC:")
        for r,rewards in enumerate(epoch_rewards): print(r, rewards)
        
        import matplotlib.pyplot as pt
        pt.plot([np.mean(rewards) for rewards in epoch_rewards], 'k-')
        pt.plot([rewards[0] for rewards in epoch_rewards], 'b-')
        x, y = zip(*[(r,reward) for r in range(num_epochs) for reward in epoch_rewards[r]])    
        pt.plot(x, y, 'k.')

        # pt.plot(np.log(-np.array(rewards)))
        # pt.ylabel("log(-R)")
        
        # trend line
        avg_rewards = np.mean(epoch_rewards, axis=1)
        for r in range(len(avg_rewards)-10):
            avg_rewards[r] += avg_rewards[r+1:r+10].sum()
            avg_rewards[r] /= 10
        pt.plot(avg_rewards, 'ro-')

        pt.show()
    
