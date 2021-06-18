import torch as tr
import sys, time
sys.path.append('../../envs')
from blocks_world import BlocksWorldEnv
from restack import random_problem_instance, compute_spatial_reward
from abstract_machine import make_abstract_machine, memorize_env
from nvm import virtualize

if __name__ == "__main__":

    num_blocks, max_levels, num_bases = 2, 3, 2
    num_episodes = 2

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
    opt = tr.optim.Adam(conn_params.values(), lr=0.001)

    # thing_below, goal_thing_below = random_problem_instance(env, num_blocks, max_levels, num_bases)
    thing_below = {"b0": "t0", "b1": "t1"}
    goal_thing_below = {"b0": "t0", "b1": "b0"}

    goal_thing_above = nvm.env.invert(goal_thing_below)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"

    lc = []
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
        reset_dict.update({"r0": nvm.registers["r0"].encode("b0")}) # stack on
        nvm.reset(reset_dict)
    
        sigma = 0.02 # radians
        log_prob = 0.0 # accumulate over episode
    
        dbg = True
        # nvm.mount("main")
        # nvm.mount("stack_all")
        nvm.mount("stack_on")
        if dbg: nvm.dbg()
        target_changed = True
        while True:
            done = nvm.tick()
            if dbg: nvm.dbg()
            if target_changed:
                mu = nvm.registers["jnt"].content
                dist = tr.distributions.normal.Normal(mu, sigma)
                position = dist.sample()
                log_probs = dist.log_prob(position)
                log_prob += log_probs.sum() # multivariate white noise
                nvm.env.goto_position(position.detach().numpy())
            tar = nvm.registers["tar"]
            target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
            if done: break
        
        spa_reward = compute_spatial_reward(nvm.env, goal_thing_below)
        
        loss = -spa_reward * log_prob
        # loss.backward(retain_graph=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        lc.append(spa_reward)
        print("%d: R = %f" % (episode, spa_reward))

    print("LC:")
    print(lc)
        
    

