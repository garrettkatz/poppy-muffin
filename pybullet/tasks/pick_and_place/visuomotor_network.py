import pickle as pk
import numpy as np
import torch as tr
import pybullet as pb
from restack import DataDump, Restacker, compute_symbolic_reward
import sys, os
sys.path.append('../../envs')    
from blocks_world import BlocksWorldEnv, random_thing_below

def spatial_softmax(inp):
    # inp should be size (N,C,H,W)
    # takes softmax over last two dimensions
    N, C, H, W = inp.shape
    r = inp.reshape(N, C, H*W)
    s = tr.softmax(r, dim=2)
    return s.reshape(N, C, H, W)

def average_coordinates(inp):
    # inp should be size (N,C,H,W)
    # takes average of image coordinates weighted by inp
    N, C, H, W = inp.shape
    h, w = tr.meshgrid(tr.arange(H), tr.arange(W))
    i = (inp * h.float()).reshape(N, C, H*W).sum(dim=2)
    j = (inp * w.float()).reshape(N, C, H*W).sum(dim=2)
    return i, j

class VisuoMotorNetwork(tr.nn.Module):
    def __init__(self):
        num_feat = 8 + 2 + 2 # joints, block coords, thing coords
        super().__init__()
        self.conv1 = tr.nn.Conv2d(3, 64, 5)
        self.conv2 = tr.nn.Conv2d(64, 32, 5)
        self.conv3 = tr.nn.Conv2d(32, 16, 5)
        self.lin1 = tr.nn.Linear(16*2 + num_feat, 32) # spatial features + linputs
        self.lin2 = tr.nn.Linear(32, 32)
        self.lin3 = tr.nn.Linear(32, num_feat)
    def forward(self, inputs):
        position, rgb, block_coords, thing_coords = inputs
        h = tr.relu(self.conv1(rgb))
        h = tr.relu(self.conv2(h))
        h = tr.relu(self.conv3(h))
        h = spatial_softmax(h)
        i, j = average_coordinates(h)
        h = tr.cat((position, block_coords, thing_coords, i, j), dim=1)
        h = tr.relu(self.lin1(h))
        h = tr.relu(self.lin2(h))
        h = self.lin3(h)
        # actions, coords = h[:, :-4], h[:, -4:]
        # residual learning
        action = h[:, :-4] + position
        new_block_coords = h[:, -4:-2] + block_coords
        new_thing_coords = h[:, -2:] + thing_coords
        return action, new_block_coords, new_thing_coords

def generate_data(num_blocks, base_name):
    
    # thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    # thing_below["b1"] = "b0"
    # goal_thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    # goal_thing_below["b2"] = "b1"

    thing_below = random_thing_below(num_blocks, max_levels=3)
    goal_thing_below = random_thing_below(num_blocks, max_levels=3)

    dump = DataDump(goal_thing_below, hook_period=1)
    env = BlocksWorldEnv(pb.POSITION_CONTROL, show=False, control_period=12, step_hook=dump.step_hook)
    env.load_blocks(thing_below)

    restacker = Restacker(env, goal_thing_below, dump)
    restacker.run()

    reward = compute_symbolic_reward(env, goal_thing_below)
    final_thing_below = env.thing_below
    commands = [frame["command"] for frame in dump.data]
    data_file = "%s/meta.pkl" % base_name
    data = (thing_below, goal_thing_below, final_thing_below, reward, commands)
    with open(data_file, "wb") as f: pk.dump(data, f)

    env.close()

    for d, frame in enumerate(dump.data):
        _, (thing, block) = frame["command"]
        position, action, rgba, coords_of, _ = zip(*frame["records"])
        
        position = tr.tensor(np.stack(position)).float()
        action = tr.tensor(np.stack(action)).float()
        rgba = tr.tensor(np.stack(rgba))
        block_coords = tr.tensor(np.stack([co[block] for co in coords_of])).float()
        thing_coords = tr.tensor(np.stack([co[thing] for co in coords_of])).float()
        
        # preprocessing
        rgb = rgba[:,:,:,:3] # drop alpha channel
        rgb = (rgb / 255).float() # convert to float
        rgb = rgb.permute(0,3,1,2) # put channels first
        rgb = rgb[:,:,32:72, 20:108] # clip rgb data
        block_coords -= tr.tensor([[32., 20.]]) # clip coordinates
        thing_coords -= tr.tensor([[32., 20.]]) # clip coordinates

        data_file = "%s/%03d.pt" % (base_name, d)
        tr.save((position, action, rgb, block_coords, thing_coords), data_file)
    
    print(" success=%s (start, end, goal)" % (reward == 0))
    print("  ", thing_below)
    print("  ", env.thing_below)
    print("  ", goal_thing_below)
    return reward

if __name__ == "__main__":
    
    import matplotlib.pyplot as pt

    # generate pickup data
    regen = True
    if regen:
        
        num_episodes = 500
        min_blocks = 3
        max_blocks = 7
        num_success = 0
    
        os.system("rm -fr episodes/*")
        print()

        for episode in range(num_episodes):
            num_blocks = np.random.randint(min_blocks, max_blocks+1)
            folder = "episodes/%03d" % episode
            os.system("mkdir " + folder)
            print("%s, %d blocks" % (folder, num_blocks))
            reward = generate_data(num_blocks, base_name=folder)
            num_success += (reward == 0)
    
        print("%d of %d successful" % (num_success, num_episodes))

    episode = 0
    folder = "episodes/%03d" % episode
    with open(folder + "/meta.pkl", "rb") as f:
        thing_below, goal_thing_below, final_thing_below, reward, commands = pk.load(f)
    position, action, rgb, block_coords, thing_coords = tr.load(folder + "/000.pt")

    print(folder)
    print(" commands:")
    for command in commands: print("  ", command)
    print(" success=%s (start, end, goal)" % (reward == 0))
    print("  ", thing_below)
    print("  ", final_thing_below)
    print("  ", goal_thing_below)
    
    inputs = (position[:-1], rgb[:-1], block_coords[:-1], thing_coords[:-1])
    targets = (action[:-1], block_coords[1:], thing_coords[1:])

    # pt.ion()
    # for t in range(len(rgb)):
    #     pt.cla()
    #     pt.imshow(rgb.permute(0,2,3,1).data[t])
    #     rb, cb = block_coords[t,0], block_coords[t,1]
    #     rt, ct = thing_coords[t,0], thing_coords[t,1]
    #     pt.plot(cb, rb, 'ro')
    #     pt.plot(ct, rt, 'ro')
    #     pt.show()
    #     pt.pause(0.5)    
    # input('.')
    
    net = VisuoMotorNetwork()
    targ_action, targ_block_coords, targ_thing_coords = targets

    train = False
    if train:
        optim = tr.optim.Adam(net.parameters(), lr=0.001)
    
        for epoch in range(10):
        
            outputs = net(inputs)
            pred_action, pred_block_coords, pred_thing_coords = outputs
            loss = tr.sum((pred_action - targ_action)**2)
            loss += tr.sum((pred_block_coords - targ_block_coords)**2)
            loss += tr.sum((pred_thing_coords - targ_thing_coords)**2)
            print("%d: %f" % (epoch, loss.item()))
            loss.backward()
            optim.step()
            optim.zero_grad()
    
        outputs = net(inputs)
        tr.save(outputs, "preds.pt")

    if False:
        outputs = tr.load("preds.pt")
        pred_action, pred_block_coords, pred_thing_coords = outputs
        
        pt.subplot(1,2,1)
        pt.plot(pred_action.detach().numpy(), color='r')
        pt.plot(targ_action.numpy(), color='b')
        
        pt.subplot(1,2,2)
        pt.imshow(rgb.permute(0,2,3,1).data[-1])
        for (pred_coords, targ_coords) in zip(
            [pred_block_coords, pred_thing_coords],
            [targ_block_coords, targ_thing_coords]
        ):
            for t in range(len(pred_action)):
                rp, cp = pred_coords[t]
                rt, ct = targ_coords[t]
                pt.plot([cp,ct], [rp, rt], 'ro-')
                pt.plot(ct, rt, 'bo')
        pt.show()
    
