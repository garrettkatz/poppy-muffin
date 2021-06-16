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
        self.conv3 = tr.nn.Conv2d(32, 32, 5)
        self.lin1 = tr.nn.Linear(32*2 + num_feat, 40) # spatial features + linputs
        self.lin2 = tr.nn.Linear(40, 40)
        self.lin3 = tr.nn.Linear(40, num_feat)
        # self.action1 = tr.nn.Linear(16*2 + num_feat, 32)
        # self.action2 = tr.nn.Linear(32, 32)
        # self.action3 = tr.nn.Linear(32, 8)
        # self.coords1 = tr.nn.Linear(16*2 + num_feat, 32)
        # self.coords2 = tr.nn.Linear(32, 32)
        # self.coords3 = tr.nn.Linear(32, 2+2)
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
        # a = tr.relu(self.action1(h))
        # a = tr.relu(self.action2(a))
        # a = self.action3(a)
        # c = tr.relu(self.coords1(h))
        # c = tr.relu(self.coords2(c))
        # c = self.coords3(c)

        # residual learning
        # action = a + position
        # new_block_coords = c[:, :2] + block_coords
        # new_thing_coords = c[:, 2:] + thing_coords

        # action = h[:, :-4] + position
        # new_block_coords = h[:, -4:-2] + block_coords
        # new_thing_coords = h[:, -2:] + thing_coords

        # # confine to reasonable limits
        # action = tr.tanh(action / 1.57)*1.57
        # bounds = tr.tensor([[20., 44.]])
        # new_block_coords = (tr.tanh(new_block_coords / bounds - 1) + 1)*bounds
        # new_thing_coords = (tr.tanh(new_thing_coords / bounds - 1) + 1)*bounds

        # confine deltas to reasonable limits
        action = tr.tanh(h[:, :-4] / .2) * .2
        new_block_coords = tr.tanh(h[:, -4:-2] / 10.) * 10.
        new_thing_coords = tr.tanh(h[:, -2:] / 10.) * 10.
        # then apply residual learning
        action += position
        new_block_coords += block_coords
        new_thing_coords += thing_coords

        return action, new_block_coords, new_thing_coords

def preprocess(rgba, block_coords, thing_coords):
    rgb = rgba[:,:,:,:3] # drop alpha channel
    rgb = rgb.float() / 255. # convert to float
    rgb = rgb.permute(0,3,1,2) # put channels first
    rgb = rgb[:,:,32:72, 20:108] # clip rgb data
    block_coords -= tr.tensor([[32., 20.]]) # clip coordinates
    thing_coords -= tr.tensor([[32., 20.]]) # clip coordinates
    return rgb, block_coords, thing_coords

def generate_data(num_blocks, base_name):
    
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
        rgb, block_coords, thing_coords = preprocess(rgba, block_coords, thing_coords)

        data_file = "%s/%03d.pt" % (base_name, d)
        tr.save((position, action, rgb, block_coords, thing_coords), data_file)
    
    print(" success=%s (start, end, goal)" % (reward == 0))
    print("  ", thing_below)
    print("  ", env.thing_below)
    print("  ", goal_thing_below)
    return reward

class DataLoader:
    def __init__(self, folder, episodes, shuffle=True, noise=True):
        self.folder = folder
        self.episodes = episodes
        self.shuffle = shuffle
        self.noise = noise
    def __iter__(self):

        episodes = self.episodes
        if self.shuffle: episodes = np.random.permutation(episodes)
        
        for episode in episodes:
            subfolder = "%s/%03d" % (self.folder, episode)

            with open(subfolder + "/meta.pkl", "rb") as f:
                _, _, _, reward, commands = pk.load(f)
            if reward < 0: continue
            
            for c in range(len(commands)):
                record = tr.load("%s/%03d.pt" % (subfolder, c))
                position, action, rgb, block_coords, thing_coords = record
                inputs = (
                    position[:-1] + tr.randn(position[:-1].shape)*.1*self.noise,
                    rgb[:-1] + tr.randn(rgb[:-1].shape)*.00*self.noise,
                    block_coords[:-1] + tr.randn(block_coords[:-1].shape)*5*self.noise,
                    thing_coords[:-1] + tr.randn(block_coords[:-1].shape)*5*self.noise,
                )
                targets = (action[:-1], block_coords[1:], thing_coords[1:])
                yield inputs, targets
        

if __name__ == "__main__":
    
    # scaling
    num_episodes = 500
    min_blocks = 3
    max_blocks = 7
    num_epochs = 500
    
    import matplotlib.pyplot as pt

    # generate pickup data
    base_name = "episodes"
    regen = False
    if regen:
        
        num_success = 0
    
        os.system("rm -fr %s/*" % base_name)
        print()

        for episode in range(num_episodes):
            num_blocks = np.random.randint(min_blocks, max_blocks+1)
            folder = "%s/%03d" % (base_name, episode)
            os.system("mkdir " + folder)
            print("%s, %d blocks" % (folder, num_blocks))
            reward = generate_data(num_blocks, base_name=folder)
            num_success += (reward == 0)
    
        print("%d of %d successful" % (num_success, num_episodes))

    # dataloader = DataLoader(base_name, list(range(num_episodes)), shuffle=False)
    # inputs, targets = next(iter(dataloader))
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
    action_scale = 1.0 / 1.57 # +/- this range for each joint
    coords_scale = 1.0 / 64.0 # +/- this range for each pixel coordinate

    train = False
    if train:
        dataloader = DataLoader(base_name, list(range(num_episodes)), shuffle=True, noise=True)
        optim = tr.optim.Adam(net.parameters(), lr=0.001)
    
        best_loss = None
        loss_curve = []
        for epoch in range(num_epochs):

            total_loss = 0.0

            for batch, (inputs, targets) in enumerate(dataloader):

                outputs = net(inputs)
                targ_action, targ_block_coords, targ_thing_coords = targets
                pred_action, pred_block_coords, pred_thing_coords = outputs

                loss = tr.sum((pred_action - targ_action)**2) * action_scale**2
                loss += tr.sum((pred_block_coords - targ_block_coords)**2) * coords_scale**2
                loss += tr.sum((pred_thing_coords - targ_thing_coords)**2) * coords_scale**2
                total_loss += loss.item()
                print("%d, %d: %f" % (epoch, batch, loss / len(pred_action)))
                loss.backward()
                optim.step()
                optim.zero_grad()

            loss_curve.append(total_loss)
            if best_loss is None or total_loss < best_loss:
                best_loss = total_loss
                tr.save(net.state_dict(), "net.pt")

            np.save("lc.npy", np.array(loss_curve))

    show_results = True
    if show_results:

        # loss_curve = np.load("lc.npy")
        # net.load_state_dict(tr.load("net.pt"))
        loss_curve = np.load("lc500.npy")
        net.load_state_dict(tr.load("net500.pt"))

        dataloader = DataLoader(base_name, list(range(num_episodes)), shuffle=False, noise=False)
        inputs, targets = next(iter(dataloader))
        outputs = net(inputs)
        position, rgb, block_coords, thing_coords = inputs
        targ_action, targ_block_coords, targ_thing_coords = targets
        pred_action, pred_block_coords, pred_thing_coords = outputs

        pt.subplot(1,3,1)
        pt.plot(pred_action.detach().numpy(), color='r')
        pt.plot(targ_action.numpy(), color='b')
        
        pt.subplot(1,3,2)
        pt.imshow(rgb.permute(0,2,3,1).data[-1])
        for (pred_coords, targ_coords) in zip(
            [pred_block_coords, pred_thing_coords],
            [targ_block_coords, targ_thing_coords]
        ):
            for t in range(len(pred_action)):
                rp, cp = pred_coords[t]
                rt, ct = targ_coords[t]
                pt.plot([cp,ct], [rp, rt], 'ro-')
            for t in range(len(pred_action)):
                rt, ct = targ_coords[t]
                pt.plot(ct, rt, 'bo')

        pt.subplot(1,3,3)
        pt.plot(loss_curve)

        pt.show()
        
