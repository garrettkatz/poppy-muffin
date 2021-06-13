import numpy as np
import torch as tr
from restack import DataDump, Restacker

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
    i = (inp * h).reshape(N, C, H*W).sum(dim=2)
    j = (inp * w).reshape(N, C, H*W).sum(dim=2)
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

if __name__ == "__main__":
    
    import pybullet as pb
    import sys
    import pickle as pk
    import matplotlib.pyplot as pt

    # generate pickup data
    regen = False
    if regen:
        sys.path.append('../../envs')    
        from blocks_world import BlocksWorldEnv
    
        num_blocks = 7
        thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
        thing_below["b1"], thing_below["b2"] = "b0", "b1"
        thing_below["b5"], thing_below["b4"] = "b6", "b5"
        
        dump = DataDump()
        
        env = BlocksWorldEnv(pb.POSITION_CONTROL, control_period=30, show=True, step_hook = dump.step_hook)
        env.load_blocks(thing_below)
    
        goal_block_above = env.invert(thing_below)
        restacker = Restacker(env, goal_block_above, dump)
        restacker.move_to("b3", "b4")
        env.close()
    
        _, (thing, block) = dump.data[0]["command"]
        position, action, rgba, coords_of = zip(*dump.data[0]["records"])
        
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
        block_coords -= tr.tensor([[32, 20]]) # clip coordinates
        thing_coords -= tr.tensor([[32, 20]]) # clip coordinates
    
        with open("tdat.pkl","wb") as f:
            pk.dump((position, action, rgb, block_coords, thing_coords), f)
    
    with open("tdat.pkl","rb") as f:
        position, action, rgb, block_coords, thing_coords = pk.load(f)
    
    inputs = (position[:-1], rgb[:-1], block_coords[:-1], thing_coords[:-1])
    targets = (action[:-1], block_coords[1:], thing_coords[1:])
    print(block_coords) # [... (r,c) ...]
    print(rgb.shape)

    # pt.imshow(rgb.permute(0,2,3,1).data[0])
    # rb, cb = block_coords[0,0], block_coords[0,1]
    # rt, ct = thing_coords[0,0], thing_coords[0,1]
    # pt.plot(cb, rb, 'ro')
    # pt.plot(ct, rt, 'ro')
    # pt.show()
    
    net = VisuoMotorNetwork()
    targ_action, targ_block_coords, targ_thing_coords = targets

    train = True
    if train:
        optim = tr.optim.Adam(net.parameters(), lr=0.001)
    
        for epoch in range(500):
        
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
        with open("preds.pkl","wb") as f: pk.dump(outputs, f)

    with open("preds.pkl","rb") as f: outputs = pk.load(f)
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
    
