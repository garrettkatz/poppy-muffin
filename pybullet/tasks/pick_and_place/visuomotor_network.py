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
        num_joints = 8
        super().__init__()
        self.conv1 = tr.nn.Conv2d(3, 64, 5)
        self.conv2 = tr.nn.Conv2d(64, 32, 5)
        self.lin1 = tr.nn.Linear(32*2 + num_joints+2, 40) # spatial features + joints and focus point
        self.lin2 = tr.nn.Linear(40, num_joints+2) # joint actions + new focus point coordinates
    def forward(self, inputs):
        position, rgb, coords_of = inputs
        h = tr.relu(self.conv1(rgb))
        h = tr.relu(self.conv2(h))
        h = spatial_softmax(h)
        i, j = average_coordinates(h)
        h = tr.cat((position, coords_of, i, j), dim=1)
        h = tr.relu(self.lin1(h))
        h = self.lin2(h)
        actions, coords = h[:, :-2], h[:, -2:]
        return actions, coords        

if __name__ == "__main__":
    
    import pybullet as pb
    import sys
    import pickle as pk

    # generate pickup data
    regen = False
    if regen:
        sys.path.append('../../envs')    
        from blocks_world import BlocksWorldEnv
    
        num_blocks = 7
        thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
        
        dump = DataDump()
        
        env = BlocksWorldEnv(pb.POSITION_CONTROL, control_period=30, show=True, step_hook = dump.step_hook)
        env.load_blocks(thing_below)
    
        goal_block_above = env.invert(thing_below)
        restacker = Restacker(env, goal_block_above, dump)
        restacker.pick_up("b1")
        env.close()
    
        _, (block,) = dump.data[0]["command"]
        position, action, rgba, coords_of = zip(*dump.data[0]["records"])
        
        position = tr.tensor(np.stack(position)).float()
        action = tr.tensor(np.stack(action)).float()
        rgba = tr.tensor(np.stack(rgba))
        coords_of = tr.tensor(np.stack([co[block] for co in coords_of])).float()
        
        # preprocessing
        rgb = rgba[:,:,:,:3] # drop alpha channel
        rgb = rgb.permute(0,3,1,2) # put channels first
        height = rgb.shape[2] # get image height
        lo, hi = height//4, 3*height//4 # get height clip range
        rgb = rgb[:,:,lo:hi,:] # clip rgb data
        coords_of[:,0] -= lo # clip row coordinates
        rgb = (rgb / 255).float() # convert to float
    
        inputs = (position[:-1], rgb[:-1], coords_of[:-1])
        targets = (action[1:], coords_of[1:])
        
        with open("tdat.pkl","wb") as f: pk.dump((inputs, targets), f)
    
    with open("tdat.pkl","rb") as f: inputs, targets = pk.load(f)
    
    position, rgb, coords_of = inputs
    print(coords_of) # [... (r,c) ...]
    print(rgb.shape)

    import matplotlib.pyplot as pt
    pt.imshow(rgb.permute(0,2,3,1).data[0])
    r, c = coords_of[0,0], coords_of[0,1]
    pt.plot(c, r, 'ro')
    pt.show()
    
    net = VisuoMotorNetwork()
    outputs = net(inputs)
    action, new_coords = outputs
    print(action)
    print(new_coords)
    
    
