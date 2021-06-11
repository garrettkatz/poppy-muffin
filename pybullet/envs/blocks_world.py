import itertools as it
import os
from math import cos, sin
import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

class BlocksWorldEnv(PoppyErgoJrEnv):
    
    def placement_of(self, name):
        if name in self.blocks:
            pos, quat = pb.getBasePositionAndOrientation(self.block_id[name])
        if name in self.towers:
            radius = -.16
            alpha = 1.57
            theta = (3.14 - alpha)/2 + alpha * self.towers.index(name) / (len(self.towers)-1)
            pos = (radius*cos(theta), radius*sin(theta), 0)
            quat = pb.getQuaternionFromEuler((0,0,theta))
        return pos, quat

    def base_and_level_of(self, block):
        thing = self.thing_below[block]
        if thing in self.towers: return thing, 0
        base, level = self.base_and_level_of(thing)
        return base, level + 1
        
    def load_blocks(self, thing_below):
        
        self.thing_below = thing_below
        self.block_above = {thing: block for (block, thing) in thing_below.items()}
        self.num_blocks = len(thing_below)
        self.blocks = ["b%d" % b for b in range(self.num_blocks)]
        self.towers = ["t%d" % t for t in range(self.num_blocks)]

        # cube urdf path
        fpath = os.path.dirname(os.path.abspath(__file__)) + '/../../urdfs/objects'
        pb.setAdditionalSearchPath(fpath)
        
        colors = list(it.product([0,1], repeat=3))[:self.num_blocks]
        self.block_id = {}
        for b, block in enumerate(self.blocks):
            base, level = self.base_and_level_of(block)
            height = .01 + level * .0201
            pos, quat = self.placement_of(base)
            pos = (pos[0], pos[1], height)
            rgba = colors[b] + (1,)
            self.block_id[block] = pb.loadURDF(
                'cube.urdf', basePosition=pos, baseOrientation=quat, useFixedBase=False)
            pb.changeVisualShape(self.block_id[block], linkIndex=-1, rgbaColor=rgba)
        
        self.step(self.get_position()) # let blocks settle

    def is_above(self, thing1, thing2):
        # true if thing1 above thing2
        pos1, _ = self.placement_of(thing1)
        pos2, _ = self.placement_of(thing2)
        if pos1[2] < pos2[2]: return False
        return all([-.01 < pos1[c] - pos2[c] < +.01 for c in [0,1]])

    def update_block_above(self):
        # update dict based on current block positions
        for thing in self.block_above:
            self.block_above[thing], height = 'none', 100
            for block in self.blocks:
                if block == thing: continue
                pos, _ = self.placement_of(block)
                if self.is_above(block, thing) and pos[2] < height:
                    self.block_above[thing], height = block, pos[2]

    def tip_targets_around(self, pos, quat, delta):
        # fingertip targets at +/- delta along local y-axis in given reference frame
        m = pb.getMatrixFromQuaternion(quat)
        t1 = pos[0]-delta*m[1], pos[1]-delta*m[4], pos[2]-delta*m[7]
        t2 = pos[0]+delta*m[1], pos[1]+delta*m[4], pos[2]+delta*m[7]
        return (t1, t2)
    

    def get_camera_image(self):
        rgb, view, proj = super().get_camera_image()
        width, height = rgb.shape[1], rgb.shape[0]
        coords_of = {} # image coordinates of things
        for thing in self.towers + self.blocks:
            pos, _ = self.placement_of(thing)
            # point in homogenous coordinates
            pt = pos + (1,)
            # matrix transforms
            for mtx in [view, proj]:
                pt = tuple(
                    sum([mtx[row+4*col]*pt[col] for col in range(4)])
                    for row in range(4))            
            # apply perspective in image plane
            pt = (pt[0]/pt[3], pt[1]/pt[3])
            # rescale to pixel units
            pt = (pt[0]*width/2 + width/2, -pt[1]*height/2 + height/2)
            coords_of[thing] = pt
        return rgb, view, proj, coords_of

if __name__ == "__main__":

    env = BlocksWorldEnv(PoppyErgoJrEnv)
    env.load_blocks(
        {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})

    env.update_block_above()
    print(env.block_above)

    rgb, view, proj, coords_of = env.get_camera_image()
    
    import matplotlib.pyplot as pt
    import numpy as np
    x, y = zip(*coords_of.values())
    pt.imshow(rgb)
    pt.plot(x, y, 'ro')
    pt.show()
    
