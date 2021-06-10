import itertools as it
import os
from math import cos, sin
import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

class BlocksWorldEnv(PoppyErgoJrEnv):

    def position_of(self, name):
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
            pos, quat = self.position_of(base)
            pos = (pos[0], pos[1], height)
            rgba = colors[b] + (1,)
            self.block_id[block] = pb.loadURDF(
                'cube.urdf', basePosition=pos, baseOrientation=quat, useFixedBase=False)
            pb.changeVisualShape(self.block_id[block], linkIndex=-1, rgbaColor=rgba)
        
        env.step() # let blocks settle

if __name__ == "__main__":
            
    env = BlocksWorldEnv(PoppyErgoJrEnv)
    env.load_blocks({"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})
    rgb = env.get_camera_image()

    width, height = 128, 96
    view = pb.computeViewMatrix(
        cameraEyePosition=(0,-.02,.02),
        cameraTargetPosition=(0,-.4,.02), # focal point
        cameraUpVector=(0,0,1),
    )
    proj = pb.computeProjectionMatrixFOV(
        fov=135,
        aspect=height/width,
        nearVal=0.01,
        farVal=.4,
    )
    print(proj)

    pts = []    
    for block in env.blocks:
        pos, _ = env.position_of(block)
        # print(pos)        
        p = (
            view[0]*pos[0] + view[4]*pos[1] + view[8]*pos[2] + view[12]*1,
            view[1]*pos[0] + view[5]*pos[1] + view[9]*pos[2] + view[13]*1,
            view[2]*pos[0] + view[6]*pos[1] + view[10]*pos[2] + view[14]*1,
            view[3]*pos[0] + view[7]*pos[1] + view[11]*pos[2] + view[15]*1,
        )
        # print(p)
        p = (
            proj[0]*p[0] + proj[4]*p[1] + proj[8]*p[2] + proj[12]*p[3],
            proj[1]*p[0] + proj[5]*p[1] + proj[9]*p[2] + proj[13]*p[3],
            proj[2]*p[0] + proj[6]*p[1] + proj[10]*p[2] + proj[14]*p[3],
            proj[3]*p[0] + proj[7]*p[1] + proj[11]*p[2] + proj[15]*p[3],
        )
        print(block)
        
        p = (p[0]/p[3], p[1]/p[3])
        print(p)
        
        p = (p[0]*width/2 + width/2, -p[1]*height/2 + height/2)
        print(p)

        pts.append(p)

    # input('.')

    import matplotlib.pyplot as pt
    import numpy as np
    x, y = zip(*pts)
    pt.imshow(rgb)
    pt.plot(x, y, 'ro')
    pt.show()
    
