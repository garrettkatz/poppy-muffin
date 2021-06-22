import itertools as it
import os
import numpy as np
from math import sin, cos
import random
import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

class BlocksWorldEnv(PoppyErgoJrEnv):
    
    def placement_of(self, name):
        if name in self.blocks:
            pos, quat = pb.getBasePositionAndOrientation(self.block_id[name])
        if name in self.bases:
            radius = -.16
            alpha = 1.57
            theta = (3.14 - alpha)/2 + alpha * self.bases.index(name) / (len(self.bases)-1)
            pos = (radius*cos(theta), radius*sin(theta), 0)
            quat = pb.getQuaternionFromEuler((0,0,theta))
        return pos, quat

    def base_and_level_of(self, block):
        # blocks starting from level 0, does not accept bases
        thing = self.thing_below[block]
        if thing in self.bases: return thing, 0
        base, level = self.base_and_level_of(thing)
        return base, level + 1
    
    def invert(self, thing_below):
        block_above = {thing: "none" for thing in self.blocks + self.bases}
        for block, thing in thing_below.items(): block_above[thing] = block
        return block_above
    
    def load_blocks(self, thing_below, num_bases=None):
        
        self.num_blocks = len(thing_below)
        self.num_bases = self.num_blocks if num_bases is None else num_bases
        self.blocks = ["b%d" % b for b in range(self.num_blocks)]
        self.bases = ["t%d" % t for t in range(self.num_bases)]
        self.thing_below = dict(thing_below) # copy
        self.block_above = self.invert(thing_below)

        # cube urdf path
        fpath = os.path.dirname(os.path.abspath(__file__)) + '/../../urdfs/objects'
        pb.setAdditionalSearchPath(fpath)
        
        colors = list(it.product([0,1], repeat=3))
        self.block_id = {}
        for b, block in enumerate(self.blocks):
            base, level = self.base_and_level_of(block)
            height = .01 + level * .0201
            pos, quat = self.placement_of(base)
            pos = (pos[0], pos[1], height)
            rgba = colors[b % len(colors)] + (1,)
            self.block_id[block] = pb.loadURDF(
                'cube.urdf', basePosition=pos, baseOrientation=quat, useFixedBase=False)
            pb.changeVisualShape(self.block_id[block], linkIndex=-1, rgbaColor=rgba)
        
        self.step() # let blocks settle

    def reset(self):
        for block in self.blocks: pb.removeBody(self.block_id[block])
        super().reset()
    
    def is_above(self, thing1, thing2):
        # true if thing1 above thing2
        pos1, _ = self.placement_of(thing1)
        pos2, _ = self.placement_of(thing2)
        if pos1[2] < pos2[2]: return False
        return all([-.01 < pos1[c] - pos2[c] < +.01 for c in [0,1]])

    def update_relations(self):
        # update dict based on current block positions
        self.block_above = {}
        for thing in self.blocks + self.bases:
            self.block_above[thing], height = 'none', 100
            for block in self.blocks:
                if block == thing: continue
                pos, _ = self.placement_of(block)
                if self.is_above(block, thing) and pos[2] < height:
                    self.block_above[thing], height = block, pos[2]
        # invert block_above for thing_below
        self.thing_below = {block: 'none' for block in self.blocks}
        for thing, block in self.block_above.items():
            if block != 'none': self.thing_below[block] = thing

    def tip_targets_around(self, pos, quat, delta):
        # fingertip targets at +/- delta along local y-axis in given reference frame
        m = pb.getMatrixFromQuaternion(quat)
        t1 = pos[0]-delta*m[1], pos[1]-delta*m[4], pos[2]-delta*m[7]
        t2 = pos[0]+delta*m[1], pos[1]+delta*m[4], pos[2]+delta*m[7]
        return (t1, t2)

    def run_trajectory(self, quat, waypoints):
        for point, delta in waypoints: 
            targs = self.tip_targets_around(point, quat, delta)
            target_position = self.inverse_kinematics([5, 7], targs)
            # actual_position = self.get_position()
            # distance = np.fabs(target_position - actual_position).max()
            # velocity = 1.5 # radians per second
            # duration = distance / velocity
            # # duration = .25
            # self.goto_position(target_position, duration)
            self.goto_position(target_position, 1.0)
    
    def pick_up(self, block):
        pos, quat = self.placement_of(block)
        stage = pos[:2] + (.1,)
        self.run_trajectory(quat, [(stage, .02), (pos, .02), (pos, .01), (stage, .01)])

    def put_down_on(self, thing, block):
        pos, quat = self.placement_of(thing)
        pos = pos[:2] + (pos[2] + .0201,)
        stage = pos[:2] + (.1,)    
        self.run_trajectory(quat, [(stage, .01), (pos, .01), (pos, .02), (stage, .02)])
    
    def get_camera_image(self):
        rgba, view, proj = super().get_camera_image()        
        width, height = rgba.shape[1], rgba.shape[0]

        coords_of = {} # image coordinates of things
        for thing in self.bases + self.blocks:
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
            row, col = -pt[1]*height/2 + height/2, pt[0]*width/2 + width/2
            coords_of[thing] = (row, col)

        return rgba, view, proj, coords_of
    
    def movement_penalty(self):
        penalty = 0
        s_5, s_7 = pb.getLinkStates(self.robot_id, [5, 7])
        xyz_5, xyz_7 = s_5[0], s_7[0]
        p_grip = np.array([xyz_5, xyz_7]).mean(axis=0)
        for bid in self.block_id.values():
            p_block, _ = pb.getBasePositionAndOrientation(bid)
            v_block, _ = pb.getBaseVelocity(bid)
            speed = np.sum(np.array(v_block)**2)**.5
            delta = np.sum((np.array(p_block) - p_grip)**2)**.5
            penalty = max(speed * delta, penalty)
        return penalty

def random_thing_below(num_blocks, max_levels, num_bases=None):
    # make a random thing_below dictionary
    # no towers have more than max_levels
    if num_bases is None: num_bases = num_blocks
    # towers = [["t%d" % n] for n in range(num_bases)] # singleton towers too likely
    active_bases = random.sample(range(num_bases), num_blocks)
    towers = [["t%d" % n] for n in active_bases]
    for n in range(num_blocks):
        short_towers = list(filter(lambda x: len(x[1:]) < max_levels, towers)) # exclude bases
        tower = random.choice(short_towers)
        tower.append("b%d" % n)
    thing_below = {}
    for tower in towers:
        for level in range(len(tower)-1):
            thing_below[tower[level+1]] = tower[level]
    return thing_below


if __name__ == "__main__":

    num_blocks = 7
    num_bases = 7
    max_levels = 3
    thing_below = random_thing_below(num_blocks, max_levels, num_bases)
    
    # # full occupancy
    # thing_below = {}
    # b = 0
    # for base in range(num_bases):
    #     thing_below["b%d" % b] = "t%d" % base
    #     b += 1
    #     for level in range(max_levels-1):
    #         thing_below["b%d" % b] = "b%d" % (b-1)
    #         b += 1

    env = BlocksWorldEnv(pb.POSITION_CONTROL)
    env.load_blocks(
        thing_below,
        # {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"},
        num_bases)

    env.update_relations()
    print(env.block_above)

    rgb, view, proj, coords_of = env.get_camera_image()
    
    import matplotlib.pyplot as pt
    import numpy as np
    r, c = zip(*coords_of.values())
    pt.imshow(rgb)
    pt.plot(c, r, 'ro')
    pt.show()
    
