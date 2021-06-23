import pybullet as pb
import time, sys
import numpy as np
import matplotlib.pyplot as pt
sys.path.append('../../envs')    
from blocks_world import BlocksWorldEnv, random_thing_below
    
def compute_symbolic_reward(env, goal_thing_below):
    env.update_relations()
    reward = 0
    for block in env.blocks:
        actual_below = env.thing_below[block]
        target_below = goal_thing_below[block]
        if actual_below == target_below: continue
        if actual_below in env.bases and target_below in env.bases: continue
        reward -= 1
    return reward

def compute_spatial_reward(env, goal_thing_below):
    def target_placement_of(block):
        if goal_thing_below[block] in env.bases:
            # don't care which table position grounds the tower
            pos, quat = env.placement_of(block)
        else:
            pos, quat = target_placement_of(goal_thing_below[block])
            pos = (pos[0], pos[1], pos[2] + .02)
        return pos, quat

    reward = 0
    for block in env.blocks:
        actual_pos, actual_quat = env.placement_of(block)
        target_pos, target_quat = target_placement_of(block)
        c = min(1, abs(sum([q1*q2 for (q1, q2) in zip(target_quat, actual_quat)])))
        reward -= 2*np.arccos(c)
        reward -= sum([(t-a)**2 for (t,a) in zip(target_pos, actual_pos)])**0.5
    
    return reward

def invert(thing_below, num_blocks, num_bases):
    blocks = ["b%d" % n for n in range(num_blocks)]
    bases = ["t%d" % n for n in range(num_bases)]
    block_above = {thing: "none" for thing in blocks + bases}
    for block, thing in thing_below.items(): block_above[thing] = block
    return block_above

def random_problem_instance(env, num_blocks, max_levels, num_bases):
    # sets env with initial blocks in the process

    # rejection sample non-trivial instance
    while True:
        thing_below = random_thing_below(num_blocks, max_levels, num_bases)
        goal_thing_below = random_thing_below(num_blocks, max_levels, num_bases)
        env.load_blocks(thing_below, num_bases)
        if compute_symbolic_reward(env, goal_thing_below) < 0: break
        env.reset()
    
    return thing_below, goal_thing_below

class DataDump:
    def __init__(self, goal_thing_below, hook_period):
        self.goal_thing_below = goal_thing_below
        self.hook_period = hook_period
        self.data = []
    def add_command(self, command):
        self.data.append({
            "command": command,
            "records": []})
        self.period_counter = 0
        print("  added command " + str(command))
    def step_hook(self, env, action):
        if action is None or len(self.data) == 0: return

        if self.period_counter % self.hook_period == 0:
            position = env.get_position()
            rgba, _, _, coords_of = env.get_camera_image()
            spatial_reward = compute_spatial_reward(env, self.goal_thing_below)
            self.data[-1]["records"].append(
                (position, action, rgba, coords_of, spatial_reward))
            # if not pt.isinteractive(): pt.ion()
            # pt.cla()
            # pt.imshow(rgba)
            # r, c = zip(*coords_of.values())
            # pt.plot(c, r, 'ro')
            # pt.show()
            # pt.pause(0.01)

        self.period_counter += 1

class Restacker:
    def __init__(self, env, goal_thing_below, dump=None):
        self.env = env
        self.goal_thing_below = goal_thing_below
        self.goal_block_above = env.invert(goal_thing_below)
        self.dump = dump

    def free_spot(self):
        for base in self.env.bases:
            if self.env.block_above[base] == 'none': return base
        return False

    def move_to(self, thing, block):
        if self.dump is not None: self.dump.add_command(("move_to", (thing, block)))
        self.env.pick_up(block)
        self.env.put_down_on(thing, block)
        self.env.block_above[self.env.thing_below[block]] = 'none'
        self.env.block_above[thing] = block
        self.env.thing_below[block] = thing

    def unstack_from(self, thing):
        block = self.env.block_above[thing]
        if block == "none": return
        self.unstack_from(block)
        self.move_to(self.free_spot(), block)

    def unstack_all(self):
        for base in self.env.bases:
            block = self.env.block_above[base]
            if block != "none": self.unstack_from(block)

    def stack_on(self, thing):
        block = self.goal_block_above[thing]
        if block == "none": return
        self.move_to(thing, block)
        self.stack_on(block)

    def stack_all(self):
        for base in self.env.bases:
            block = self.goal_block_above[base]
            if block != 'none': self.stack_on(block)

    def run(self):
        self.unstack_all()
        self.stack_all()

if __name__ == "__main__":

    # failure case:
    num_blocks = 5
    thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
    goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}

    # num_blocks = 4
    # thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    # thing_below.update({"b1": "b0", "b2": "b3"})

    # goal_thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    # goal_thing_below.update({"b1": "b2", "b2": "b0"})

    # # thing_below = random_thing_below(num_blocks, max_levels=3)
    # # goal_thing_below = random_thing_below(num_blocks, max_levels=3)

    dump = DataDump(goal_thing_below, hook_period=1)
    # env = BlocksWorldEnv(pb.POSITION_CONTROL, show=True, control_period=12, step_hook=dump.step_hook)
    env = BlocksWorldEnv()
    env.load_blocks(thing_below)

    # from check/camera.py
    pb.resetDebugVisualizerCamera(
        1.2000000476837158, 56.799964904785156, -22.20000648498535,
        (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))
    
    restacker = Restacker(env, goal_thing_below, dump)
    restacker.run()
    
    reward = compute_symbolic_reward(env, goal_thing_below)
    print("symbolic reward = %f" % reward)

    reward = compute_spatial_reward(env, goal_thing_below)
    print("spatial reward = %f" % reward)

    # rewards = []
    # for frame in dump.data:
    #     for record in frame["records"]:
    #         rewards.append(record[-1])
    # print("spatial reward = %f" % rewards[-1])    
    # pt.plot(rewards)
    # pt.show()

    env.close()

