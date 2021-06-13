import pybullet as pb
import time, sys
import matplotlib.pyplot as pt

class DataDump:
    def __init__(self):
        self.data = []
    def add_command(self, command):
        self.data.append({
            "command": command,
            "records": []})
    def step_hook(self, env, action):
        if action is None or len(self.data) == 0: return
        position = env.get_position()
        rgba, _, _, coords_of = env.get_camera_image()
        self.data[-1]["records"].append((position, action, rgba, coords_of))

        # if not pt.isinteractive(): pt.ion()
        # pt.cla()
        # pt.imshow(rgba)
        # r, c = zip(*coords_of.values())
        # pt.plot(c, r, 'ro')
        # pt.show()
        # pt.pause(0.01)

class Restacker:
    def __init__(self, env, goal_block_above, dump=None):
        self.env = env
        self.goal_block_above = goal_block_above
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

    num_blocks = 7
    thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    # thing_below["b3"] = "b4"
    # thing_below["b4"] = "b5"
    # thing_below["b2"] = "b1"

    goal_thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    goal_thing_below["b3"] = "b2"
    # goal_thing_below["b2"] = "b6"

    dump = DataDump()

    sys.path.append('../../envs')    
    from blocks_world import BlocksWorldEnv
    
    env = BlocksWorldEnv(pb.POSITION_CONTROL, control_period=20, show=True, step_hook = dump.step_hook)
    env.load_blocks(thing_below)

    # from check/camera.py
    pb.resetDebugVisualizerCamera(
        1.2000000476837158, 56.799964904785156, -22.20000648498535,
        (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))
    
    goal_block_above = env.invert(goal_thing_below)
    restacker = Restacker(env, goal_block_above, dump)
    restacker.run()
    
    env.close()

