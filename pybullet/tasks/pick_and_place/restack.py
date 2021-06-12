import pybullet as pb
import time, sys

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
        delta = action - position
        rgb, _, _, coords_of = env.get_camera_image()
        self.data[-1]["records"].append((position, delta, rgb, coords_of))

class Restacker:
    def __init__(self, env, goal_block_above, dump=None):
        self.env = env
        self.goal_block_above = goal_block_above
        self.dump = dump

    def free_spot(self):
        for base in self.env.bases:
            if self.env.block_above[base] == 'none': return base
        return False
    
    def run_trajectory(self, quat, waypoints):
        for point, delta in waypoints: 
            targs = self.env.tip_targets_around(point, quat, delta)
            angles = self.env.inverse_kinematics([5, 7], targs)
            self.env.goto_position(angles, .25)
    
    def pick_up(self, block):
        if self.dump is not None: dump.add_command(("pick_up", (block,)))
        pos, quat = self.env.placement_of(block)
        stage = pos[:2] + (.1,)
        self.run_trajectory(quat, [(stage, .02), (pos, .02), (pos, .01), (stage, .01)])
        self.env.block_above[self.env.thing_below[block]] = 'none'

    def put_down_on(self, thing, block):
        if self.dump is not None: dump.add_command(("put_down_on", (thing, block)))
        pos, quat = self.env.placement_of(thing)
        pos = pos[:2] + (pos[2] + .0201,)
        stage = pos[:2] + (.1,)    
        self.run_trajectory(quat, [(stage, .005), (pos, .005), (pos, .02), (stage, .02)])
        self.env.block_above[thing] = block
        self.env.thing_below[block] = thing

    def unstack_from(self, thing):
        block = self.env.block_above[thing]
        if block == "none": return
        self.unstack_from(block)
        self.pick_up(block)
        self.put_down_on(self.free_spot(), block)
    
    def unstack_all(self):
        for base in self.env.bases:
            block = self.env.block_above[base]
            if block != "none": self.unstack_from(block)
    
    def stack_on(self, thing):
        block = self.goal_block_above[thing]
        if block == "none": return
        self.pick_up(block)
        self.put_down_on(thing, block)
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

