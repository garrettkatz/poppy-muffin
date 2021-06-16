import itertools as it
import sys
sys.path.append('../../envs')    
from blocks_world import BlocksWorldEnv, random_thing_below

def get_joint_positions(env, num_blocks, max_levels):
    joint_position = {}
    for base in range(num_blocks):
        for level in range(max_levels+1):
            pos, quat = env.placement_of("t%d" % base)
            pos = (pos[0], pos[1], .01 + level * .02)
            for (grasp, delta) in [(0, .02), (1, .01)]:
                targs = env.tip_targets_around(pos, quat, delta)
                target_position = env.inverse_kinematics([5, 7], targs)
                joint_position[base, level, grasp] = target_position
    return joint_position

class AbstractRegister:
    def __init__(self, name, content=None):
        self.name = name
        self.reset(content)
    def __str__(self):
        return "reg %s: %s, %s, %s" % (self.name, self.old_content, self.content, self.new_content)
    def reset(self, content):
        self.content = content
        self.new_content = None
        self.old_content = None
    def update(self):
        self.old_content = self.content
        self.content = self.new_content
        self.new_content = None

class AbstractConnection:
    def __init__(self, name, src, dst):
        self.name = name
        self.src = src
        self.dst = dst
        self.memory = {}

    def __str__(self):
        return "%s (%s -> %s): %s" % (self.name, self.src.name, self.dst.name, self.memory)
    def __getitem__(self, key):
        return self.memory[key]
    def __setitem__(self, key, value):
        self.memory[key] = value
    def __contains__(self, item):
        return item in self.memory
    def __len__(self):
        return len(self.memory)

    def store(self):
        key = self.src.content
        val = self.dst.content
        self[key] = val
    def recall(self):
        key = self.src.content
        self.dst.new_content = self[key]

class Compiler:
    def __init__(self, machine):
        self.machine = machine
        self.old_ipt = None
        self.cur_ipt = None
    
    def step_ipt(self):
        self.old_ipt = self.cur_ipt
        self.cur_ipt = self.machine.get_new_ipt()
        for ipt in [self.old_ipt, self.cur_ipt]:
            self.machine.connections["gts"][ipt] = (), ("gts","ipt")
            self.machine.connections["ipt"][ipt] = self.cur_ipt
    
    def flash(self, routine):
        self.cur_ipt = self.machine.get_new_ipt()
        self.machine.ipt_of[routine.__name__] = self.cur_ipt
        self.step_ipt()
        routine(self)

    def ungate(self, store=(), recall=()):
        # always recall ipt and gts
        if "gts" not in recall: recall += ("gts",)
        if "ipt" not in recall: recall += ("ipt",)
        
        # sort for consistency in nvm codec
        store, recall = tuple(sorted(store)), tuple(sorted(recall))
        
        # associate gates and new ipt with previous ipt (one-step delay)
        self.machine.connections["gts"][self.old_ipt] = (store, recall)
        self.step_ipt()
    
    def recall(self, conn_name):
        self.ungate(recall = (conn_name,))
    
    def mov(self, src, dst):
        name = self.machine.connection_between(src, dst)
        self.recall(name)
        

class AbstractMachine:
    def __init__(self, env, num_blocks, max_levels):
        self.env = env
        self.num_blocks = num_blocks
        self.max_levels = max_levels
        
        # maps to starting ipt of each routine
        self.ipt_of = {}

        self.registers = {
            "ipt": AbstractRegister("ipt", content=0),
            "gts": AbstractRegister("gts"),

            "obj": AbstractRegister("obj"), # object names (blocks and spots)
            "loc": AbstractRegister("loc"), # (base, level) for object
            "tar": AbstractRegister("tar"), # (base, level/perch, grasp) for gripper
            "jnt": AbstractRegister("jnt"), # joints for target
        }
        
        self.connections = {
            "ipt": AbstractConnection("ipt", src=self.registers["ipt"], dst=self.registers["ipt"]),
            "gts": AbstractConnection("gts", src=self.registers["ipt"], dst=self.registers["gts"]),

            # memorized inverse kinematics
            "ik": AbstractConnection("ik", src=self.registers["tar"], dst=self.registers["jnt"]),
            # targ/perch open/closed
            "to": AbstractConnection("to", src=self.registers["loc"], dst=self.registers["tar"]),
            "tc": AbstractConnection("tc", src=self.registers["loc"], dst=self.registers["tar"]),
            "po": AbstractConnection("po", src=self.registers["loc"], dst=self.registers["tar"]),
            "pc": AbstractConnection("pc", src=self.registers["loc"], dst=self.registers["tar"]),
            # location relations
            "above": AbstractConnection("above", src=self.registers["loc"], dst=self.registers["loc"]),
            # object locations
            "loc": AbstractConnection("loc", src=self.registers["obj"], dst=self.registers["loc"]),
        }

        # keep joint positions symbolic in abstract machine
        self.ik = get_joint_positions(env, num_blocks, max_levels)
        self.ik["rest"] = env.get_position()
        self.connections["ik"].memory = {key: key for key in self.ik}

        for base, level in it.product(range(num_blocks), range(max_levels)):
            self.connections["tc"][(base, level)] = (base, level, 1)
            self.connections["to"][(base, level)] = (base, level, 0)
            self.connections["pc"][(base, level)] = (base, max_levels, 1)
            self.connections["po"][(base, level)] = (base, max_levels, 0)
            self.connections["above"][(base, level)] = (base, level+1)
        
        # constant base locations
        for b, base in enumerate(env.bases):
            self.connections["loc"][base] = (b, 0)

        # general purpose registers
        gen_regs = ["r0", "r1"]
        gen_toks = self.env.bases + self.env.blocks
        for name in gen_regs:
            self.registers[name] = AbstractRegister(name)
        for src, dst in it.permutations(gen_regs + ["obj"], 2):
            name = "%s > %s" % (src, dst)
            self.connections[name] = AbstractConnection(name, src=self.registers[src], dst=self.registers[dst])
            for token in gen_toks:
                self.connections[name][token] = token
    
    def get_memories(self):
        memories = {}
        for name, conn in self.connections.items():
            memories[name] = dict(conn.memory)
        return memories
    
    def set_memories(self, memories):
        # overwrites all user-facing memories (but not ipt or gts)
        for name, memory in memories.items():
            if name in ["gts","ipt"]: continue # don't overwrite programming
            self.connections[name].memory = dict(memory)

    def dbg(self):
        for register in self.registers.values(): print(" ", register)
        for connection in self.connections.values(): print(" ", connection)

    def tick(self):
        # returns True when program is done
        
        # apply current gates
        store, recall = self.registers["gts"].content
        for name in store: self.connections[name].store()
        
        for register in self.registers.values(): register.new_content = register.content
        for name in recall: self.connections[name].recall()
        for register in self.registers.values(): register.update()

        # self-loop indicates end-of-program
        ipt = self.registers["ipt"]
        if ipt.content == ipt.old_content: return True # program done
        else: return False # program not done
    
    def mount(self, routine):
        # initialize default gates at initial ipt
        self.registers["ipt"].reset(self.ipt_of[routine])
        self.registers["gts"].reset(((), ("ipt", "gts")))        

    def get_new_ipt(self):
        new_ipt = len(self.connections["ipt"]) + 1
        self.connections["ipt"][new_ipt] = new_ipt # self-loop by default
        return new_ipt
        
    def connection_between(self, src, dst):
        name = None
        for conn in self.connections.values():
            if (src, dst) != (conn.src.name, conn.dst.name): continue
            if name is not None: raise ValueError("Multiple connections from %s to %s" % (src, dst))
            name = conn.name
        if name is None: raise ValueError("No connections from %s to %s" % (src, dst))
        return name

    def reset(self, contents):
        for name, register in self.registers.items():
            if name in contents: register.reset(contents[name])
            if name == "ipt": register.reset(0)
            if name == "gts": register.reset(((), ("ipt", "gts")))

# def put_down_on(am):
#     # assume obj register has place to put down
#     am.instruct(recall = ("loc",))
#     am.instruct(recall = ("above",))
#     for conn in ["pc","tc","to","po"]:
#         am.instruct(recall = (conn,))
#         am.instruct(recall = ("ik",))

# def move_to(am):
#     # assume obj has block to pick and ob2 has place to put
#     pick_up(am)
#     am.mov(src="ob2", dst="obj")
#     put_down_on(am)
    
# def program(am):
#     move_to(am)

def restore_env(machine):
    env = machine.env
    for block in env.blocks:
        base, level = env.base_and_level_of(block)
        loc = (env.bases.index(base), level)
        machine.connections["loc"][block] = loc

def pickup(comp):
    # assume r0 has block to pickup
    comp.mov("r0", "obj")
    comp.recall("loc")
    for conn in ["po","to","tc","pc"]:
        comp.recall(conn)
        comp.recall("ik")

def rout(comp):
    pickup(comp)

def make_abstract_machine(env, num_blocks, max_levels):

    am = AbstractMachine(env, num_blocks, max_levels)

    # memories = am.get_memories()
    
    # # place holders
    # restore_env(am)

    compiler = Compiler(am)
    compiler.flash(rout)

    # erase dynamic user memories from execution
    # am.set_memories(memories)

    return am

if __name__ == "__main__":
    
    # num_blocks, max_levels = 7, 3
    num_blocks, max_levels = 2, 1
    # thing_below = random_thing_below(num_blocks=7, max_levels=3)
    # thing_below = {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})
    thing_below = {"b%d" % n: "t%d" % n for n in range(num_blocks)}
    # thing_below["b6"] = "b1"

    env = BlocksWorldEnv()
    env.load_blocks(thing_below)
    
    am = make_abstract_machine(env, num_blocks, max_levels)

    # thing_below["b6"] = "b3"
    env.reset()
    env.load_blocks(thing_below)

    restore_env(am)
    am.reset({
        "r0": "b0",
        # "obj": "b0",
        # "jnt": (num_blocks//2, max_levels, 0),
        "jnt": "rest",
    })

    am.mount("rout")
    am.dbg()
    while True:
        # input('.')
        done = am.tick()
        am.dbg()
        position = am.ik[am.registers["jnt"].content]
        am.env.goto_position(position)
        if done: break
        



