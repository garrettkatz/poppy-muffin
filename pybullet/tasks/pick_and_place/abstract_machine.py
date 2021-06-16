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

class AbstractMachine:
    def __init__(self, env, num_blocks, max_levels):
        self.env = env
        self.num_blocks = num_blocks
        self.max_levels = max_levels

        self.registers = {
            "ipt": AbstractRegister("ipt", content=0),
            "gts": AbstractRegister("gts"),

            "obj": AbstractRegister("obj"), # object names (blocks and spots)
            "ob2": AbstractRegister("ob2"), # secondary storage for object names, only linked to obj
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
            # location occupancy
            "occ": AbstractConnection("occ", src=self.registers["loc"], dst=self.registers["obj"]),
            # secondary object storage
            "ob2<-obj": AbstractConnection("ob2<-obj", src=self.registers["obj"], dst=self.registers["ob2"]),
            "obj<-ob2": AbstractConnection("obj<-ob2", src=self.registers["ob2"], dst=self.registers["obj"]),
        }

        # keep joint positions symbolic in abstract machine
        self.ik = get_joint_positions(env, num_blocks, max_levels)
        self.connections["ik"].memory = {key: key for key in self.ik}

        for base, level in it.product(range(num_blocks), range(max_levels)):
            self.connections["tc"][(base, level)] = (base, level, 1)
            self.connections["to"][(base, level)] = (base, level, 0)
            self.connections["pc"][(base, level)] = (base, max_levels, 1)
            self.connections["po"][(base, level)] = (base, max_levels, 0)
            self.connections["above"][(base, level)] = (base, level+1)
        
        for obj in self.env.blocks + env.bases:
            self.connections["ob2<-obj"][obj] = obj
            self.connections["obj<-ob2"][obj] = obj

        # constant base locations
        for b, base in enumerate(env.bases):
            self.connections["loc"][base] = (b, 0)
            self.connections["occ"][(b, 0)] = base
            for level in range(1, max_levels+1):
                self.connections["occ"][(b, level)] = "nil"
    
    def get_memories(self):
        memories = {}
        for name, conn in self.connections.items():
            memories[name] = dict(conn.memory)
        return memories
    
    def set_memories(self, memories):
        # overwrites all user-facing memories (but not ipt or gts)
        for name, memory in memories.items():
            if name in ["ipt","gts"]: continue # don't overwrite programming
            self.connections[name].memory = dict(memory)

    def dbg(self):
        for register in self.registers.values(): print(" ", register)
        for connection in self.connections.values(): print(" ", connection)

    def tick(self):
        # returns True when program is done

        # print("ticking...")
        # self.dbg()
        
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
    
    def extend_ipt(self):
        # extend instruction address space beyond current
    
        cur_ipt = self.registers["ipt"].content
        if cur_ipt not in self.connections["ipt"]:

            # associate current ipt with new one
            new_ipt = len(self.connections["ipt"]) + 1
            self.connections["ipt"][cur_ipt] = new_ipt

            # queue default gates at new ipt (one-step delay)
            self.connections["gts"][cur_ipt] = (), ("ipt", "gts")
    
    def instruct(self, store=(), recall=()):

        # always recall ipt and gts
        if "ipt" not in recall: recall += ("ipt",)
        if "gts" not in recall: recall += ("gts",)
        
        # sort for consistency in nvm codec
        store, recall = tuple(sorted(store)), tuple(sorted(recall))
        
        # associate gates with previous ipt (one-step delay)
        old_ipt = self.registers["ipt"].old_content
        self.connections["gts"][old_ipt] = (store, recall)
        
        # tick to new ipt with the new gates applied
        self.extend_ipt()
        self.registers["gts"].content = (store, recall)
        self.tick()
    
    def mov(self, src, dst):
        # helper to move data from src register to dst register
        name = None
        for conn in self.connections.values():
            if (src, dst) != (conn.src.name, conn.dst.name): continue
            if name is not None: raise ValueError("Ambiguous pathway in mov")
            name = conn.name
        if name is None: raise ValueError("Missing pathway in mov")
        self.instruct(recall = (name,))
            
    def start(self):
        # initialize default gates at initial ipt
        self.registers["ipt"].reset(0)
        self.registers["gts"].reset(((), ("ipt", "gts")))
        # add new ipt and tick before first real instruction
        self.extend_ipt()
        self.tick()
        
    def halt(self):
        # empty instruction
        self.instruct()

        # overwrite new ipt with self-loop to indicate end-of-program
        old_ipt = self.registers["ipt"].old_content
        self.connections["ipt"][old_ipt] = old_ipt
        self.registers["ipt"].content = old_ipt
    
    def reset(self, contents):
        for name, register in self.registers.items():
            if name in contents: register.reset(contents[name])
            if name == "ipt": register.reset(0)
            if name == "gts": register.reset(((), ("ipt", "gts")))
        
def pick_up(am):
    # assume obj register has block to pickup
    am.mov(src="obj", dst="loc")
    for conn in ["po","to","tc","pc"]:
        am.instruct(recall = (conn,))
        am.instruct(recall = ("ik",))    

def put_down_on(am):
    # assume obj register has place to put down
    am.instruct(recall = ("loc",))
    am.instruct(recall = ("above",))
    for conn in ["pc","tc","to","po"]:
        am.instruct(recall = (conn,))
        am.instruct(recall = ("ik",))

def move_to(am):
    # assume obj has block to pick and ob2 has place to put
    pick_up(am)
    am.mov(src="ob2", dst="obj")
    put_down_on(am)
    
def program(am):
    move_to(am)

def store_block_locations(env, am):
    for block in env.blocks:
        base, level = env.base_and_level_of(block)
        loc = (env.bases.index(base), level)
        am.connections["loc"][block] = loc
        am.connections["occ"][loc] = block

def make_abstract_machine(env, num_blocks, max_levels):

    am = AbstractMachine(env, num_blocks, max_levels)

    memories = am.get_memories()
    
    # place holders
    store_block_locations(env, am)
    am.reset({"obj":"b0", "ob2":"b1"}) # placeholders
    

    am.start()
    program(am)
    am.halt()

    # erase dynamic user memories from execution
    am.set_memories(memories)

    return am

if __name__ == "__main__":
    
    num_blocks, max_levels = 7, 3
    # thing_below = random_thing_below(num_blocks=7, max_levels=3)
    # thing_below = {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})
    thing_below = {"b%d" % n: "t%d" % n for n in range(num_blocks)}
    thing_below["b6"] = "b1"

    env = BlocksWorldEnv()
    env.load_blocks(thing_below)
    
    am = make_abstract_machine(env, num_blocks, max_levels)

    # thing_below["b6"] = "b3"
    # env.load_blocks(thing_below)

    store_block_locations(env, am)
    am.reset({
        "obj": "b5",
        "ob2": "t6",
        "jnt": (num_blocks//2, max_levels, 0),
    })

    while True:
        done = am.tick()
        position = am.ik[am.registers["jnt"].content]
        am.env.goto_position(position)
        # input('.')
        if done: break
        



