import itertools as it
import sys
sys.path.append('../../envs')    
from blocks_world import BlocksWorldEnv, random_thing_below

def get_joint_positions(env, num_bases, max_levels):
    joint_position = {}
    for base in range(num_bases):
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
    def reset(self):
        self.memory = {}

    def __str__(self):
        return "%s (%s -> %s): %s" % (self.name, self.src.name, self.dst.name, self.memory)
    def __getitem__(self, key):
        if key not in self.memory:
            raise KeyError("%s not memorized in %s" % (key, self))
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
        # creates new ipt and steps to it
        self.old_ipt = self.cur_ipt
        self.cur_ipt = self.machine.get_new_ipt()
        for ipt in [self.old_ipt, self.cur_ipt]:
            self.machine.connections["gts"][ipt] = (), ("gts","ipt")
            self.machine.connections["ipt"][ipt] = self.cur_ipt
    
    def flash(self, routine):
        self.cur_ipt = self.machine.get_new_ipt()
        self.machine.ipt_of[routine.__name__] = self.cur_ipt
        self.machine.inst_at[self.cur_ipt] = "'start %s'" % routine.__name__
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
        self.machine.inst_at[self.old_ipt] = "'ungate'"
    
    def recall(self, conn_name):
        self.ungate(recall = (conn_name,))
        self.machine.inst_at[self.old_ipt] = "'recall %s'" % conn_name

    def store(self, conn_name):
        self.ungate(store = (conn_name,))
        self.machine.inst_at[self.old_ipt] = "'store %s'" % conn_name
    
    def move(self, src, dst):
        name = self.machine.connection_between(src, dst)
        self.recall(name)
        self.machine.inst_at[self.old_ipt] = "'mov %s to %s'" % (src, dst)
    
    def put(self, tok, dst):
        name = "put %s" % dst
        # memorize ipt -> dst token
        self.machine.connections[name][self.cur_ipt] = tok
        # recall memory into dst
        self.recall(name)
        self.machine.inst_at[self.old_ipt] = "'put %s in %s'" % (tok, dst)
    
    def call(self, routine, regs=[]):
        # memorize sub-routine ipt in call connection
        self.machine.connections["call"][self.cur_ipt] = self.machine.ipt_of[routine]
        # set gates for call instruction
        store = ("spt",) # store current ipt at current spt
        recall = tuple(sorted(["call", "gts", "push"])) # call new ipt with default gates and increment spt
        self.machine.connections["gts"][self.old_ipt] = store, recall
        # step once to memorize default gates for sub-routine entry
        self.step_ipt()
        self.machine.inst_at[self.old_ipt] = "'call %s'" % routine
        # step again so next instructions after call do not overwrite default gates
        self.step_ipt()
    
    def ret(self):
        self.recall("pop") # decrement spt
        self.machine.inst_at[self.old_ipt] = "'ret pop'"
        # memorize default gates for return to calling ipt
        self.machine.connections["gts"][self.cur_ipt] = (), ("gts","ipt")
        # set gates for call instruction
        recall = tuple(sorted(["gts","spt"])) # restore calling ipt from spt with default gates
        self.machine.connections["gts"][self.old_ipt] = (), recall
        # no need to link new ipt since it will be restored from spt
        self.machine.inst_at[self.cur_ipt] = "'ret'"

    def ret_if_nil(self):
        # set gates for call instruction
        store = ("spt",) # store current ipt at current spt
        recall = tuple(sorted(["jmp", "gts", "push"])) # call ipt from jmp register with default gates and increment spt
        self.machine.connections["gts"][self.old_ipt] = store, recall
        # step once to memorize default gates for sub-routine entry
        self.step_ipt()
        self.machine.inst_at[self.old_ipt] = "'ret if nil'"
        # step again so next instructions after call do not overwrite default gates
        self.step_ipt()
    
    def push(self, regs=()):
        self.ungate(store=tuple("spt > %s" % reg for reg in regs), recall=("push",))
        self.machine.inst_at[self.old_ipt] = "'push %s'" % ", ".join(regs)

    def pop(self, regs=()):
        self.recall("pop")
        self.machine.inst_at[self.old_ipt] = "'pop dec'"
        self.ungate(recall=tuple("spt > %s" % reg for reg in regs))
        self.machine.inst_at[self.old_ipt] = "'pop %s'" % ", ".join(regs)
    
    def breakpoint(self, message):
        self.machine.message_at[self.cur_ipt] = message

class AbstractMachine:
    def __init__(self, env, num_bases, max_levels, spt_range=32, gen_regs=None):
        self.env = env
        self.num_blocks = num_bases
        self.max_levels = max_levels
        self.num_bases = num_bases
        self.spt_range = spt_range
        self.tick_counter = 0
        
        # maps to starting ipt of each routine
        self.ipt_of = {}
        # maps ipt to compiled instruction
        self.inst_at = {}
        # stores breakpoint messages
        self.message_at = {}

        self.registers = {
            "ipt": AbstractRegister("ipt", content=0),
            "spt": AbstractRegister("spt", content=0),
            "jmp": AbstractRegister("jmp"),
            "gts": AbstractRegister("gts"),

            "obj": AbstractRegister("obj"), # object names (blocks and spots)
            "loc": AbstractRegister("loc"), # (base, level) for object
            "tar": AbstractRegister("tar"), # (base, level/perch, grasp) for gripper
            "jnt": AbstractRegister("jnt"), # joints for target
        }
        
        self.connections = {
            "ipt": AbstractConnection("ipt", src=self.registers["ipt"], dst=self.registers["ipt"]),
            "call": AbstractConnection("call", src=self.registers["ipt"], dst=self.registers["ipt"]),
            "spt": AbstractConnection("spt", src=self.registers["spt"], dst=self.registers["ipt"]),
            "push": AbstractConnection("push", src=self.registers["spt"], dst=self.registers["spt"]),
            "pop": AbstractConnection("pop", src=self.registers["spt"], dst=self.registers["spt"]),
            "jmp": AbstractConnection("jmp", src=self.registers["jmp"], dst=self.registers["ipt"]),
            "gts": AbstractConnection("gts", src=self.registers["ipt"], dst=self.registers["gts"]),

            # memorized inverse kinematics
            "ik": AbstractConnection("ik", src=self.registers["tar"], dst=self.registers["jnt"]),
            # targ/perch open/closed
            "to": AbstractConnection("to", src=self.registers["loc"], dst=self.registers["tar"]),
            "tc": AbstractConnection("tc", src=self.registers["loc"], dst=self.registers["tar"]),
            "po": AbstractConnection("po", src=self.registers["loc"], dst=self.registers["tar"]),
            "pc": AbstractConnection("pc", src=self.registers["loc"], dst=self.registers["tar"]),
            # object locations
            "loc": AbstractConnection("loc", src=self.registers["obj"], dst=self.registers["loc"]),
            "obj": AbstractConnection("obj", src=self.registers["loc"], dst=self.registers["obj"]), # reverse lookup
            # location relations
            "above": AbstractConnection("above", src=self.registers["loc"], dst=self.registers["loc"]),
            "right": AbstractConnection("right", src=self.registers["loc"], dst=self.registers["loc"]),
            # goal thing above
            "goal": AbstractConnection("goal", src=self.registers["obj"], dst=self.registers["obj"]),
            # base list
            "base": AbstractConnection("base", src=self.registers["obj"], dst=self.registers["obj"]),
        }
        
        # stack addresses
        for spt in range(self.spt_range-2):
            self.connections["push"][spt] = spt + 1
            self.connections["pop"][spt + 1] = spt
        
        # keep joint positions symbolic in abstract machine
        self.ik = get_joint_positions(env, num_bases, max_levels)
        self.ik["rest"] = env.get_position()
        self.connections["ik"].memory = {key: key for key in self.ik}

        self.locs = list(it.product(range(num_bases), range(max_levels+1)))
        for base, level in it.product(range(num_bases), range(max_levels)):
            self.connections["tc"][(base, level)] = (base, level, 1)
            self.connections["to"][(base, level)] = (base, level, 0)
            self.connections["pc"][(base, level)] = (base, max_levels, 1)
            self.connections["po"][(base, level)] = (base, max_levels, 0)
            self.connections["above"][(base, level)] = (base, level+1)
            if base + 1 < num_bases:
                self.connections["right"][(base, level)] = (base+1, level)
            else:
                self.connections["right"][(base, level)] = "nil"
        
        # constant base loop
        for b in range(num_bases):
            next_base = env.bases[b+1] if b+1 < len(env.bases) else "nil"
            self.connections["base"][env.bases[b]] = next_base

        # general purpose registers and jmp
        self.blocks = ["b%d" % b for b in range(num_bases)]
        self.objs = self.env.bases + self.blocks
        if gen_regs is None: gen_regs = ["r0", "r1", "r2"]
        for name in gen_regs:
            self.registers[name] = AbstractRegister(name)
        for src, dst in it.permutations(gen_regs + ["jmp"], 2):
            name = "%s > %s" % (src, dst)
            self.connections[name] = AbstractConnection(name, src=self.registers[src], dst=self.registers[dst])
            for token in self.objs + self.locs + ["nil"]: self.connections[name][token] = token
        for reg in gen_regs + ["jmp"]:
            for (src, dst) in [(reg, "obj"), ("obj", reg)]:
                name = "%s > %s" % (src, dst)
                self.connections[name] = AbstractConnection(name, src=self.registers[src], dst=self.registers[dst])
                for token in self.objs + ["nil"]: self.connections[name][token] = token
            for (src, dst) in [(reg, "loc"), ("loc", reg)]:
                name = "%s > %s" % (src, dst)
                self.connections[name] = AbstractConnection(name, src=self.registers[src], dst=self.registers[dst])
                for token in self.locs + ["nil"]: self.connections[name][token] = token            

        # put instruction
        for reg in gen_regs + ["obj", "loc"]:
            name = "put %s" % reg
            self.connections[name] = AbstractConnection(name, src=self.registers["ipt"], dst=self.registers[reg])

        # general registers and stack
        for reg in gen_regs:
            name = "spt > %s" % reg
            self.connections[name] = AbstractConnection(name, src=self.registers["spt"], dst=self.registers[reg])
    
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
        print("****************** dbg: tick %d **********************" % self.tick_counter)
        inst = self.inst_at.get(self.registers["ipt"].content, "internal")
        print(inst)
        for register in self.registers.values(): print(" ", register)
        # for connection in self.connections.values(): print(" ", connection)

    def tick(self):
        # returns True when program is done
        
        # apply current gates
        store, recall = self.registers["gts"].content
        for name in store: self.connections[name].store()
        
        for register in self.registers.values(): register.new_content = register.content
        for name in recall: self.connections[name].recall()
        for register in self.registers.values(): register.update()

        # check breakpoints
        ipt = self.registers["ipt"]
        if ipt.content in self.message_at:
            input(self.message_at[ipt.content])

        self.tick_counter += 1

        # self-loop indicates end-of-program
        if ipt.content == ipt.old_content: return True # program done
        else: return False # program not done
    
    def mount(self, routine):
        # initialize default gates at initial ipt
        self.registers["ipt"].reset(self.ipt_of[routine])
        self.registers["spt"].reset(0) # fresh call stack
        self.registers["gts"].reset(((), ("ipt", "gts")))
        self.tick_counter = 0

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
    
    def run(self, dbg=False):
        self.mount("main")
        if dbg: self.dbg()
        while True:
            done = self.tick()
            if dbg: self.dbg()
            if self.registers["jnt"].content != self.registers["jnt"].old_content:
                position = self.ik[self.registers["jnt"].content]
                self.env.goto_position(position)
            if done: break
        return self.tick_counter

def setup_abstract_machine(env, num_bases, max_levels, gen_regs=None):

    am = AbstractMachine(env, num_bases, max_levels, gen_regs=gen_regs)
    compiler = Compiler(am)
    
    # special firmware routines for return-if-nil
    def rin_not_nil(comp):
        comp.ret()
    def rin_nil(comp):
        comp.recall("pop")
        comp.ret()
    compiler.flash(rin_not_nil)
    compiler.flash(rin_nil)

    # return-if-nil jmp connections
    # need to be added after flashing special rin_programs to get their ipts
    for token in am.objs + am.locs:
        am.connections["jmp"][token] = am.ipt_of["rin_not_nil"]
    am.connections["jmp"]["nil"] = am.ipt_of["rin_nil"]
    
    return am, compiler

def memorize_env(machine, goal_thing_above={}):
    # reset input connections
    machine.connections["obj"].reset()
    machine.connections["loc"].reset()
    machine.connections["goal"].reset()
    # start with all locations empty    
    for loc in machine.locs:
        machine.connections["obj"][loc] = "nil"
    # overwrite non-empty occupancies
    env = machine.env
    for block in env.blocks:
        base, level = env.base_and_level_of(block)
        loc = (env.bases.index(base), level)
        machine.connections["obj"][loc] = block
        machine.connections["loc"][block] = loc
    for thing, block in goal_thing_above.items():
        machine.connections["goal"][thing] = block

# testing routines
def test_rin(comp):
    comp.move("r1", "jmp")
    comp.ret_if_nil()
    comp.move("r0", "r1")
    comp.ret()

def test_push(comp):
    comp.put("b0", "r0")
    comp.put("b0", "r1")
    comp.push(("r0",))
    comp.put("b1", "r0")
    comp.push(("r0",))
    comp.put("b0", "r0")
    comp.pop(("r0",))
    comp.pop(("r0",))
    comp.ret()

# MACRO, don't flash
def pick_up(comp):
    # r0: obj to pick; r1: loc to place
    
    # # orig
    # comp.move("r0", "obj")
    # comp.recall("loc")
    # for conn in ["po","to","tc","pc"]:
    #     comp.recall(conn)
    #     comp.recall("ik")
    # # unbind r0 from location
    # comp.put("nil", "obj")
    # comp.store("obj")
    # comp.move("r0", "obj")
    # comp.put("nil", "loc")
    # comp.store("loc")
    
    # optimized
    comp.move("r0", "obj")
    comp.recall("loc")
    comp.recall("po")
    for conn in ["to","tc","pc"]: comp.ungate(recall=("ik",conn))
    comp.recall("ik")
    comp.put("nil", "obj")
    comp.ungate(store=("obj",), recall=("r0 > obj",))
    comp.put("nil", "loc")
    comp.store("loc")

# MACRO, don't flash
def put_down(comp):
    # r0: obj to pick; r1: loc to place
    
    # # orig
    # comp.move("r1", "loc")
    # for conn in ["pc","tc","to","po"]:
    #     comp.recall(conn)
    #     comp.recall("ik")
    # # bind r0 to new location
    # comp.move("r0", "obj")
    # comp.store("loc")
    # comp.store("obj")
    
    # opt
    comp.move("r1", "loc")
    comp.recall("pc")
    for conn in ["tc","to","po"]: comp.ungate(recall=("ik", conn))
    comp.recall("ik")
    # bind r0 to new location
    comp.move("r0", "obj")
    comp.ungate(store=("loc","obj"))

# MACRO, don't flash
def move_to(comp):
    # r0: obj to pick; r1: loc to place
    pick_up(comp)
    put_down(comp)
    # comp.call("pick_up")
    # comp.call("put_down")
    # comp.ret()

def free_spot(comp):
    # after recursion, loc register has free spot
    # comp.put((0,0), "loc") # before top-level recursive call
    comp.recall("obj")
    comp.move("obj", "jmp")
    comp.ret_if_nil() # nil object means free spot
    comp.recall("right") # not free, move right
    comp.move("loc", "jmp")
    comp.ret_if_nil() # nil loc means done checking
    comp.call("free_spot")
    comp.ret()

def unstack_from(comp):
    # r0: block to be cleared
    # do not initiate if block is nil
    comp.move("r0", "jmp")
    comp.ret_if_nil()
    
    # block = self.env.block_above[thing]
    comp.move("r0", "obj")
    comp.recall("loc")
    comp.recall("above")
    comp.recall("obj")

    # if block == "none": return
    comp.move("obj", "jmp")
    comp.ret_if_nil()

    # self.unstack_from(block)
    comp.move("obj", "r0")
    comp.push(regs=("r0",))
    comp.call("unstack_from")
    comp.pop(regs=("r0",))

    # self.move_to(self.free_spot(), block)
    comp.put((0,0), "loc")
    comp.call("free_spot")
    comp.move("loc", "r1")
    move_to(comp)
    
    # return
    comp.ret()

def unstack_all(comp):
    # comp.put((0,0), "loc") # before top-level recursive call

    # for base in self.env.bases:
    comp.move("loc", "jmp")
    comp.ret_if_nil()

    #     block = self.env.block_above[base]
    comp.recall("obj")

    #     if block != "none": # handled in unstack_from
    #         self.unstack_from(block)
    comp.move("loc", "r0")
    comp.push(regs=("r0",))
    comp.move("obj", "r0")
    comp.call("unstack_from")
    
    # continue loop recursively
    comp.pop(regs=("r0",))
    comp.move("r0", "loc")
    comp.recall("right")
    comp.call("unstack_all")

    # return
    comp.ret()

def stack_on(comp):
    # r0 should have thing to stack on
    
    # # orig
    # # don't initiate if thing itself is nil
    # comp.move("r0", "jmp")
    # comp.ret_if_nil()
    
    # # get location to stack on in r1
    # comp.move("r0", "obj")
    # comp.recall("loc")
    # comp.recall("above")
    # comp.move("loc", "r1")

    # # block = self.goal_block_above[thing]
    # comp.recall("goal")

    # # if block == "none": return
    # comp.move("obj", "jmp")
    # comp.ret_if_nil()

    # # self.move_to(thing, block)
    # comp.move("obj", "r0")
    # move_to(comp)
    
    # # self.stack_on(block)
    # comp.call("stack_on")

    # # return
    # comp.ret()

    # opt
    comp.move("r0", "jmp")
    comp.ret_if_nil()    
    comp.move("r0", "obj")
    comp.ungate(recall=("goal","loc"))
    comp.recall("above")
    comp.move("loc", "r1")
    comp.ungate(recall=("loc > r1", "obj > jmp", "obj > r0"))
    comp.ret_if_nil()
    move_to(comp)    
    comp.call("stack_on")
    comp.ret()

def stack_all(comp):
    # comp.put("t0", "obj") # before top-level recursive call
    
    # # orig
    # # save current base in r0
    # comp.move("obj", "r0")

    # # for base in self.env.bases:
    # comp.move("obj", "jmp")
    # comp.ret_if_nil()

    # #     block = self.goal_block_above[base]
    # comp.recall("goal")
    
    # #     if block != 'none': # checked inside stack_on
    # #         self.stack_on(block)
    # comp.push(regs=("r0",))
    # comp.move("obj", "r0")
    # comp.call("stack_on")

    # # continue loop recursively
    # comp.pop(regs=("r0",))
    # comp.move("r0", "obj")
    # comp.recall("base")
    # comp.call("stack_all")

    # # return
    # comp.ret()

    # opt
    comp.ungate(recall=("obj > jmp", "obj > r0"))
    comp.ret_if_nil()
    comp.recall("goal")
    comp.push(regs=("r0",))
    comp.move("obj", "r0")
    comp.call("stack_on")
    comp.pop(regs=("r0",))
    comp.move("r0", "obj")
    comp.recall("base")
    comp.call("stack_all")
    comp.ret()

def main(comp):

    comp.put((0, 0), "loc")
    comp.call("unstack_all")
    comp.put("t0", "obj")
    comp.call("stack_all")

    # comp.put("b0", "r0")
    # comp.call("stack_on")

    # comp.put("b0", "r0")
    # comp.put((1,1), "r1")
    # move_to(comp)

    # comp.call("test_rin")
    # comp.call("test_push")

    # comp.put((0,1), "loc")
    # comp.call("free_spot")

def make_abstract_machine(env, num_bases, max_levels, gen_regs=None):

    am, compiler = setup_abstract_machine(env, num_bases, max_levels, gen_regs=gen_regs)

    # # tests
    # compiler.flash(test_rin)
    # compiler.flash(test_push)

    # block restacking routines
    compiler.flash(free_spot)
    compiler.flash(unstack_from)
    compiler.flash(unstack_all)
    compiler.flash(stack_on)
    compiler.flash(stack_all)

    compiler.flash(main)

    return am

if __name__ == "__main__":
    
    # num_bases = 7
    # # num_blocks, max_levels = 7, 3
    # num_blocks, max_levels = 4, 3
    # thing_below = random_thing_below(num_blocks, max_levels, num_bases)
    # goal_thing_below = random_thing_below(num_blocks, max_levels, num_bases)

    # one failure case:
    max_levels = 3
    num_blocks = 5
    num_bases = 5
    thing_below = {'b0': 't1', 'b2': 'b0', 'b4': 'b2', 'b1': 't4', 'b3': 't2'}
    goal_thing_below = {'b1': 't1', 'b2': 't3', 'b3': 'b2', 'b0': 't0', 'b4': 'b0'}

    env = BlocksWorldEnv(show=True)
    env.load_blocks(thing_below, num_bases)
    am = make_abstract_machine(env, num_bases, max_levels)

    goal_thing_above = env.invert(goal_thing_below)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"
    memorize_env(am, goal_thing_above)

    # restack test
    am.reset({
        "jnt": "rest",
    })    
    num_ticks = am.run(dbg=True)
    
    input('...')
    env.close()
    print("ticks:", num_ticks)


