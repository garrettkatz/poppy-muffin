import torch as tr
import itertools as it
from abstract_machine import make_abstract_machine, memorize_env

def hadamard_matrix(N):
    H = tr.tensor([[1.]])
    while H.shape[0] < N:
        H = tr.cat((
            tr.cat((H,  H), dim=1),
            tr.cat((H, -H), dim=1),
        ), dim=0)
    # return H / H.shape[0]**.5
    return H

class NVMRegister:
    def __init__(self, name, size, codec, σ=None):
        if σ is None: σ = lambda v: v

        self.name = name
        self.size = size
        self.codec = codec
        self.σ = σ

        self.content = tr.zeros(self.size)
        self.old_content = tr.zeros(self.size)
        self.new_content = tr.zeros(self.size)
        
        # faster decoding
        self.decode_tokens = list(codec.keys())
        self.decode_matrix = tr.stack([codec[token] for token in self.decode_tokens])
        
    def __str__(self):
        return "reg %s: %s, %s, %s" % (self.name,
            self.decode(self.old_content),
            self.decode(self.content),
            self.decode(self.new_content)
        )
    def reset(self, content):
        self.content = content
        self.new_content = tr.zeros(self.size)
        self.old_content = tr.zeros(self.size)
    def activate(self):
        self.content = self.σ(self.content)        
    def update(self):
        # shift buffers
        self.old_content = self.content
        self.content = self.new_content
        self.new_content = tr.zeros(self.size)
    def encode(self, token):
        if token not in self.codec:
            raise KeyError("%s not in %s codec" % (token, self.name))
        return self.codec[token]
    def decode(self, content):
        # for token, pattern in self.codec.items():
        #     if (pattern*content > 0).all(): return token
        # return None
        return self.decode_tokens[self.decode_matrix.mv(content).argmax()]

def FSER(W, x, y):
    # dW = tr.outer(y - W.mv(x), x) / float(W.shape[1])
    dW = (y - W.mv(x)).reshape(-1,1) * x / float(W.shape[1]) # backwards compatible
    return dW

class NVMConnection:
    def __init__(self, name, src, dst):
        self.name = name
        self.src = src
        self.dst = dst
        self.W = tr.zeros(dst.size, src.size)
    def __str__(self):
        return "%s (%s -> %s)" % (self.name, self.src.name, self.dst.name)
    def __setitem__(self, key, val):
        x = self.src.encode(key)
        y = self.dst.encode(val)
        self.W = self.W + FSER(self.W, x, y)

    def reset(self, W=None):
        if W is None: W = tr.zeros(self.dst.size, self.src.size)
        self.W = W

    def store(self, gate = 1.0):
        dW = FSER(self.W, self.src.content, self.dst.content)
        # self.W += gate * dW # bad if self.W is a leaf variable requiring grad
        self.W = self.W + gate * dW
    def recall(self, gate = 1.0):
        # self.dst.new_content = self.dst.new_content * (1. - gate)
        self.dst.new_content = self.dst.new_content + self.W.mv(self.src.content) * gate

class NeuralVirtualMachine:
    def __init__(self, env, registers, connections):
        self.registers = registers
        self.connections = connections
        self.connection_names = list(sorted(connections.keys()))
        self.env = env
        self.tick_counter = 0
    
    def dbg(self):
        print("****************** dbg: tick %d **********************" % self.tick_counter)
        print(self.inst_at.get(self.registers["ipt"].decode(self.registers["ipt"].content), "internal"))
        for register in self.registers.values():
            print(" ", register)
            print(register.content.detach().numpy())
        # for connection in self.connections.values(): print(" ", connection)

    def tick(self, diff_gates=False):
        
        # apply current gates
        gates = self.registers["gts"].content
        split = int(len(gates)/2)
        gs, gr = gates[:split], gates[split:] # store, recall
        
        # storage
        for c, name in enumerate(self.connection_names):
            # if diff_gates or gs[c] > 0.5: self.connections[name].store(gs[c])
            if diff_gates: self.connections[name].store(gs[c])
            elif gs[c] > 0.5: self.connections[name].store()

        # # recall
        # for register in self.registers.values():
        #     register.new_content = register.content.clone() # clone important since gr is a view, don't want to update gates
        # recalled = set() # track if each register is recall target
        # for c, name in enumerate(self.connection_names):
        #     # if diff_gates or gr[c] > 0.5:
        #     #     self.connections[name].recall(gr[c])
        #     dst = self.connections[name].dst
        #     if diff_gates:
        #         if dst.name not in recalled: dst.new_content = dst.new_content * (1 - gr[c])
        #         self.connections[name].recall(gr[c])
        #     elif gr[c] > 0.5:
        #         if dst.name not in recalled: dst.new_content = tr.zeros(dst.size)
        #         self.connections[name].recall()
        #     recalled.add(dst.name)

        # recall
        recalled = set()
        for register in self.registers.values():
            register.new_content = tr.zeros(register.size)
        for c, name in enumerate(self.connection_names):
            if diff_gates:
                self.connections[name].recall(gr[c])
                recalled.add(self.connections[name].dst.name)
            elif gr[c] > 0.5:
                self.connections[name].recall()
                recalled.add(self.connections[name].dst.name)

        for register in self.registers.values():
            if register.name not in recalled: register.new_content = register.content

        # shift buffers and apply activation function
        for register in self.registers.values():
            register.update()
            register.activate()

        self.tick_counter += 1

        # self-loop indicates end-of-program
        ipt = self.registers["ipt"]
        if ipt.decode(ipt.content) == ipt.decode(ipt.old_content): return True # program done
        else: return False # program not done

    def mount(self, routine):
        # initialize default gates at initial ipt
        self.registers["ipt"].reset(self.ipt_of[routine])
        self.registers["spt"].reset(self.registers["spt"].encode(0))
        self.registers["gts"].reset(self.registers["gts"].encode(((), ("gts","ipt"))))
        self.tick_counter = 0

    def reset(self, contents):
        for name, register in self.registers.items():
            register.reset(tr.zeros(register.size))
            if name in contents: register.reset(contents[name])
            if name == "ipt": register.reset(register.encode(0))
            if name == "gts": register.reset(register.encode(((), ("gts","ipt"))))
    
    def size(self):
        reg_sizes, conn_sizes, total = {}, {}, 0
        for name, reg in self.registers.items():
            reg_sizes[name] = (reg.size, len(reg.codec))
        for name, conn in self.connections.items():
            conn_sizes[name] = conn.W.shape
            total += conn.W.shape[0] * conn.W.shape[1]
        return reg_sizes, conn_sizes, total
    
    def run(self, dbg=False):
        self.mount("main")
        if dbg: self.dbg()
        target_changed = True
        while True:
            done = self.tick()
            if dbg: self.dbg()
            if target_changed:
                position = self.registers["jnt"].content.detach().numpy()
                self.env.goto_position(position)
            tar = self.registers["tar"]
            target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))
            if done: break
        return self.tick_counter
    
    def get_state(self):
        registers = {name: reg.content for name, reg in self.registers.items()}
        connections = {name: conn.W for name, conn in self.connections.items()}
        return registers, connections
    def reset_state(self, registers, connections):
        for name, content in registers.items():
            self.registers[name].reset(content)
        for name, W in connections.items():
            self.connections[name].reset(W)
        

def hadamard_codec(tokens):
    N = len(tokens)
    H = hadamard_matrix(N)
    return H.shape[0], {token: H[t] for t, token in enumerate(tokens)}

def virtualize(am, σ=None):
    
    registers = {}
    
    tokens = {
        "ipt": list(range(len(am.connections["ipt"].memory)+1)),
        "spt": list(range(am.spt_range)),
        "loc": am.locs + ["nil"],
        "tar": list(it.product(range(am.num_blocks), range(am.max_levels+1), [0, 1])) + ["rest"],
        "obj": am.objs + ["nil"]
    }
    for name in ["r0", "r1", "r2", "jmp"]:
        tokens[name] = am.objs + am.locs + ["nil"]

    for name in tokens:
        size, codec = hadamard_codec(tokens[name])
        registers[name] = NVMRegister(name, size, codec, σ=σ)

    jnt_codec = {key: tr.tensor(val).float() for key, val in am.ik.items()}
    registers["jnt"] = NVMRegister("jnt", am.env.num_joints, jnt_codec)

    gts_codec = {}
    connection_names = list(sorted(am.connections.keys()))
    for store, recall in am.connections["gts"].memory.values():
        gs = tr.tensor([1. if name in store else 0. for name in connection_names])
        gr = tr.tensor([1. if name in recall else 0. for name in connection_names])
        gts_codec[store, recall] = tr.cat((gs, gr))
    registers["gts"] = NVMRegister("gts", 2*len(connection_names), gts_codec)

    connections = {}
    for name, am_conn in am.connections.items():
        src, dst = registers[am_conn.src.name], registers[am_conn.dst.name]
        connections[name] = NVMConnection(name, src, dst)
        for key, val in am_conn.memory.items(): connections[name][key] = val
        
    nvm = NeuralVirtualMachine(am.env, registers, connections)
    
    nvm.ipt_of = {
        routine: registers["ipt"].encode(ipt)
        for routine, ipt in am.ipt_of.items()}
    nvm.inst_at = dict(am.inst_at)

    nvm.objs = list(am.objs)
    nvm.locs = list(am.locs)

    return nvm
    
if __name__ == "__main__":
    
    x = tr.tensor([1., 1., -1.])
    y = tr.tensor([-1., 1., 1.])
    W = tr.zeros(3,3)
    FSER(W, x, y)
    input('.')


        
    import sys
    sys.path.append('../../envs')
    from blocks_world import BlocksWorldEnv, random_thing_below

    num_bases = 3
    # num_blocks, max_levels = 7, 3
    num_blocks, max_levels = 3, 3
    thing_below = random_thing_below(num_blocks, max_levels, num_bases)
    goal_thing_below = random_thing_below(num_blocks, max_levels, num_bases)

    # # thing_below = {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})
    # thing_below = {"b%d" % n: "t%d" % n for n in range(num_blocks)}
    # thing_below["b0"] = "b1"
    # # thing_below["b3"] = "b2"
    # goal_thing_below = {"b%d" % n: "t%d" % n for n in range(num_blocks)}
    # goal_thing_below.update({"b1": "b0", "b2": "b3"})

    env = BlocksWorldEnv()    
    env.load_blocks(thing_below, num_bases)
    am = make_abstract_machine(env, num_bases, max_levels)
    nvm = virtualize(am)

    goal_thing_above = env.invert(goal_thing_below)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"
    memorize_env(nvm, goal_thing_above)

    # # rin test
    # nvm.reset({
    #     "r0": nvm.registers["r0"].encode("b0"),
    #     "r1": nvm.registers["r1"].encode("nil"),
    #     "jnt": tr.tensor(am.ik["rest"]).float()
    # })

    # block stacking test
    nvm.reset({
        "r0": nvm.registers["r0"].encode("b0"),
        "r1": nvm.registers["r1"].encode("b1"),
        "jnt": tr.tensor(am.ik["rest"]).float()
    })
    
    # _, _, total = nvm.size()
    # input("%d weights total!" % total)

    nvm.run(dbg=True)

    env.close()    

