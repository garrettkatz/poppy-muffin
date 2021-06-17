import torch as tr
import itertools as it
from abstract_machine import make_abstract_machine, restore_env

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
    def __init__(self, name, size, codec):
        self.name = name
        self.size = size
        self.codec = codec
        self.content = tr.zeros(self.size)
        self.old_content = tr.zeros(self.size)
        self.new_content = tr.zeros(self.size)
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
    def update(self):
        # activation?
        # shift buffers
        self.old_content = self.content
        self.content = self.new_content
        self.new_content = tr.zeros(self.size)
    def encode(self, token):
        if token not in self.codec:
            raise KeyError("%s not in %s codec" % (token, self.name))
        return self.codec[token]
    def decode(self, content):
        for token, pattern in self.codec.items():
            if (pattern*content > 0).all(): return token
        return None

def FSER(W, x, y):
    x, y = x.reshape(-1,1), y.reshape(-1,1)
    dW = (y - W.mm(x)) * x.t() / W.shape[1]
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
        self.W += FSER(self.W, x, y)

    def store(self, gate = 1.0):
        dW = FSER(self.W, self.src.content, self.dst.content)
        self.W += gate * dW
    def recall(self, gate = 1.0):
        self.dst.new_content *= 1. - gate
        self.dst.new_content += self.W.mv(self.src.content) * gate
    
class NeuralVirtualMachine:
    def __init__(self, env, registers, connections):
        self.registers = registers
        self.connections = connections
        self.connection_names = list(sorted(connections.keys()))
        self.env = env
    
    def dbg(self):
        print("****************** dbg **********************")
        print(self.inst_at.get(self.registers["ipt"].decode(self.registers["ipt"].content), "internal"))
        for register in self.registers.values(): print(" ", register)
        # for connection in self.connections.values(): print(" ", connection)

    def tick(self):
        
        # apply current gates
        gates = self.registers["gts"].content
        split = int(len(gates)/2)
        gs, gr = gates[:split], gates[split:] # store, recall
        for c, name in enumerate(self.connection_names): self.connections[name].store(gs[c])

        for register in self.registers.values(): register.new_content = register.content.clone() # clone important since gr is a view
        for c, name in enumerate(self.connection_names): self.connections[name].recall(gr[c])
        for register in self.registers.values(): register.update()

        # self-loop indicates end-of-program
        ipt = self.registers["ipt"]
        if ipt.decode(ipt.content) == ipt.decode(ipt.old_content): return True # program done
        else: return False # program not done

    def mount(self, routine):
        # initialize default gates at initial ipt
        self.registers["ipt"].reset(self.ipt_of[routine])
        self.registers["spt"].reset(self.registers["spt"].encode(0))
        self.registers["gts"].reset(self.registers["gts"].encode(((), ("gts","ipt"))))

    def reset(self, contents):
        for name, register in self.registers.items():
            register.reset(tr.zeros(register.size))
            if name in contents: register.reset(contents[name])
            if name == "ipt": register.reset(register.encode(0))
            if name == "gts": register.reset(register.encode(((), ("gts","ipt"))))

def hadamard_codec(tokens):
    N = len(tokens)
    H = hadamard_matrix(N)
    return H.shape[0], {token: H[t] for t, token in enumerate(tokens)}

def virtualize(am):
    
    registers = {}
    
    objs = am.env.bases + am.env.blocks
    locs = list(it.product(range(num_blocks), range(max_levels+1)))
    tokens = {
        "ipt": list(range(len(am.connections["ipt"].memory)+1)),
        "spt": list(range(am.spt_range)),
        "loc": locs + ["nil"],
        "tar": list(it.product(range(am.num_blocks), range(am.max_levels+1), [0, 1])) + ["rest"],
        "obj": objs + ["nil"]
    }
    for name in ["r0", "r1", "r2", "jmp"]:
        tokens[name] = objs + locs + ["nil"]

    for name in tokens:
        size, codec = hadamard_codec(tokens[name])
        registers[name] = NVMRegister(name, size, codec)

    jnt_codec = {key: tr.tensor(val) for key, val in am.ik.items()}
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

    return nvm
    
if __name__ == "__main__":
        
    import sys
    sys.path.append('../../envs')
    from blocks_world import BlocksWorldEnv, random_thing_below

    # num_blocks, max_levels = 7, 3
    num_blocks, max_levels = 4, 3
    # thing_below = random_thing_below(num_blocks=7, max_levels=3)
    # thing_below = {"b0": "t0", "b1": "t1", "b2": "t2", "b3": "b2", "b4": "b3", "b5": "t5", "b6":"b5"})
    thing_below = {"b%d" % n: "t%d" % n for n in range(num_blocks)}
    # thing_below["b1"] = "b0"
    # thing_below["b3"] = "b2"
    goal_thing_below = {"b1": "b0", "b2": "b1"}

    env = BlocksWorldEnv(show=True)
    env.load_blocks(thing_below)
    
    am = make_abstract_machine(env, num_blocks, max_levels)
    nvm = virtualize(am)
    
    # env.reset()
    # env.load_blocks(thing_below)

    goal_thing_above = env.invert(goal_thing_below)
    for key, val in goal_thing_above.items():
        if val == "none": goal_thing_above[key] = "nil"
    restore_env(nvm, num_blocks, max_levels, goal_thing_above)

    # # rin test
    # nvm.reset({
    #     "r0": nvm.registers["r0"].encode("b0"),
    #     "r1": nvm.registers["r1"].encode("nil"),
    #     "jnt": tr.tensor(am.ik["rest"])
    # })

    # block stacking test
    nvm.reset({
        "r0": nvm.registers["r0"].encode("b0"),
        "r1": nvm.registers["r1"].encode("b1"),
        "jnt": tr.tensor(am.ik["rest"])
    })

    nvm.mount("main")
    nvm.dbg()
    target_changed = True
    while True:
        # input('.')
        done = nvm.tick()
        nvm.dbg()
        
        if target_changed:
            position = nvm.registers["jnt"].content.detach().numpy()
            env.goto_position(position)

        tar = nvm.registers["tar"]
        target_changed = (tar.decode(tar.content) != tar.decode(tar.old_content))

        if done: break

    env.close()    

