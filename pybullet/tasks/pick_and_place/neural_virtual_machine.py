import torch as tr

def default_activator(v):
    tanh1 = tr.tanh(tr.tensor(1.))
    return tr.tanh(v) / tanh1

def fast_store_erase_rule(W, x, aσy):
    x, aσy = x.reshape(-1,1), aσy.reshape(-1,1)
    dW = (aσy - W.mm(x)) * x.t() / float(W.shape[1])
    return dW

class NeuralVirtualMachine:
    def __init__(self,
        register_sizes, activators, gate_register_name,
        connectivity, plastic_connections=None):
        
        # set up registers
        self.gate_register_name = gate_register_name
        self.register_sizes = dict(register_sizes)
        self.register_sizes[gate_register_name] = len(connectivity) + len(plastic_connections)
        self.register_names = tuple(self.register_sizes.keys())
        
        # set up activator functions
        self.activators = {
            r: activators.get(r, default_activator)
            for r in self.register_names}
        
        # set up connectivity
        self.connectivity = dict(connectivity)
        self.plastic_connections = tuple(plastic_connections)
        self.incoming_connections_to = {r: () for r in self.register_names}
        for c, (q, r) in connectivity.items():
            self.incoming_connections_to[r] += ((c, q),)
        
        # set up gate layer indexing
        self.storage_index = {c:i for i,c in enumerate(plastic_connections)}
        self.recall_index = {}
        for r in self.register_names:
            for c, q in self.incoming_connections_to[r]:
                self.recall_index[c] = len(self.recall_index) + len(plastic_connections)
        
        # set up state sequence buffers
        self.tick_counter = 0
        self.activities = {
            r: {0: tr.zeros(size)}
            for r,size in self.register_sizes.items()}
        self.weights = {
            c: {0: tr.zeros((self.register_sizes[r], self.register_sizes[q]))}
            for c, (q, r) in self.connectivity.items()}
    
    def unpack(self, g):
        u = {c: g[self.recall_index[c]] for c in self.connectivity} # g[:,idx] for batch mode?
        ℓ = {c: g[self.storage_index[c]] for c in self.plastic_connections}
        return u, ℓ

    def tick(self, W_in={}, v_in={}):
        # setup variables
        t = self.tick_counter
        σ = self.activators
        v = self.activities
        W = self.weights
        H = fast_store_erase_rule
        
        # set inputs
        for c in W_in:
            if t in W_in[c]: W[c][t] = W_in[c][t]
        for r in v_in:
            if t in v_in[r]: v[r][t] = v_in[r][t]

        # unpack gate values
        u, ℓ = self.unpack(v[self.gate_register_name][t])

        # recall
        for r in self.register_names:
            v[r][t+1] = tr.zeros_like(v[r][t])
            for c, q in self.incoming_connections_to[r]:
                v[r][t+1] += u[c] * W[c][t].mv(v[q][t])
            v[r][t+1] = σ[r](v[r][t+1])
        
        for r in self.register_names:
            v[r][t+1] = σ[r](
                u[r].mv(tr.stack([W[c][t].mv(v[q][t]) for c, q in self.incoming_connections_to[r]])))

        # storage
        for c, (q, r) in self.connectivity.items():
            if c in self.plastic_connections:
                W[c][t+1] = W[c][t] + ℓ[c] * H(W[c][t], v[q][t], v[r][t])
            else:
                W[c][t+1] = W[c][t]
        
        # update counter
        self.tick_counter += 1

    def run(self, W_in={}, v_in={}, num_time_steps=1):
        for step in range(num_time_steps): self.tick(W_in, v_in)
        return self.weights, self.activities

if __name__ == "__main__":
    
    register_sizes = {"r0": 4, "r1": 4, "ipt": 8}
    activators = {"gts": lambda v: v}
    gate_register_name = "gts"
    connectivity = {
        "r0>r1": ("r0", "r1"),
        "ipt": ("ipt", "ipt"),
        "gts": ("ipt", "gts")}
    plastic_connections = ["r0>r1"]

    W0 = tr.ones((register_sizes["r1"], register_sizes["r0"]), requires_grad=True)
    v0 = tr.ones((register_sizes["r0"],))

    nvm = NeuralVirtualMachine(
        register_sizes, activators, gate_register_name,
        connectivity, plastic_connections)
    W, v = nvm.run(v_in = {"r0": {0: v0, 1: v0}}, W_in = {"r0>r1": {0: W0}}, num_time_steps=2)
    print(nvm.tick_counter)
    print(W)
    print(v)
