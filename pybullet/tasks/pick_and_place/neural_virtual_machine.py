import torch as tr

def default_activator(v):
    tanh1 = tr.tanh(tr.tensor(1.))
    return tr.tanh(v) / tanh1

def fast_store_erase_rule(W, x, aσy):
    # W: (batch, x size, y size)
    # x, aσy: (batch, size, 1)
    return (aσy - W @ x) * x.transpose(1, 2) / float(x.shape[1])

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
        self.connections_to = {r: () for r in self.register_names}
        for c, (q, r) in connectivity.items():
            self.connections_to[r] += ((c, q),)
        
        # set up gate layer indexing
        self.recall_slice_for = {}
        recall_offset = 0
        for r in self.register_names:
            self.recall_slice_for[r] = slice(
                recall_offset,
                recall_offset + len(self.connections_to[r]))
            recall_offset += len(self.connections_to[r])
        self.storage_index = {c: recall_offset + i for i,c in enumerate(plastic_connections)}
        
        # set up state sequence buffers
        self.tick_counter = 0
        self.activities = {
            r: {0: tr.zeros(1,size,1)}
            for r,size in self.register_sizes.items()}
        self.weights = {
            c: {0: tr.zeros((1,self.register_sizes[r], self.register_sizes[q]))}
            for c, (q, r) in self.connectivity.items()}
    
    def batchify_weights(self, W):
        if len(W.shape) == 3: return W
        if len(W.shape) == 2: return W.unsqueeze(dim=0)
        raise ValueError("W should be 2 or 3 dimensional, not %d dimensional" % len(W.shape))

    def batchify_activities(self, v):
        if len(v.shape) == 3: return v
        if len(v.shape) == 1: return v.reshape(1, len(v), 1)
        raise ValueError("v should be 1 or 3 dimensional, not %d dimensional" % len(v.shape))
    
    def unpack(self, g):
        u = {r: g[:, self.recall_slice_for[r]] for r in self.register_names}
        ℓ = {c: g[:, [self.storage_index[c]]] for c in self.plastic_connections}
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
            if t in W_in[c]: W[c][t] = self.batchify_weights(W_in[c][t])
        for r in v_in:
            if t in v_in[r]: v[r][t] = self.batchify_activities(v_in[r][t])

        # unpack gate values
        u, ℓ = self.unpack(v[self.gate_register_name][t])

        # recall
        for r in self.register_names:
            v[r][t+1] = v[r][t]
            if len(self.connections_to[r]) > 0:
                Wv = [W[c][t] @ v[q][t] for c, q in self.connections_to[r]]
                v[r][t+1] = σ[r](tr.cat(Wv, dim=2) @ u[r])

        # storage
        for c, (q, r) in self.connectivity.items():
            W[c][t+1] = W[c][t]
            if c in self.plastic_connections:
                W[c][t+1] = W[c][t] + ℓ[c] * H(W[c][t], v[q][t], v[r][t])
        
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
    plastic_connections = [] #["r0>r1"]

    # W0 = tr.ones((1,register_sizes["r1"], register_sizes["r0"]), requires_grad=True)
    # v0 = tr.ones((1,register_sizes["r0"],1))
    W0 = tr.ones((register_sizes["r1"], register_sizes["r0"]), requires_grad=True)
    v0 = tr.ones((register_sizes["r0"],))

    nvm = NeuralVirtualMachine(
        register_sizes, activators, gate_register_name,
        connectivity, plastic_connections)
    
    W, v = nvm.run(v_in = {"r0": {0: v0, 1: v0}}, W_in = {"r0>r1": {0: W0}}, num_time_steps=2)
    print(nvm.tick_counter)
    print(W)
    print(v)
