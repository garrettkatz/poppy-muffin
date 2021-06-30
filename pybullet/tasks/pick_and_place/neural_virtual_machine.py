import torch as tr

def default_activator(v):
    tanh1 = tr.tanh(tr.tensor(1.))
    return tr.tanh(v) / tanh1

def fast_store_erase_rule(W, x, y):
    # W: (batch, x size, y size)
    # x, y: (batch, size, 1)
    return (y - W @ x) * x.transpose(1, 2) / float(x.shape[1])

class NeuralVirtualMachine:
    def __init__(self,
        register_sizes, gate_register_name,
        connectivity, activators=None, plastic_connections=None):

        # Defaults
        if activators is None: activators = {}
        if plastic_connections is None: plastic_connections = []
        
        # set up registers
        self.gate_register_name = gate_register_name
        self.register_sizes = dict(register_sizes)
        self.register_sizes[gate_register_name] = len(connectivity) + len(plastic_connections)
        self.register_names = tuple(sorted(self.register_sizes.keys()))
        
        # set up activator functions
        self.activators = {
            r: activators.get(r, default_activator)
            for r in self.register_names}
        
        # set up connectivity
        self.connectivity = dict(connectivity)
        self.plastic_connections = tuple(sorted(plastic_connections))
        self.connections_to = {r: () for r in self.register_names}
        for c, (q, r) in connectivity.items():
            self.connections_to[r] += ((c, q),)
        for r in self.register_names:
            self.connections_to[r] = tuple(sorted(self.connections_to[r]))
        
        # set up gate layer indexing
        self.recall_index = {}
        for r in self.register_names:
            for c, q in self.connections_to[r]:
                self.recall_index[c] = len(self.recall_index)
        self.storage_index = {}
        for i, c in enumerate(plastic_connections):
            self.storage_index[c] = len(self.recall_index) + i

        # set up recall gate slicing
        self.recall_slice = {}
        stop = 0
        for r in self.register_names:
            start = stop
            stop = start + len(self.connections_to[r])
            self.recall_slice[r] = slice(start, stop)
        
        # initialize tick counter and state buffers
        self.clear_ticks()

    def clear_ticks(self):
        self.tick_counter = 0
        self.activities = {
            r: {0: tr.zeros(1, size, 1)}
            for r,size in self.register_sizes.items()}
        self.weights = {
            c: {0: tr.zeros((1, self.register_sizes[r], self.register_sizes[q]))}
            for c, (q, r) in self.connectivity.items()}
    
    def batchify_weights(self, W):
        if len(W.shape) == 3: return W
        if len(W.shape) == 2: return W.unsqueeze(dim=0)
        raise ValueError("W should be 2 or 3 dimensional, not %d dimensional" % len(W.shape))

    def batchify_activities(self, v):
        if len(v.shape) == 3: return v
        if len(v.shape) == 2: return v.unsqueeze(dim=2)
        if len(v.shape) == 1: return v.reshape(1, len(v), 1)
        raise ValueError("v should be 1, 2, or 3 dimensional, not %d dimensional" % len(v.shape))
    
    def unpack(self, g):
        u = {r: g[:, self.recall_slice[r]] for r in self.register_names}
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

def test_batching():
    register_sizes = {"r": 3}
    nvm = NeuralVirtualMachine(register_sizes,
        gate_register_name="g",
        connectivity={"r>g": ("r", "g")},
        plastic_connections=["r>g"])

    assert nvm.activities["g"][0].numel() == 2 # one connection x2 gates
    assert nvm.activities["g"][0].shape == (1,2,1)

    # no batching
    r0 = tr.ones(3)
    W0 = tr.ones((2,3))
    W, v = nvm.run(v_in = {"r": {0: r0}}, W_in = {"r>g": {0: W0}}, num_time_steps=1)
    assert v["r"][1].shape == (1,3,1)
    assert v["g"][1].shape == (1,2,1)
    assert W["r>g"][1].shape == (1,2,3)

    # batching v
    r0 = tr.ones(2,3,1)
    W0 = tr.ones((2,3))
    nvm.clear_ticks()
    W, v = nvm.run(v_in = {"r": {0: r0}}, W_in = {"r>g": {0: W0}}, num_time_steps=1)
    assert v["r"][1].shape == (2,3,1)
    assert v["g"][1].shape == (2,2,1)
    assert W["r>g"][1].shape == (2,2,3)

    # batching W
    r0 = tr.ones(3)
    W0 = tr.ones((4,2,3))
    nvm.clear_ticks()
    W, v = nvm.run(v_in = {"r": {0: r0}}, W_in = {"r>g": {0: W0}}, num_time_steps=1)
    assert v["r"][1].shape == (1,3,1)
    assert v["g"][1].shape == (4,2,1)
    assert W["r>g"][1].shape == (4,2,3)

    # batching v without plastic
    nvm = NeuralVirtualMachine(register_sizes,
        gate_register_name="g",
        connectivity={"r>g": ("r", "g")},
        plastic_connections=[])
    r0 = tr.ones(2,3,1)
    W0 = tr.ones((1,2,3))
    W, v = nvm.run(v_in = {"r": {0: r0}}, W_in = {"r>g": {0: W0}}, num_time_steps=1)
    assert v["r"][1].shape == (2,3,1)
    assert v["g"][1].shape == (2,2,1)
    assert W["r>g"][1].shape == (1,2,3)

    # batching v without plastic for multiple time-steps
    nvm.clear_ticks()
    W, v = nvm.run(v_in = {"r": {0: r0}}, W_in = {"r>g": {0: W0}}, num_time_steps=3)
    assert v["r"][3].shape == (2,3,1)
    assert v["g"][3].shape == (2,2,1)
    assert W["r>g"][3].shape == (1,2,3)

def test_dynamics():
    register_sizes = {"r": 3}
    nvm = NeuralVirtualMachine(register_sizes,
        gate_register_name="g",
        connectivity={"r>g": ("r", "g")},
        plastic_connections=["r>g"])
    r0 = tr.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 0]]).float()
    g0 = tr.tensor([1, 0]).float()
    W0 = tr.tensor([[1, 0, 0], [0, 1, 0]]).float()
    W, v = nvm.run(
        W_in = {"r>g": {0: W0}},
        v_in = {"r": {0: r0}, "g": {0: g0}},
        num_time_steps=2)
    assert tr.allclose(v["g"][1], nvm.batchify_activities(
        tr.tensor([[1, 1], [0, 1], [0, 0]]).float()))
    assert tr.allclose(v["g"][2], nvm.batchify_activities(
        tr.tensor([[1, 1], [0, 0], [0, 0]]).float()))
    assert tr.allclose(W["r>g"][1], nvm.batchify_activities(tr.tensor([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0]],
        ]).float()))
    # y = g, x = r, so y - Wx = 0
    assert tr.allclose(W["r>g"][2], nvm.batchify_activities(tr.tensor([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0]],
        ]).float()))

    register_sizes = {"q": 2, "r": 2}
    nvm = NeuralVirtualMachine(register_sizes,
        gate_register_name="g",
        connectivity={"q>r": ("q", "r"), "q>g": ("q", "g")},
        plastic_connections=["q>r"])
    W, v = nvm.run(
        W_in = {
            "q>r": {0: tr.tensor([[1, -1],[1, -1]]).float()},
        },
        v_in = {
            "q": {0: tr.tensor([1,  1]).float()},
            "r": {0: tr.tensor([1, -1]).float()},
            "g": {0: tr.tensor([[0, 0, 1], [0, 0, 0]]).float()},
        },
        num_time_steps=1)
    assert tr.allclose(W["q>r"][1], nvm.batchify_activities(tr.tensor([
        [[1.5, -.5], [.5, -1.5]],
        [[1, -1], [1, -1]],
        ]).float()))

if __name__ == "__main__":
    
    test_batching()
    test_dynamics()

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
        register_sizes, gate_register_name, connectivity,
        activators, plastic_connections)
    
    W, v = nvm.run(W_in = {"r0>r1": {0: W0}}, v_in = {"r0": {0: v0, 1: v0}}, num_time_steps=2)
    print(nvm.tick_counter)
    print(W)
    print(v)
