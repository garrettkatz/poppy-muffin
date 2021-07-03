import random

class BlockStackingProblem:
    def __init__(self, domain, thing_below, goal_thing_below, goal_thing_above):
        self.domain = domain
        self.thing_below = thing_below
        self.goal_thing_below = goal_thing_below
        self.goal_thing_above = goal_thing_above

    def blocks(self):
        return self.domain.blocks
    def bases(self):
        return self.domain.bases

    def base_and_level_of(self, block):
        # blocks starting from level 0, does not accept bases
        thing = self.thing_below[block]
        if thing in self.domain.bases: return thing, 0
        base, level = self.base_and_level_of(thing)
        return base, level + 1

class BlockStackingDomain:
    def __init__(self, num_blocks, num_bases, max_levels):
        # no towers have more than max_levels
        self.num_blocks = num_blocks
        self.num_bases = num_bases
        self.max_levels = max_levels

        self.blocks = ["b%d" % b for b in range(self.num_blocks)]
        self.bases = ["t%d" % t for t in range(self.num_bases)]
    
    def invert(self, thing_below):
        block_above = {thing: "nil" for thing in self.blocks + self.bases}
        for block, thing in thing_below.items(): block_above[thing] = block
        return block_above
    
    def symbolic_distance(self, thing_below, goal_thing_below):
        distance = 0
        for block in self.blocks:
            actual_below = thing_below[block]
            target_below = goal_thing_below[block]
            if actual_below == target_below: continue
            if actual_below in self.bases and target_below in self.bases: continue
            distance += 1
        return distance

    def random_thing_below(self):
        # make a random thing_below dictionary

        # towers = [["t%d" % n] for n in range(self.num_bases)] # singleton towers too likely
        towers = [["t%d" % n] for n in random.sample(range(self.num_bases), self.num_blocks)]
        for n in range(self.num_blocks):
            short_towers = list(filter(lambda x: len(x[1:]) < self.max_levels, towers)) # exclude bases
            tower = random.choice(short_towers)
            tower.append("b%d" % n)
        thing_below = {}
        for tower in towers:
            for level in range(len(tower)-1):
                thing_below[tower[level+1]] = tower[level]
        return thing_below

    def random_problem_instance(self):
    
        # rejection sample non-trivial instance
        while True:
            thing_below = self.random_thing_below()
            goal_thing_below = self.random_thing_below()
            if self.symbolic_distance(thing_below, goal_thing_below) > 0: break
        
        goal_thing_above = self.invert(goal_thing_below)
        return BlockStackingProblem(self, thing_below, goal_thing_below, goal_thing_above)

if __name__ == "__main__":
    
    prob = BlockStackingDomain(3, 3, 3).random_problem_instance()
    print(prob.goal_thing_above)

