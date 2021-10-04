import sys
sys.path.append('../../envs')    
import block_stacking_problem as bp
from blocks_world import BlocksWorldEnv
from abstract_machine import setup_abstract_machine

def proc(comp):
    comp.ret_if_nil()
    comp.put("b0", "r0")
    comp.ret()

def main(comp):
    comp.call("proc")

if __name__ == "__main__":

    max_levels = 3
    num_blocks = 5
    num_bases = 5
    domain = bp.BlockStackingDomain(num_blocks, num_bases, max_levels)
    env = BlocksWorldEnv(show=False)
    am, compiler = setup_abstract_machine(env, domain, gen_regs=["r0"])

    compiler.flash(proc)
    compiler.flash(main)

    code = am.machine_code()
    ipt, asm, mach, store, recall = zip(*code)
    store = [", ".join(conn) for conn in store]
    recall = [", ".join(conn) for conn in recall]

    width = [
        max(map(len, map(str, column)))
        for column in (ipt, asm, mach, store, recall)
    ]
    width[0] = max(width[0], len("ipt"))
    width[3] = max(width[3], len("store"))
    line = "ipt: assembly" + " "*(width[1] - len("assembly")) + " | "
    line += "store" + " "*(width[3] - len("store")) + " | "
    line += "recall" + " "*(width[4] - len("recall"))
    print(line)
    for c in range(len(code)):
        line = ""
        line += ("%" + str(width[0]) + "d: ") % ipt[c]
        line += asm[c] + " "*(width[1] - len(asm[c])) + " | "
        # line += mach[c] + " "*(width[2] - len(mach[c])) + " | "
        line += store[c] + " "*(width[3] - len(store[c])) + " | "
        line += recall[c] + " "*(width[4] - len(recall[c]))
        print(line)

    print(am.connections["ipt"].memory)

    am.reset({
        # "jmp": "t0",
        "jmp": "nil",
    })    
    num_ticks = am.run(dbg=True)
    # num_ticks = am.run(dbg=False)
    print(num_ticks)

