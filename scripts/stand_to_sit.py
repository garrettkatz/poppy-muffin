from pypot.creatures import PoppyHumanoid as PH
import pickle as pkl

p = PH()

raw_input('[Enter] to go to zero angles (hold strap)')

for m in p.motors: m.compliant = False
p.goto_position({m.name: 0. for m in p.motors}, 5, wait=True)

raw_input('[Enter] to go to sit angles')

with open("sit_angles.pkl","r") as f: sit_angles = pkl.load(f)
p.goto_position(sit_angles, 5, wait=True)

print('closing...')
p.close()
print('closed')


