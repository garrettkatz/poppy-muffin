from pypot.creatures import PoppyHumanoid as PH
import pickle as pkl

p = PH()

raw_input('Ready to turn off compliance and lift out of bag?')

for m in p.motors: m.compliant = False

raw_input('Ready to make legs compliant and put into sit position?')

for rl in 'rl':
    for m in ['ankle', 'knee', 'hip']:
        getattr(p,'%s_%s_y'%(rl,m)).compliant = True

raw_input('Ready to make hips/abs/chest/arms compliant and put to sit position?')

for rl in 'rl':
    for m in ['ankle', 'knee']:
        getattr(p,'%s_%s_y'%(rl,m)).compliant = False

for m in p.arms: m.compliant = True
p.abs_y.compliant = True
p.bust_y.compliant = True

raw_input('Ready to sit?')

with open("sit_angles.pkl","r") as f: sit_angles = pkl.load(f)

for m in p.motors: m.compliant = False
p.goto_position(sit_angles, 5, wait=True)

p.close()

