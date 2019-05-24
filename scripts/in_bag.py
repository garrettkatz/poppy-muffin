from pypot.creatures import PoppyHumanoid as PH

p = PH()

raw_input('Ready to turn off compliance?')

for m in p.motors: m.compliant = False

raw_input('Ready to make legs/arms/abs/chest compliant?')

for rl in 'rl':
    for m in ['ankle', 'knee', 'hip']:
        getattr(p,'%s_%s_y'%(rl,m)).compliant = True
for m in p.arms: m.compliant = True
p.abs_y.compliant = True
p.bust_y.compliant = True

raw_input('Ready to turn off compliance?')

for m in p.motors: m.compliant = False

raw_input('Compliance should now be off.')

