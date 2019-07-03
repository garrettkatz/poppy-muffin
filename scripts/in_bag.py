from pypot.creatures import PoppyHumanoid as PH

p = PH()

raw_input('Ready to turn off compliance before turning on?')

for m in p.motors: m.compliant = False

# raw_input('Ready to make legs/arms/abs/chest compliant?')

# for rl in 'rl':
#     for m in ['ankle', 'knee', 'hip']:
#         getattr(p,'%s_%s_y'%(rl,m)).compliant = True
# for m in p.arms: m.compliant = True
# p.abs_y.compliant = True
# p.bust_y.compliant = True

raw_input('Ready to turn on compliance to get to bag pose?')
for m in p.motors: m.compliant = True

raw_input('In bag pose and ready to turn off compliance again?')

for m in p.motors: m.compliant = False

raw_input('Compliance should now be off.')

p.close()

