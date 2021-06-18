from nvm_am_compare import run_trial

while True:
    print("trying..")
    num_bases, num_blocks, max_levels = 5, 5, 3
    result = run_trial(num_bases, num_blocks, max_levels)
    am_results, nvm_results, nvm_size, thing_below, goal_thing_below = result
    ticks, running_time, sym_reward, spa_reward = am_results
    
    if sym_reward <= -2: break

print(thing_below)
print(goal_thing_below)

