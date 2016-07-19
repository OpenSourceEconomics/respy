def record_simulation_progress(i):
    if (i != 0) and (i % 100 == 0):
        fmt_ = '  ... simulated {:>10} agents\n\n'
        with open('sim.respy.log', 'a') as outfile:
            outfile.write(fmt_.format(*[i]))


def record_simulation_start(num_agents_sim, seed_sim):
    line = ['Starting simulation of model for', num_agents_sim]
    line += ['agents with seed', seed_sim]
    with open('sim.respy.log', 'w') as outfile:
        fmt = '  {:>32} {:>8} {:>16} {:>8}\n\n'
        outfile.write(fmt.format(*line))


def record_simulation_stop():
    with open('sim.respy.log', 'a') as outfile:
        outfile.write('  ... finished\n\n')
