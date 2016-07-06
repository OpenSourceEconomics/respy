def record_simulation_progress(i):
    if (i != 0) and (i % 100 == 0):
        fmt_ = '  ... simulated {:>10} agents\n\n'
        with open('sim.respy.log', 'a') as outfile:
            outfile.write(fmt_.format(*[i]))
