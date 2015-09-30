#!/usr/bin/env python
""" This module allows to simulate a dynamic programming model using the
terminal.
"""

# standard library
import pickle as pkl
import argparse
import sys
import os

# ROBUPY
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.simulate import simulate as robupy_simulate

def _distribute_inputs(parser):
    """ Process input arguments.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    num_agents = args.num_agents
    solution = args.solution
    seed = args.seed

    # Assertions.
    assert (isinstance(solution, str))
    assert (os.path.exists(solution))

    if num_agents is not None:
        assert (isinstance(num_agents, int))
        assert (num_agents > 0)

    if seed is not None:
        assert (isinstance(seed, int))

    # Finishing.
    return solution, num_agents, seed


def simulate(solution, num_agents, seed):
    """ Simulate the dynamic programming model.
    """
    # Solve model
    robupy_obj = pkl.load(open(solution, 'rb'))

    # Modifications
    with_modifications = (num_agents is not None) or (seed is not None)

    # Update attributes
    if with_modifications:
        robupy_obj.unlock()

        if num_agents is not None:
            robupy_obj.set_attr('num_agents', num_agents)

        if seed is not None:
            robupy_obj.set_attr('seed_simulation', seed)

        robupy_obj.lock()

    # Simulate the model
    robupy_simulate(robupy_obj)

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate dynamic discrete '
                                                 'choice model.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--solution', action='store', dest='solution',
                        default='solution.robupy.pkl',
                        help='solution object')

    parser.add_argument('--agents', type=int, default=None, dest='num_agents',
                        help='number of agents')

    parser.add_argument('--seed', type=int, default=None, dest='seed',
                        help='seed for simulation')

    solution, num_agents, seed = _distribute_inputs(parser)

    simulate(solution, num_agents, seed)