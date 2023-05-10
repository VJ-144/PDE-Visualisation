"""
This file is used to run the the checkpoint.

This file must be run through terminal with the input arguments as follows:

'python IVP_Run_new.py N phi algorithm field'

A full explanation of the parameters can be found in the README.txt file
"""


import Functions as func
import numpy as np
import time
import sys


def main():

    # number of times the simulation is run, default is 1
    NumOfRun = 1

    # run conditions read in from terminal + offset noise
    N, conditions = func.initialise_simulation()
    offset = 0.1

    # constant parameters
    D, q, p = conditions

    a = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
    b = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
    c = np.random.uniform(low=0.00, high=0.3333, size=[N,N])

    matrix = (a, b, c)

    for Run in range(NumOfRun):


        # returns converged 3D matrix of point charge or charged wire and number of iteration for convergence
        phi_converged, interation = func.PDE_converge(N, conditions, matrix)


main()