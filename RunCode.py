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
import random


def main():

    # number of times the simulation is run, default is 1
    NumOfRun = 1

    # run conditions read in from terminal + offset noise
    N, conditions = func.initialise_simulation()
    nstep = 5000
    noise = 0.1

    omega, kappa, v0, algorithm = conditions
    phi0 = np.random.choice([0.5-noise, 0.5+noise], size=[N,N])

    for Run in range(NumOfRun):

        # runs simulation
        phi0_convg = func.diff_equation(N, conditions, phi0, nstep)

        np.savetxt(f'Data/{algorithm}Diffusion_Mat_{N}N_omega{omega}_kappa{kappa}_time{nstep}_v0{v0}.txt', phi0_convg)

main()