"""
This file is used to run the ising model simulation for a single temperature and
for batches of temperatures which reuse spin configurations from previous calculations.
This file require the existence of the /Data/Glauber and /Data/Kawasaki directories in the same directory
to store calculated data and run error analysis. The directory names are spelling and case sensitive.

This file must be run through terminal with the input arguments as follows:

'python run.ising.simulation.py N kT, model, BatchRun'

A full explanation of the parameters can be found in the README.txt file
"""


import IVP_Functions as ivp
import numpy as np
import time
import sys


def main():

    NumOfRun = 1

    # run conditions
    N, phi, conditions, PDE = ivp.initialise_simulation()
    offset = 0.4

    a, k, M, dx, dt, phi, field = conditions

    for Run in range(NumOfRun):

        # generating initial lattice
        phi0 = np.random.uniform(low=phi-offset, high=phi+offset, size=[N,N])


        if (PDE=='hilliard'): ivp.cahn_hilliard(N, conditions, phi0)

        if (PDE=='jacobi'): ivp.jacobi(N, conditions, PDE)

        if (PDE=='seidel'): 

            data=open(f'Data/seidel_{field}_{N}N_phi{phi}.txt','w')
            omega_range = np.linspace(0.1, 3, 32) 

            for i, w in enumerate(omega_range):

                interation = ivp.gauss_seidel(N, conditions, PDE, w)
                data.write('{0:5.5e} {1:5.5e}\n'.format(iteration, w))
                print(f'simulation $\omega$={w} at iteration = {iteration}')

            data.close()


main()