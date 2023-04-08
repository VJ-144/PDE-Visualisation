"""
This file is used to run the ising model simulation for a single temperature and
for batches of temperatures which reuse spin configurations from previous calculations.
This file require the existence of the /Data/Glauber and /Data/Kawasaki directories in the same directory
to store calculated data and run error analysis. The directory names are spelling and case sensitive.

This file must be run through terminal with the input arguments as follows:

'python run.ising.simulation.py N kT, model, BatchRun'

A full explanation of the parameters can be found in the README.txt file
"""


import IVP_Functions_new as ivp
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

        if (PDE=='jacobi'): 
            
            phi_converged, interation = ivp.charged_cube(N, conditions, PDE)

            # counts number of 2D slices saved to file
            count = 0

            # saves converged 3D charged matrix to file by saving 2D slices
            with open(f'Data/Jacobi_{field}Field_{N}N_phi{phi}.txt', 'w') as outfile:
                for data_slice in phi_converged:

                    np.savetxt(outfile, data_slice, fmt='%-7.6f')
                    outfile.write(f'# New slice {count}\n')
                    count += 1

        if (PDE=='seidel'): 

            data=open(f'Data/seidel_{field}_{N}N_phi{phi}.txt','w')
            omega_range = np.linspace(0.1, 3, 30) 
            print(omega_range)
            for i, w in enumerate(omega_range):

                phi_converged, interation = ivp.charged_cube(N, conditions, PDE, w)
                data.write('{0:5.5e} {1:5.5e}\n'.format(iteration, w))
                print(f'simulation $\omega$={w} at iteration = {iteration}')

            data.close()


main()