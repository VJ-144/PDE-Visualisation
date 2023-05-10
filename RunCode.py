"""
This file is used to run the the checkpoint.

This file must be run through terminal with the input arguments as follows:

'python IVP_Run_new.py N phi algorithm field'

A full explanation of the parameters can be found in the README.txt file
"""


import IVP_Functions_new as ivp
import numpy as np
import time
import sys


def main():

    # number of times the simulation is run, default is 1
    NumOfRun = 1

    # run conditions read in from terminal + offset noise
    N, phi, conditions, PDE = ivp.initialise_simulation()
    offset = 0.1

    # constant parameters
    a, k, M, dx, dt, phi, field = conditions

    for Run in range(NumOfRun):

        # generating initial cube with noise for hilliard
        phi0 = np.random.uniform(low=phi-offset, high=phi+offset, size=[N,N])

        # runs hilliard simulation
        if (PDE=='hilliard'): ivp.cahn_hilliard(N, conditions, phi0)

        # runs jacobi simulation for point charge or charged wire
        if (PDE=='jacobi'): 
            
            # returns converged 3D matrix of point charge or charged wire and number of iteration for convergence
            phi_converged, interation = ivp.jacobi_converge(N, conditions)

            # counts number of 2D slices saved to file
            count = 0

            # saves converged 3D charged matrix to file by saving 2D slices
            with open(f'Data/Jacobi_{field}Field_{N}N_phi{phi}.txt', 'w') as outfile:
                for data_slice in phi_converged:

                    np.savetxt(outfile, data_slice, fmt='%-7.6f')
                    outfile.write(f'# New slice {count}\n')
                    count += 1

        # runs the seidel simulation to test omega convergence
        if (PDE=='seidel'): 

            # creates data file to write to
            data=open(f'Data/seidel_{field}_{N}N_phi{phi}.txt','w')

            # tests omega values 1.75 -> 1.91 in increments of 0.01
            omega_range = np.arange(1.75, 1.91, 0.01)

            # iterates over different omega values
            for i, w in enumerate(omega_range):

                # caculates converged phi and number of iterations used seidel method
                phi_converged, iteration = ivp.seidel_converge(N, conditions, w)

                # saves number of iterations to convergence and omega value used to file
                data.write('{0:5.5e} {1:5.5e}\n'.format(iteration, w))
                print(f'\nsimulation omega={w} at iteration = {iteration}')

            data.close()

main()