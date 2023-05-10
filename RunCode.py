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


def inital_conditions(N):

    a = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
    b = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
    c = np.random.uniform(low=0.00, high=0.3333, size=[N,N])

    same=True

    if (same):

        a = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
        b = np.random.uniform(low=0.00, high=0.3333, size=[N,N])
        c = np.random.uniform(low=0.00, high=0.3333, size=[N,N])

        ab = np.allclose(a,b)
        ac = np.allclose(a,c)
        bc = np.allclose(b,c)
        

        if (ab==False and ac==False and bc==False): same=False

    return (a, b, c)

def main():

    # run conditions read in from terminal + offset noise
    N, conditions = func.initialise_simulation()

    # constant parameters
    D, q, p, Type = conditions

    converged_sweeps = []

    NumOfRun = 1
    for Run in range(NumOfRun):

        matrix = inital_conditions(N)

        # returns converged 3D matrix of point charge or charged wire and number of iteration for convergence
        
        sweeps, tao = func.PDE_converge(N, conditions, matrix)
        converged_sweeps.append(sweeps)
        np.savetxt(f'Data/Tao_Mat_{N}N_D{D}_q{q}_p{p}.txt', tao)

        print(f'Simulation Run {Run} Complete')

    np.savetxt('AbsorbtionTimeData.txt', converged_sweeps)

main()