"""
This file contains all the functions required to run checkpoint 3. It contains the functions for the hilliard 
visualisation and the convergence of a charged cube (point charge or charged wire) using teh jacobi algorithm.
It also contains testing for the convergence of the optimal omega value

The functions require the existence of the /Data/ and /Plots/ directories to store calculated 
data and run error analysis. The directory names are spelling and case sensitive.
"""

import datetime
import os
import sys
import math
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Laplacian(phi0):
    """
    Calculates the laplacian of the 3D cube passed to it
    """

    # calculates laplacian Y and X componets for full matrix + center all correction
    yGrad_comp = np.roll(phi0, -1, axis=0) + np.roll(phi0, 1, axis=0)
    xGrad_comp = np.roll(phi0, -1, axis=1) + np.roll(phi0, 1, axis=1)
    center_corr = -4 * phi0

    # calculates laplacian
    laplacian = yGrad_comp + xGrad_comp + center_corr

    return laplacian



def initialise_simulation():
    """
    Sets up all parameters for the simulation + reads arguments from terminal
    """

    # # checks number of command line arguments are correct otherwise stops simulation
    # if (len(sys.argv) < 5 or len(sys.argv) > 6):
    #     print("Error! \nFile Input: python RunCode.py N omega kappa v0 Algorithm")
    #     sys.exit()

    # reads arguments from terminal command line
    N=int(sys.argv[1]) 
    phi0=float(sys.argv[2])
    chi=float(sys.argv[3])
    algorithm = str(sys.argv[4])
    alpha = float(sys.argv[5])
    Batch = str(sys.argv[6])

    # ends program if incorrect algorithm is input
    valid_algorithm = True
    if (algorithm=='standard'): valid_algorithm=False
    elif (algorithm=='reaction'): valid_algorithm=False
    if (valid_algorithm):
        print('Error: invalid algorithm input')
        sys.exit()

    conditions = (phi0, chi, algorithm, alpha, Batch)

    # returns all imporant simulation parameters
    return N, conditions



def diff_equation(N, conditions, phi, m, nstep):

    # extracting constant parameters
    phi0, chi, algorithm, alpha, Batch = conditions

    # they must converge based on the condition: dt/dx < 1/2
    dt=0.2
    dx=1
    a = 0.1
    c = 0.1
    kappa = 0.1
    M = 0.1
    D = 1
    phi_bar = 0.5

    # setting up animantion figure
    fig = plt.figure()

    im=plt.imshow(m, animated=True)

    # number of sweeps and terminal display counter
    sweeps = 0

    
    data=open(f'Data/{algorithm}/{algorithm}_Evol_{N}N_phi{phi0}_chi_{chi}_Time{nstep}_alpha{alpha}.txt','w')
    
    algorithm_factor = 0

    for n in range(nstep):

        # calculating laplacian with np.roll
        laplacian_phi = Laplacian(phi)

        # calculating chemical potential matrix
        chem_pot = -a * phi + a * phi**3 - (chi/2) * m**2 - kappa * laplacian_phi

        laplacian_chem = Laplacian(chem_pot)

        if (algorithm=='reaction'): algorithm_factor = alpha * (phi - phi_bar)

        phi_new = phi + ((dt*M)/dx**2) * laplacian_chem - algorithm_factor*dt

        laplacian_m = Laplacian(m)

        m_new = m + ((dt*D)/dx**2) * laplacian_m - dt*((c-chi*phi)*m + c*m**3)

        # feeding phi and m back into algorithm
        phi = phi_new.copy()
        m = m_new.copy()

        # visuals in set number of sweeps
        if(n%10==0 and n>=500): 

            # prints current number of sweep to terminal
            sweeps += 10
            print(f'sweeps={sweeps}', end='\r')

            # animates configuration 
            plt.cla()
            im=plt.imshow(m, animated=True)
            plt.draw()
            plt.pause(0.0001) 

            # calculating average
            phi_avg = np.mean(phi)
            m_avg = np.mean(m)

            data.write('{0:5.5e} {1:5.5e} {2:5.5e}\n'.format(sweeps, phi_avg, m_avg))

    data.close()

    return phi, m
