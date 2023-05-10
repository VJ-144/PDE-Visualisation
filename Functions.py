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
    omega=float(sys.argv[2])
    kappa=float(sys.argv[3])


    # # ends program if incorrect algorithm is input
    # valid_algorithm = True
    # if (algorithm=='standard'): valid_algorithm=False
    # elif (algorithm=='velocity'): valid_algorithm=False
    # if (valid_algorithm):
    #     print('Error: invalid algorithm input')
    #     sys.exit()

    conditions = (omega, kappa, v0, algorithm)

    # returns all imporant simulation parameters
    return N, conditions



def source_matrix(omega, N):

    # initial positiona/source matrix
    rho = np.zeros(shape=(N,N), dtype=float)

    half_size = int(N/2)

    # calculates all radius and all source matrix elements
    for ij in itertools.product(range(N), repeat=2):

        i,j =ij

        # calculate radius vector coordinates
        xx = np.abs(i-half_size)
        yy = np.abs(j-half_size)

        # calcuating radius magnitude
        Radius = np.sqrt(xx**2 + yy**2)

        rho[i,j] = np.exp( -(Radius**2)/(omega**2) )

    return rho

def vel_dependence(v0, N, dx):

    vel_mat = np.zeros(shape=(N,N), dtype=float)

    half_size = int(N/2)

    # calculates all contributions of velocity adjustment to algorithm
    for ij in itertools.product(range(N), repeat=2):

        i,j = ij

        factor = -v0 * np.sin((2*np.pi*i)/N)/2*dx

        vel_mat[i,j] = factor

    return vel_mat

def diff_equation(N, conditions, phi0, nstep):

    # extracting constant parameters
    omega, kappa, v0, algorithm = conditions

    # they must converge based on the condition: dt/dx < 1/2
    dt=0.2
    dx=1
    D=1

    # setting up animantion figure
    fig = plt.figure()
    im=plt.imshow(phi0, animated=True)

    # number of sweeps and terminal display counter
    sweeps = 0

    # calculate position vector squared
    rho = source_matrix(omega, N)

    data=open(f'Data/{algorithm}Diffusion_Evol_{N}N_omega{omega}_kappa{kappa}_time{nstep}_v0{v0}.txt','w')

    # calculates contribution to algorithm for velocity dependence
    algorithm_factor = 0
    if (algorithm=='velocity'): vel = vel_dependence(v0, N, dx)

    for n in range(nstep):

        # calculating laplacian with np.roll
        laplacian_phi0 = Laplacian(phi0)

        # calculating new factor based on velocity algorithm
        if (algorithm=='velocity'): algorithm_factor = vel * (np.roll(phi0, -1, axis=1) - np.roll(phi0, 1, axis=1))

        phi0_new = phi0 + ((dt*D)/dx**2) * laplacian_phi0 + (dt * rho) - (dt * kappa * phi0) - dt*algorithm_factor

        # feeding phi back into algorithm
        phi0 = phi0_new.copy()

        # visuals in set number of sweeps
        if(n%10==0): 

            # prints current number of sweep to terminal
            sweeps += 10
            print(f'sweeps={sweeps}', end='\r')

            # animates configuration 
            plt.cla()
            im=plt.imshow(phi0, animated=True)
            plt.draw()
            plt.pause(0.0001) 

            # calculating average
            phi0_avg = np.mean(phi0)

            # saving time and free energy density data
            data.write('{0:5.5e} {1:5.5e}\n'.format(sweeps, phi0_avg))

    data.close()

    return phi0
