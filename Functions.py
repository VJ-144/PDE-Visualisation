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
from matplotlib.colors import ListedColormap


def Laplacian(phi0):
    """
    Calculates the laplacian of the 3D cube passed to it
    """

    # calculates laplacian Y and X componets for full matrix + center all correction
    yGrad_comp = np.roll(phi0, -1, axis =0) + np.roll(phi0, 1, axis=0)
    xGrad_comp = np.roll(phi0, -1, axis =1) + np.roll(phi0, 1, axis =1)
    center_corr = -4 * phi0

    # calculates laplacian
    laplacian = yGrad_comp + xGrad_comp + center_corr

    return laplacian


def nabla(phi0, dt):
    """
    Calculates the laplacian of the 3D cube passed to it
    """

    # calculates laplacian Y and X componets for full matrix + center all correction
    yGrad_comp = np.roll(phi0, -1, axis =0) - np.roll(phi0, 1, axis=0)
    xGrad_comp = np.roll(phi0, -1, axis =1) - np.roll(phi0, 1, axis =1)

    # calculates laplacian
    nabla_phi0 = (1/2*dt)*yGrad_comp + (1/2*dt)*xGrad_comp

    return nabla_phi0




def initialise_simulation():
    """
    Sets up all parameters for the simulation + reads arguments from terminal
    """

    # checks number of command line arguments are correct otherwise stops simulation
    if (len(sys.argv) <= 4 or len(sys.argv) > 5):
        print("Error! \nFile Input: python IVP_Run.py N Phi field")
        sys.exit()

    # reads arguments from terminal command line
    N=int(sys.argv[1]) 
    D=float(sys.argv[2])
    q=float(sys.argv[3])
    p=float(sys.argv[4])

    # # check if algorithm name from terminal  is valid
    # valid_algorithm = False
    # if (PDE=="jacobi" ): valid_algorithm = True
    # elif (PDE=="hilliard"): valid_algorithm = True
    # elif (PDE=="seidel"): valid_algorithm = True

    # # ends program if algorithm in invalid
    # if (valid_algorithm == False):
    #     print("Error! \nInvalid PDE Algorithm Parameter, choose from:\n1--hilliard\n2--jacobi\n3--seidel")
    #     sys.exit()


    # valid_field = False
    # if (field=="electric" ): valid_field = True
    # elif (field=="magnetic"): valid_field = True

    # # ends program if algorithm in invalid
    # if (field == False):
    #     print("Error! \nInvalid field Parameter, choose from:\n1--electric\n2--magnetic")
    #     sys.exit()


    conditions = (D, q, p)

    # returns all imporant simulation parameters
    return N, conditions



def PDE_converge(N, conditions, matrix):

    # simulation constants hard coded
    dx = 1
    dt = 0.1

    # extracting constant parameters
    D, q, p = conditions
    a, b, c = matrix

    tao = a + b + c 

    cmap = ListedColormap(['orange', 'red', 'yellow', 'dodgerblue'])
    vmin = -2
    vmax = 1

    # setting up animantion figure
    fig = plt.figure()
    im=plt.imshow(tao, animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    # number of sweeps and terminal display counter
    nstep=1000000
    sweeps = 0

    # data=open(f'Data/hilliard_{N}N_phi{phi_param}.txt','w')

    for n in range(nstep):

        # calculating laplacian with loop or np.roll
        laplacian_a = Laplacian(a)
        laplacian_b = Laplacian(b)
        laplacian_c = Laplacian(c)

        a_new = a + ((D*dt)/(dx**2)) * laplacian_a + dt*q*a*(1-a-b-c) - dt*p*a*c
        b_new = b + ((D*dt)/(dx**2)) * laplacian_b + dt*q*b*(1-a-b-c) - dt*p*a*b
        c_new = c + ((D*dt)/(dx**2)) * laplacian_c + dt*q*c*(1-a-b-c) - dt*p*b*c

        abc = a_new + b_new + c_new
        tao_new = Laplacian(abc)


        # tao_new = nabla(abc, dt)

        # print('tao')
        # print(tao_new)
        # print('a')
        # print(a_new)           

        a = a_new.copy()
        b = b_new.copy()
        c = c_new.copy()

        # visuals in set number of sweeps
        if(n%10==0): 

            # prints current number of sweep to terminal
            sweeps +=10
            print(f'sweeps={sweeps}', end='\r')

            # animates configuration 
            plt.cla()
            im=plt.imshow(tao_new, animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.draw()
            plt.pause(0.0001) 

            # calculating free energy density
            # energy_density = freeEnergy(phi0, N, conditions)

            # saving time and free energy density data
            # data.write('{0:5.5e} {1:5.5e}\n'.format(sweeps, energy_density))

    # data.close()

    return phi0
