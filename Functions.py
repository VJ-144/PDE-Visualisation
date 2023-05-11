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

def correlation_prob(tao, N, correlation_1, arrayLength):

    # correlation_1 = np.zeros(shape=(int(N/2)))
    # arrayLength = np.zeros(shape=(int(N/2)))

    radius_list = []

    center = int(N/2)

    for ij in itertools.product(range(int(N/2)), repeat=2):

        i,j = ij

        cell_rowCenter = tao[i, center]

        cell1 = tao[i,j]
        
        xdiff = np.abs(center - j)
        
        radius_list.append(xdiff)

        if (cell1==cell_rowCenter):
            correlation_1[xdiff] +=1

        arrayLength[xdiff] +=1

    prob = correlation_1/arrayLength

    return prob


def Laplacian(phi0):
    """
    Calculates the laplacian of the 3D cube passed to it
    """

    # calculates laplacian Y and X componets for full matrix + center all correction
    yGrad_comp = np.roll(phi0, -1, axis =0) + np.roll(phi0, 1, axis =0)
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
    yGrad_comp = np.roll(phi0, -1, axis =0) - np.roll(phi0, 1, axis =0)
    xGrad_comp = np.roll(phi0, -1, axis =1) - np.roll(phi0, 1, axis =1)

    # calculates laplacian
    nabla_phi0 = (1/2*dt)*yGrad_comp + (1/2*dt)*xGrad_comp

    return nabla_phi0




def initialise_simulation():
    """
    Sets up all parameters for the simulation + reads arguments from terminal
    """

    # checks number of command line arguments are correct otherwise stops simulation
    # if (len(sys.argv) <= 4 or len(sys.argv) > 5):
    #     print("Error! \nFile Input: python IVP_Run.py N Phi field")
    #     sys.exit()

    # reads arguments from terminal command line
    N=int(sys.argv[1]) 
    D=float(sys.argv[2])
    q=float(sys.argv[3])
    p=float(sys.argv[4])
    Type=str(sys.argv[5])

    # check if algorithm name from terminal  is valid
    valid_algorithm = False
    if (Type=="standard" ): valid_algorithm = True
    elif (Type=="absorbing"): valid_algorithm = True
    elif (Type=="points"): valid_algorithm = True
    elif (Type=="prob"): valid_algorithm = True

    # # ends program if algorithm in invalid
    if (valid_algorithm == False):
        print("Error! \nInvalid Type Parameter, choose from:\n1--standard\n2--absorbing\n3--points\n4--prob")
        sys.exit()


    # valid_field = False
    # if (field=="electric" ): valid_field = True
    # elif (field=="magnetic"): valid_field = True

    # # ends program if algorithm in invalid
    # if (field == False):
    #     print("Error! \nInvalid field Parameter, choose from:\n1--electric\n2--magnetic")
    #     sys.exit()


    conditions = (D, q, p, Type)

    # returns all imporant simulation parameters
    return N, conditions


def GetTao(N, a, b, c):

    tao = np.zeros(shape=(N,N), dtype=float)

    for ij in itertools.product(range(N), repeat=2):

        i,j = ij

        a_cell = a[i,j] 
        b_cell = b[i,j]
        c_cell = c[i,j]
        extra = (1 - a_cell - b_cell - c_cell)

        if (a_cell > b_cell and a_cell > c_cell and a_cell > extra): tao[i,j] = 1
        elif (b_cell > a_cell and b_cell > c_cell and b_cell > extra): tao[i,j] = 2
        elif (c_cell > a_cell and c_cell > b_cell and c_cell > extra): tao[i,j] = 3
        elif (extra > a_cell and extra > b_cell and extra> c_cell): tao[i,j] = 0

    return tao


def PDE_converge(N, conditions, matrix):

    # simulation constants hard coded
    dx = 1
    dt = 0.1

    # extracting constant parameters
    D, q, p, Type = conditions
    a, b, c = matrix

    tao = GetTao(N, a, b, c)

    cmap = ListedColormap(['grey', 'red', 'green', 'dodgerblue'], N=4)
    vmin=0
    vmax=4

    # setting up animantion figure
    fig, ax = plt.subplots()
    im=plt.imshow(tao, animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    # number of sweeps and terminal display counter
    nstep=200
    sweeps = 0

    if (Type=='points'): data=open(f'Data/PointsTao_{N}N_D{D}_q{q}_p{p}.txt','w')
    else: data=open(f'Data/tao_frac_{N}N_D{D}_q{q}_p{p}.txt','w')


    correlation_1 = np.zeros(shape=(int(N/2)))
    arrayLength = np.zeros(shape=(int(N/2)))

    for n in range(nstep):

        # calculating laplacian with loop or np.roll
        laplacian_a = Laplacian(a)
        laplacian_b = Laplacian(b)
        laplacian_c = Laplacian(c)

        a_new = a + dt*((D/(dx**2)) * laplacian_a + q*a*(1-a-b-c) - p*a*c)
        b_new = b + dt*((D/(dx**2)) * laplacian_b + q*b*(1-a-b-c) - p*a*b)
        c_new = c + dt*((D/(dx**2)) * laplacian_c + q*c*(1-a-b-c) - p*b*c)

        tao_new = GetTao(N, a_new, b_new, c_new)        

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
            # plt.colorbar(im, ax=ax)
            plt.draw()
            plt.pause(0.0001) 

            number_a_True = tao_new[tao_new==1]
            Num_a = np.count_nonzero(number_a_True)
            a_frac = Num_a/N**2

            number_b_True = tao_new[tao_new==2]
            Num_b = np.count_nonzero(number_b_True )
            b_frac = Num_b/N**2

            number_c_True = tao_new[tao_new==3]
            Num_c = np.count_nonzero(number_c_True )
            c_frac = Num_c/N**2 

            # saving time and free energy density data
            if (Type=='standard'): 
                data.write('{0:5.5e} {1:5.5e} {2:5.5e} {3:5.5e}\n'.format(sweeps, a_frac, b_frac, c_frac))


            if (Type=='absorbing'):
                if (a_frac>=1 or b_frac>=1 or c_frac>=1):
                    print('system converged')
                    return sweeps*dt, tao_new

                if (sweeps*dt>=1000): 
                    print('system unconverged')
                    sys.exit()

            if (Type=='points'): 
                point1 = a_new[int(N/2), int(N/2)]
                point2 = a_new[int(N/4), int(N/4)]
                data.write('{0:5.5e} {1:5.5e} {2:5.5e}\n'.format(sweeps*dt, point1, point2))

            if (Type=='prob'):
                prob = 0
                prob += correlation_prob(tao, N, correlation_1, arrayLength)
                


    if (Type=='prob'):
        np.savetxt('correlationData.txt', prob)

    data.close()

    return sweeps, tao_new
