"""
This file contains all the functions required to run and visualise an implmentation of the Ising Model
using both Glauber and Kawasaki dynamics. These functions are used to run the simulation in 
the run.ising.simulation.py file.

The functions require the existence of the /Data/Glauber and /Data/Kawasaki directories
to store calculated data and run error analysis. The directory names are spelling and case sensitive.

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


def Laplacian(phi0, N, conditions):
    """
    Calculates the laplacian of the 3D cube passed to it
    """

    # extract neccesary system parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # calculates laplacian Y and X componets for full matrix + center all correction
    yGrad_comp = np.roll(phi0, -1, axis =0) + np.roll(phi0, 1, axis=0)
    xGrad_comp = np.roll(phi0, -1, axis =1) + np.roll(phi0, 1, axis =1)
    center_corr = -4 * phi0

    # calculates laplacian
    laplacian = yGrad_comp + xGrad_comp + center_corr

    return laplacian


def freeEnergy(phi0, N, conditions):
    """
    Calculates the free energy density for 3D cube passed to it
    """

    # extract neccesary system parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # calculates laplacian Y and X componets for nabla + combines them for full expression
    yGrad_comp = ( np.roll(phi0, -1, axis =0) - np.roll(phi0, 1, axis=0) ) 
    xGrad_comp = ( np.roll(phi0, -1, axis =1) - np.roll(phi0, 1, axis =1) ) 
    grad_phi = (yGrad_comp/2*dx)**2 + (xGrad_comp/2*dx)**2

    # calculates free energy density matix
    E_density = -(a/2) * phi0**2 + (a/4) * phi0**4 + (k/2) * grad_phi

    # returns total free energy density
    return np.sum(E_density)



def initialise_simulation():
    """
    Sets up all parameters for the simulation + reads arguments from terminal
    """

    # checks number of command line arguments are correct otherwise stops simulation
    if (len(sys.argv) <= 3 or len(sys.argv) > 4):
        print("Error! \nFile Input: python IVP_Run.py N Phi")
        sys.exit()

    N=int(sys.argv[1]) 
    phi=float(sys.argv[2])
    PDE=str(sys.argv[3])

    # check if algorithm is valid
    valid_algorithm = False
    if (PDE=="jacobi" ): valid_algorithm = True
    elif (PDE=="hilliard"): valid_algorithm = True
    elif (PDE=="seidel"): valid_algorithm = True

    if (valid_algorithm == False):
        print("Error! \nInvalid PDE Method Parameter, choose from:\n1--hilliard\n2--jacobi\n3--seidel")
        sys.exit()

    # simulation constants hard coded
    a = 0.1
    M = 0.1
    k = 0.1
    dx = 1
    dt = 1

    # toggle if magnetic or electic fields are wanted
    # toggles between guassian charge (electric) and charged wire (magnetic)
    field = 'magnetic'
    # field = 'electric'

    conditions = (a, k, M, dx, dt, phi, field)

    # returns all imporant simulation parameters
    return N, phi, conditions, PDE



def pad_edges(cube, field):
    """
    Pads edges of cube passed to it and makes them all zero, i.e., sets boundry conditions.
    """

    # sets X and Y cube edges to zero
    cube[[0,-1], :, :]=0
    cube[:, [0,-1], :]=0

    # sets Z cube edge to zero if simulation is a guassian charge
    if (field=='electric'): cube[:, :, [0,-1]]=0

    # returns cube with boundry counditions
    return cube

def addChargedParticle(N):
    """
    Creates a 3D cube of set size (N) and places gussian charge in it
    """

    # creating empty cube and finding center coordinate
    cube = np.zeros(shape=(N,N,N))

    center_idx = int(N/2)

    # adding electric charge in the center of cube
    cube[center_idx, center_idx, center_idx] = 1

    # returning cube with gussian charge
    return cube


def addChargedWire(N):
    """
    Creates a 3D cube of set size (N) and places charges wire in it
    """

    # creating empty cube and finding center coordinate
    cube = np.zeros(shape=(N,N,N))

    center_idx = int(N/2)

    # adding electric wire to to center of cube
    cube[center_idx, center_idx,:] = 1

    # returns cube with charged wire through the center
    return cube


def jacobi_algorithm(phi, conditions, phi_charged):

    # extracts all simulation constants and parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # Jacobi algorithm 
    xGrad_comp = np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
    yGrad_comp = np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
    zGrad_comp = np.roll(phi, 1, axis=2) + np.roll(phi, -1, axis=2)
    rho = phi_charged * dx**2

    new_phi = (1/6) * (xGrad_comp + yGrad_comp + zGrad_comp + rho)

    return new_phi


def seidel_algorithm(N, phi, phi_charged, w):

    old_phi = np.copy(phi)
    for ijk in itertools.product(range(N), repeat=3):

        i, j, k = ijk

        phi_gs = (1/6)*(phi[(i+1)%N, j, k] + phi[i, (j+1)%N, k] + phi[i, j, (k+1)%N] + phi[(i-1)%N, j, k] // 
        + phi[i, (j-1)%N, k] + phi[i, j, (k-1)%N] + phi_charged[i, j, k])

        phi[i,j,k] = w*phi_gs + (1-w)*old_phi[i, j, k]

    return phi


def charged_cube(N, conditions, PDE, w=0):
    """
    Creates a 3D cube of set size (N) and places charges wire in it
    """

    # extracts all simulation constants and parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # creates initial empty simulation cube - phi
    phi = np.zeros(shape=(N,N,N))

    # creates a new cube with either a guassian charge or chared wire + pads edges
    if (field=='electric'):
        phi_charged = addChargedParticle(N)
    elif (field=='magnetic'):
        phi_charged = addChargedWire(N)
        phi_charged = pad_edges(phi_charged, field)


    # counts interations
    iterations = 0

    # starts self consisten field algorithm
    while True:

        if (PDE=='jacobi'): new_phi = jacobi_algorithm(phi, conditions, phi_charged)
        elif(PDE=='seidel'): new_phi = seidel_algorithm(N, phi, phi_charged, w)

        # pads edges, i.e. enforces boundry conditions dependent on if its a gusassin charge or wire  
        new_phi = pad_edges(new_phi, field)

        # convergence criteria
        # diff = np.mean(np.abs(new_phi-phi))
        # if (diff < 10e-3): break
        if (np.allclose(new_phi, phi, rtol=1e-9, atol=1e-9)): break   

        # counts number of iterations + prints to terminal
        iterations += 1
        print(f'sweeps={iterations}', end='\r')

        # feeds charged cube back into algorithm
        phi = np.copy(new_phi)

    return phi, iterations


def cahn_hilliard(N, conditions, phi0):

    # extracting constant parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # setting up animantion figure
    fig = plt.figure()
    im=plt.imshow(phi0, animated=True)

    # number of sweeps and terminal display counter
    nstep=1000000
    sweeps = 0

    data=open(f'Data/hilliard_{N}N_phi{phi_param}.txt','w')

    for n in range(nstep):

        # calculating laplacian with loop or np.roll
        laplacian_phi = Laplacian(phi0, N, conditions)

        # calculating chemical potential matrix
        chem_matrix = -a * phi0 + a * phi0**3 - k * laplacian_phi            

        # calculating laplacian of chemical potential
        laplacian_chem = Laplacian(chem_matrix, N, conditions)

        # calculating new d(phi) = M*dt*c*laplancian
        phi0 += (((M*dt)/(dx**2))*laplacian_chem)


        # visuals in set number of sweeps
        if(n%100==0): 

            # prints current number of sweep to terminal
            sweeps +=100
            print(f'sweeps={sweeps}', end='\r')

            # animates configuration 
            plt.cla()
            im=plt.imshow(phi0, interpolation='gaussian', animated=True)
            plt.draw()
            plt.pause(0.0001) 

            # saving time and free energy density data
            energy_density = freeEnergy(phi0, N, conditions)

            data.write('{0:5.5e} {1:5.5e}\n'.format(sweeps, energy_density))

    data.close()

    return phi0
