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
    # if (len(sys.argv) <= 4 or len(sys.argv) > 5):
    #     print("Error! \nFile Input: python IVP_Run.py N Phi field")
    #     sys.exit()

    # reads arguments from terminal command line
    N=int(sys.argv[1]) 
    D=float(sys.argv[2])
    p=float(sys.argv[3])
    q=float(sys.argv[4])


    # check if algorithm name from terminal  is valid
    # valid_algorithm = False
    # if (PDE=="jacobi" ): valid_algorithm = True
    # elif (PDE=="hilliard"): valid_algorithm = True
    # elif (PDE=="seidel"): valid_algorithm = True

    # ends program if algorithm in invalid
    # if (valid_algorithm == False):
    #     print("Error! \nInvalid PDE Algorithm Parameter, choose from:\n1--hilliard\n2--jacobi\n3--seidel")
    #     sys.exit()




    conditions = (D, p, q)

    # returns all imporant simulation parameters
    return N, conditions



def pad_edges(cube, field):
    """
    Pads edges of cube passed to it and makes them all zero, i.e., sets boundry conditions.
    """

    # sets X and Y cube edges to zero
    cube[[0,-1], :, :]=0
    cube[:, [0,-1], :]=0

    # sets Z cube edge to zero if simulation is a point charge
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
    """
    updates 3D charged cube using jacobi algorithm
    """

    # extracts all simulation constants and parameters
    a, k, M, dx, dt, phi_param, field = conditions

    # Jacobi algorithm 
    xGrad_comp = np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
    yGrad_comp = np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
    zGrad_comp = np.roll(phi, 1, axis=2) + np.roll(phi, -1, axis=2)
    rho = phi_charged * dx**2

    new_phi = (1/6) * (xGrad_comp + yGrad_comp + zGrad_comp + rho)

    return new_phi


def seidel_algorithm(N, phi, phi_charged, w, field):
    """
    updates 3D charged cube using seidel algorithm
    """

    if (field=='electric'): z_lowBnd, z_highDnd = 1, N-1
    elif (field=='magnetic'): z_lowBnd, z_highDnd = 0, N


    # keeps a copy of old phi before update
    old_phi = np.copy(phi)

    # updating phi charged cube 
    for ijk in itertools.product(range(1, N-1), range(1, N-1), range(z_lowBnd, z_highDnd)):

        i, j, k = ijk

        if (field=='electric'):
            phi_gs = (1/6)*(phi[(i+1), j, k] + phi[i, (j+1), k] + phi[i, j, (k+1)] + phi[(i-1), j, k]  \
            + phi[i, (j-1), k] + phi[i, j, (k-1)] + phi_charged[i, j, k])

        # magnetic requires the treatment of different boundry conditions
        if (field=='magnetic'):
            phi_gs = (1/6)*(phi[(i+1), j, k] + phi[i, (j+1), k] + phi[i, j, (k+1)%N] + phi[(i-1), j, k]  \
            + phi[i, (j-1), k] + phi[i, j, (k-1)%N] + phi_charged[i, j, k])

        phi[i,j,k] = w*phi_gs + (1-w)*old_phi[i, j, k]

    # return updated and old phi
    return phi, old_phi


def jacobi_converge(N, conditions):
    """
    Creates a 3D cube of set size (N) with a point charge or charged wire and 
    converges the potential field generated using jacobi algorithm
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
        
        # update charged cube using jacobi algorithm
        new_phi = jacobi_algorithm(phi, conditions, phi_charged)

        # pads edges, i.e. enforces boundry conditions dependent on if its a gusassin charge or wire  
        new_phi = pad_edges(new_phi, field)

        # convergence criteria
        diff = np.sum(np.abs(new_phi - phi))
        if (np.allclose(new_phi, phi, rtol=5e-8, atol=5e-8)): break   

        # counts number of iterations + prints to terminal
        iterations += 1
        print(f'sweeps={iterations}, update difference={diff}', end='\r')

        # feeds charged cube back into algorithm
        phi = np.copy(new_phi)

    # returns converged phi and number of iterations for convergence
    return phi, iterations


def seidel_converge(N, conditions, w):
    """
    Creates a 3D cube of set size (N) with a point charge only and 
    converges the potential field generated using seidel algorithm
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
        
        # using guass-seidel algorithm for update rule
        phi, old_phi= seidel_algorithm(N, phi, phi_charged, w, field)

        # pads edges, i.e. enforces boundry conditions dependent on if its a point charge or charged wire  
        phi = pad_edges(phi, field)

        # convergence criteria
        diff = np.sum(np.abs(phi - old_phi))
        if (np.allclose(old_phi, phi, rtol=1e-8, atol=1e-8)): break   

        # counts number of iterations + prints to terminal
        iterations += 1
        print(f'sweeps={iterations}, update difference={diff}', end='\r')

        # phi is fed back into the algorithm and becomes phi_old, and a new updated version is generated called the phi

    # returns converged charged cube and numer of iterations for convergence
    return phi, iterations


def cahn_hilliard(N, conditions, phi0):

    # extracting constant parameters
    D, p, q = conditions

    # setting up animantion figure
    fig = plt.figure()
    im=plt.imshow(phi0, animated=True)

    # number of sweeps and terminal display counter
    nstep=1000000
    sweeps = 0

    # data=open(f'Data/hilliard_{N}N_phi{phi_param}.txt','w')

    for n in range(nstep):

        # calculating laplacian with loop or np.roll
        # laplacian_phi = Laplacian(phi0, N, conditions)

        # # calculating chemical potential matrix
        # chem_matrix = -a * phi0 + a * phi0**3 - k * laplacian_phi            

        # # calculating laplacian of chemical potential
        # laplacian_chem = Laplacian(chem_matrix, N, conditions)

        # # calculating new d(phi) = M*dt*c*laplancian
        # phi0 += (((M*dt)/(dx**2))*laplacian_chem)


        # visuals in set number of sweeps
        if(n%1==0): 

            # prints current number of sweep to terminal
            sweeps += 1
            print(f'sweeps={sweeps}', end='\r')

            # animates configuration 
            plt.cla()
            im=plt.imshow(phi0, interpolation='gaussian', animated=True)
            plt.draw()
            plt.pause(0.0001) 

            # calculating free energy density
            # energy_density = freeEnergy(phi0, N, conditions)

            # saving time and free energy density data
            # data.write('{0:5.5e} {1:5.5e}\n'.format(sweeps, energy_density))

    # data.close()

    return phi0
