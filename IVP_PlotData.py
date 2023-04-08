import string
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def Grad(phi, dx):
    center = int(N/2)
    xGrad_comp = (np.roll(phi[:,:,], -1, axis=2) - np.roll(phi[:,:,], 1, axis=2))[center, :, :] / (2*dx)
    yGrad_comp = (np.roll(phi[:,:,], -1, axis=1) - np.roll(phi[:,:,], 1, axis=1))[center, :, :] / (2*dx)
    zGrad_comp = (np.roll(phi[:,:,], -1, axis=0) - np.roll(phi[:,:,], 1, axis=0))[center, :, :] / (2*dx)
    return -xGrad_comp, -yGrad_comp, -zGrad_comp


def Curl(phi, dx):
    center = int(N/2)
    xCurl_comp = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))[:, :, center] / (2*dx)
    yCurl_comp = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))[:, :, center] / (2*dx)
    zCurl_comp = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2))[:, :, center] / (2*dx)
    return xCurl_comp, -yCurl_comp, zCurl_comp


def jacobi_slice():

    condition = 'electric'
    data = np.loadtxt(f"Data/Jacobi_electricField_50N_phi0.5.txt") 

    # condition = 'magnetic'
    # data = np.loadtxt(f"Data/Jacobi_magneticField_50N_phi0.5.txt")

    phi = data.reshape(N, N, N)

    if (condition=='electric'): x, y, z = Grad(phi, dx)
    elif (condition=='magnetic'): x, y, z = Curl(phi, dx)

    a = N + 2
    b = N // 2

    # Creating plot
    plt.title(f'Jacobi {condition} Slice')
    plt.imshow(phi[0:a,b,0:a])
    
    plt.colorbar()
    plt.show()
    # plt.savefig(f'Plots/Jacobi_{condition}_Slice.png')
    
    return 0

def jacobi_field():

    # condition = 'electric'
    # data = np.loadtxt(f"Data/Jacobi_electricField_50N_phi0.5.txt") 

    condition = 'magnetic'
    data = np.loadtxt(f"Data/Jacobi_magneticField_50N_phi0.5.txt")

    phi = data.reshape(N, N, N)

    if (condition=='electric'): x, y, z = Grad(phi, dx)
    elif (condition=='magnetic'): x, y, z = Curl(phi, dx)

    # Creating plot
    plt.title(f'Jacobi {condition} Field')
    plt.quiver(x, y)
    
    plt.show()
    # plt.savefig(f'Plots/Jacobi_{condition}_Field.png')
    
    return 0

def hilliard_freeEnergy():

    # filename = 'hilliard_50N_phi0.0.txt'
    filename = 'hilliard_50N_phi0.5.txt'

    phi = float(filename[16:19])

    rawData = np.loadtxt(filename)
    time = rawData[:,0]
    freeEnergy = rawData[:,1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # setting figure title
    ax.set_title(fr'Hilliard Free Energy Density $\phi$={phi}', pad=16)
    ax.errorbar(time, freeEnergy, marker='o', markersize = 4, linestyle='--', color='black')
    ax.set_xlabel('Time [sweeps]')
    ax.set_ylabel('Free Energy Density [?]')
    plt.savefig(f'Plots/hilliard_freeEnergy_phi{phi}.png')

    return 0



def main():

    global N, dx

    N = 50
    dx = 1

    # hilliard_freeEnergy()
    jacobi_field()
    # jacobi_slice()

    return 0

main()