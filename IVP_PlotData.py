import string
import sys
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler


def Grad(phi, dx, total=False):
    center = int(N/2)
    xGrad_comp = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2))[center, :, :] / (2*dx)
    yGrad_comp = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))[center, :, :] / (2*dx)
    zGrad_comp = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))[center, :, :] / (2*dx)

    normailse = np.sqrt(xGrad_comp**2 + yGrad_comp**2 + zGrad_comp**2)

    if (total==True): return np.sqrt((xGrad_comp)**2 + (yGrad_comp)**2)

    return -xGrad_comp/normailse, -yGrad_comp/normailse, -zGrad_comp/normailse


def Curl(phi, dx):
    center = int(N/2)
    xCurl_comp = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))[:, :, center] / (2*dx)
    yCurl_comp = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))[:, :, center] / (2*dx)
    zCurl_comp = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2))[:, :, center] / (2*dx)

    normailse = np.sqrt(xCurl_comp**2 + yCurl_comp**2 + zCurl_comp**2)
    return xCurl_comp/normailse, -yCurl_comp/normailse, zCurl_comp/normailse

def ExtractRadialData():

    data = np.loadtxt(f"Data/Jacobi_electricField_50N_phi0.5.txt") 
    # data = np.loadtxt(f"Data/Jacobi_magneticField_50N_phi0.5.txt") 

    phi = data.reshape(N, N, N)
    half_size = int(N/2)
    phi_slice = phi[:, half_size, :]

    charge_pos = np.sqrt(half_size**2 + half_size**2)

    data=open(f'Data/Radius_Dependence_{N}N.txt','w')

    iteration=0

    for ij in itertools.product(range(N), repeat=2):
        iteration+=1
        print(f'iteration={iteration}', end='\r')

        i, j = ij

        idx_pos = np.sqrt(i**2 + j**2)

        dis2Radius = np.abs(charge_pos - idx_pos)

        E_potential = phi_slice[i, j]

        E_field = Grad(phi, dx, total=True)

        E_field_idx = E_field[i, j]


        data.write('{0:5.5e} {1:10.10e} {2:10.10e}\n'.format(dis2Radius, E_potential, E_field_idx))

    data.close()

    return 0


def plotRadial():

    data = np.loadtxt('Data/Radius_Dependence_50N.txt')

    radius = data[:,0]
    potential = data[:,1]
    electric = data[:,2]

    fig, (ax2, ax1) = plt.subplots(1,2, figsize=(9,4))

    ax1.set_title('Electric Field Strength')
    ax1.set_xlabel('ln(R)')
    ax1.set_ylabel('Electric Field')
    ax1.scatter(radius, electric)

    ax2.set_title('Potential Field Strength')
    ax2.set_xlabel('ln(R)')
    ax2.set_ylabel('Potential Strength')
    ax2.scatter(radius, potential, marker='o')

    plt.show()


    return 0 

def jacobi_slice():

    condition = 'electric'
    data = np.loadtxt(f"Data/Jacobi_electricField_50N_phi0.5.txt") 

    # condition = 'magnetic'
    # data = np.loadtxt(f"Data_good/Jacobi_magneticField_100N_phi0.5.txt")

    phi = data.reshape(N, N, N)
    half_size = int(N/2)

    if (condition=='magnetic'): charge_title='Charged Wire'
    elif (condition=='electric'): charge_title='Gaussian Charge'

    # Creating plot
    plt.title(f'Jacobi {charge_title} Slice, N=100')
    plt.imshow(phi[:, half_size, :], cmap='gnuplot')
    
    plt.colorbar()
    # plt.show()
    plt.savefig(f'Plots/Jacobi_{condition}_{N}N_Slice.png')
    
    return 0

def jacobi_field():

    condition = 'electric'
    data = np.loadtxt(f"Data_good/Jacobi_electricField_100N_phi0.5.txt") 

    # condition = 'magnetic'
    # data = np.loadtxt(f"Data/Jacobi_magneticField_100N_phi0.5.txt")

    phi = data.reshape(N, N, N)

    if (condition=='electric'):
        x, y, z = Grad(phi, dx)
        scale1 = 0.7

    elif (condition=='magnetic'): 
        x, y, z = Curl(phi, dx)
        scale1 = 1

    # Creating plot
    plt.title(f'Normalised Jacobi {condition} Field, N=100')
    plt.quiver(x, y, scale_units='xy', scale=scale1)
    plt.xlim([40,60])
    plt.ylim([40,60])
    
    # plt.show()
    plt.savefig(f'Plots/Jacobi_{condition}_{N}N_Field.png')
    
    return 0

def hilliard_freeEnergy():

    # filename = 'Data/hilliard_100N_phi0.0.txt'
    filename = 'Data/hilliard_100N_phi0.5.txt'

    phi = float(filename[22:25])

    rawData = np.loadtxt(filename)
    time = rawData[:,0]
    freeEnergy = rawData[:,1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # setting figure title
    ax.set_title(fr'Hilliard Free Energy Density $\phi$={phi}, N={N}', pad=16)
    ax.errorbar(time, freeEnergy, marker='o', markersize = 4, linestyle='--', color='black')
    ax.set_xlabel('Time [sweeps]')
    ax.set_ylabel('Free Energy Density [?]')
    plt.savefig(f'Plots/hilliard_freeEnergy_phi{phi}_{N}N.png')

    return 0




def main():

    global N, dx

    N = 50
    dx = 1

    # hilliard_freeEnergy()
    # jacobi_field()
    # jacobi_slice()
    # ExtractRadialData()
    plotRadial()

    return 0

main()