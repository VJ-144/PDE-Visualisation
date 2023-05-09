"""
This script is for plotting data from IVP_Run_new.py

It has functionality to plot both vector fields for the magnetic and electric fields for the charged wire and point charge.
It can also plot the omega convergence for the SORS algorithm and the radial dependence of the electric and megnetic fields
within the cube. The file also contains functions which plot the minimization of the free energy in the hilliard model.

This file is specifically only compatible with the outputs of the IVP_Run_new.py file and requires existing a directory 
called 'Data_good'.

The plot options are contained in a main() function at the bottom of the file and can be toggled on/off.

Pre-existing data filenames are hard coded for the hilliard and SORS algorithm but can be changed in the function.

The data for the point charge and charged wire can be toggled on/off in the main function.

This file uses the conditional arguments 'electric' and 'magnetic' to refer to the point charge or charged wire 

The file can be run with: python IVP_PlotData.py
"""


import string
import sys
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ExtractRadialData(phi):

    half_size = int(N/2)

    # file to store calculated data
    data=open(f'Data/RadiusData_{N}N_omega{omega}_kappa{kappa}.txt','w')

    # counts iterations
    iteration=0
    
    # iterating over 2D sliced matrix
    for ij in itertools.product(range(N), repeat=2):

        i, j = ij

        # calculate radius vector coordinates
        xx = np.abs(i-half_size)
        yy = np.abs(j-half_size)

        # calcuating radius magnitude
        Radius = np.sqrt(xx**2 + yy**2)

        # calculating potential at that point
        value = phi[i, j]

        # storing radius, potential feild strength and electic/magnetic field strength
        data.write('{0:5.5e} {1:10.10e}\n'.format(Radius, value))

        iteration+=1
        print(f'iteration={iteration}', end='\r')        

    # close data file
    data.close()


def expon(x, a, b):
    return a*np.exp(b*x)

def powerLaw(x, a, b):
    return a*(x**b)

def plotRadial():

    data = np.loadtxt(f'Data\RadiusData_50N_omega10_kappa0.01.txt')

    radius = data[:,0]
    value = data[:,1]

    # sortind data into acending order
    order_idx = np.argsort(radius)

    radius = radius[order_idx]
    value = value[order_idx]

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    ax.set_title(fr'{Algorithm} Algorithm Radial Analysis, Time={sweeps}, N={N}')
    ax.set_ylabel('Diffusion Matrix Value [-]')
    ax.set_xlabel('Radius [-]')
    ax.plot(radius, value, marker='x', ls='')

    popt1, pcov1 = curve_fit(expon, radius[100:1500], value[100:1500])
    popt2, pcov2 = curve_fit(powerLaw, radius[100:1500], value[100:1500])

    xfit = np.linspace(3, 35, 100)
    ax.plot(xfit, expon(xfit, *popt1), marker='', ls='--', color='black', label=f'exp')
    ax.plot(xfit, powerLaw(xfit, *popt2), marker='', ls='-', color='red', label=f'power law')

    ax.legend()

    # save plots
    plt.savefig(f'Plots/{Algorithm}_RadiusDep_T{sweeps}_omega{omega}_kappa{kappa}.png')
    plt.show()


def phi_average(data):

    # extracting data
    time = data[:,0]
    average = data[:,1]

    # setting up plot figure/titles/labels
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.set_title(fr'Average $\phi$ Evolution {N}x{N} Matrix', pad=16)
    ax.errorbar(time, average, marker='o', markersize = 4, linestyle='--', color='black')
    ax.set_xlabel('Time [sweeps]')
    ax.set_ylabel('Average $\phi$')

    # saving to plot to file
    # plt.savefig(f'Plots/{Algorithm}Average_T{sweeps}_v0{v0}.png')
    plt.savefig(f'Plots/{Algorithm}Average_T{sweeps}_omega{omega}_kappa{kappa}.png')
    plt.show()

def plotContour(matrix):

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.set_title(fr'{Algorithm} Algorithm at Time={sweeps}, N={N}', pad=16)
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')

    im = ax.imshow(matrix, origin='lower', extent=[0,1,0,1], cmap='gnuplot')
    fig.colorbar(im, ax=ax, orientation='vertical')

    plt.savefig(f'Plots/{Algorithm}Snapshot_T{sweeps}_v0{v0}.png')
    # plt.savefig(f'Plots/{Algorithm}Snapshot_T{sweeps}_omega{omega}_kappa{kappa}.png')
    plt.show()

def main():

    global N, dx, omega, kappa, Algorithm, sweeps, v0

    N = 50
    dx = 1
    omega=10
    kappa=0.01
    sweeps = 5000
    v0 = 0.5

    # Algorithm = 'Standard'
    Algorithm = 'Velocity'

    # data = np.loadtxt('Data/standardDiffusion_Evol_50N_omega10.0_kappa0.01_time5000.txt')
    # data = np.loadtxt('Data/standardDiffusion_Mat_50N_omega10.0_kappa0.1_time5000.txt')

    # data = np.loadtxt('Data/velocityDiffusion_Evol_50N_omega10.0_kappa0.01_time5000_v00.01.txt')
    data = np.loadtxt('Data/velocityDiffusion_Mat_50N_omega10.0_kappa0.01_time5000_v00.5.txt')


    # uncomment to use functions, they are all plotting functions
    # phi_average(data)
    plotContour(data)    
    # ExtractRadialData(data)
    # plotRadial()


    return 0

main()