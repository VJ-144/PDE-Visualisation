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

def Grad(phi, dx, total=False):
    center = int(N/2)
    xGrad_comp = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2))[center, :, :] / (2*dx)
    yGrad_comp = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))[center, :, :] / (2*dx)
    zGrad_comp = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))[center, :, :] / (2*dx)

    normailse = np.sqrt(xGrad_comp**2 + yGrad_comp**2 + zGrad_comp**2)

    if (total==True): return -xGrad_comp, -yGrad_comp, -zGrad_comp

    return -xGrad_comp/normailse, -yGrad_comp/normailse, -zGrad_comp/normailse


def Curl(phi, dx, total=False):
    center = int(N/2)
    xCurl_comp = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))[:, :, center] / (2*dx)
    yCurl_comp = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))[:, :, center] / (2*dx)
    zCurl_comp = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2))[:, :, center] / (2*dx)

    normailse = np.sqrt(xCurl_comp**2 + yCurl_comp**2 + zCurl_comp**2)

    if (total==True): return xCurl_comp, -yCurl_comp, zCurl_comp

    return xCurl_comp/normailse, -yCurl_comp/normailse, zCurl_comp/normailse

def ExtractRadialData(data, condition):

    # reshaping stored 2D matrix data back into 3D matrix
    phi = data.reshape(N, N, N)
    half_size = int(N/2)

    # obtaining slice of 3D matrix to calculate radial dependence
    phi_slice = phi[:, :, half_size]

    # file to store calculated data
    data=open(f'Data/Radius_{condition}_Dependence_{N}N.txt','w')

    # counts iterations
    iteration=0
    
    # calculates appropriate field for point charge or charged wire
    if (condition=='electric'): x, y, z = Grad(phi, dx, total=True)
    elif (condition=='magnetic'): x, y, z = Curl(phi, dx, total=True)
    
    # iterating over 2D sliced matrix
    for ij in itertools.product(range(N), repeat=2):

        i, j = ij

        # calculate radius vector coordinates
        xx = np.abs(i-half_size)
        yy = np.abs(j-half_size)

        # calcuating radius magnitude
        Radius = np.sqrt(xx**2 + yy**2)

        # calculating potential at that point
        potential = phi_slice[i, j]
     
        # calculating electic or magnetic field
        field_idx = np.sqrt(x[i, j]**2 + y[i, j]**2)

        # storing radius, potential feild strength and electic/magnetic field strength
        data.write('{0:5.5e} {1:10.10e} {2:10.10e}\n'.format(Radius, potential, field_idx))

        iteration+=1
        print(f'iteration={iteration}', end='\r')        

    # close data file
    data.close()

    return 0

# line for fitting
def line(x, m, c):
    return m * x + c

def plotRadial(condition):

    # reading in data with radial dependence of potential strength and electric or magnetic field strength
    data = np.loadtxt(f'Data_good/Radius_{condition}_Dependence_100N.txt')

    # settign appropriate title names
    if (condition=='magnetic'): charge_title='Charged Wire'
    elif (condition=='electric'): charge_title='Point Charge'

    # reading in data for radius and electric/magnetic field strength
    radius = data[:,0]
    potential = data[:,1]
    electric = data[:,2]

    # sortind data into acending order
    order_idx = np.argsort(electric)

    radius = radius[order_idx]
    potential = potential[order_idx]
    electric = electric[order_idx]

    # setting up plot of potential and field strength
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(f'{charge_title} Radial Dependence Analysis, N={N}', fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.4, hspace=0.4)
    
    # conditions for plotting electric field of point charge
    if (condition=='electric'): 
        
        # plotting log(x) vs log(y) point charge potential data
        ax1.plot(np.log(radius), np.log(potential), marker='x', ls='')
        
        # line of best fit in data subset
        # 9900:10000 index gives m1=1.08 but covers shorter range, 9000:10000 index gives m1=1.19 but convers range 1-3
        popt1, pcov1 = curve_fit(line, np.log(radius)[9900:10000], np.log(potential)[9900:10000])
        m1, c1 = popt1

        # plotting
        ax1.set_title(f'{charge_title} Potential')
        ax1.set_xlabel('Log Radius - ln(R)')
        ax1.set_ylabel('Log Potential Field Strength - ln(P)')

    # conditions for plotting magnetic field of charged wire
    if (condition=='magnetic'): 

        # plotting log(x) vs y point charge potential data
        ax1.plot(np.log(radius), potential, marker='x', ls='')

        # line of best fit in data subset
        popt1, pcov1 = curve_fit(line, np.log(radius)[8500:9980], potential[8500:9980])
        m1, c1 = popt1

        # plotting
        ax1.set_title(f'{charge_title} Potential')
        ax1.set_xlabel('Log Radius - ln(R)')
        ax1.set_ylabel('Potential Field Strength - P')

    # line of best fit for electric/magnetic field data
    popt2, pcov2 = curve_fit(line, np.log(radius)[8500:9980], np.log(electric)[8500:9980])
    m2, c2 = popt2

    xfit = np.linspace(0, 4, 100)

    # ploting electric/magnetic field data and lines of best fit
    ax2.plot(np.log(radius), np.log(electric), marker='x', ls='')
    ax2.plot(xfit, line(xfit, m2, c2), marker='', ls='-', color='black', label=f'm = {np.round(m2,2)}')
    ax1.plot(xfit, line(xfit, m1, c1), marker='', ls='-', color='black', label=f'm = {np.round(m1,2)}')

    # plotting + titles
    ax2.set_title(f'{charge_title} {condition} Field Strength')
    ax2.set_xlabel('Log Radius - ln(R)')
    ax2.set_ylabel(f'Log {condition} Field Strength - ln(E)')

    ax1.legend()
    ax2.legend()

    # save plots
    plt.savefig(f'Plots/RadiusDep_{condition}_{N}N.png')
    plt.show()

    return 0 

def jacobi_slice(data, condition):

    # reshaping stored 2D matrix data back into 3D matrix
    phi = data.reshape(N, N, N)
    half_size = int(N/2)

    # plots both XY and ZY plane for charged wire
    if (condition=='magnetic'): 

        charge_title='Charged Wire'
        
        # set ups two plots
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
        fig.suptitle(f'{charge_title} Slice Potential Strength, N={N}', fontsize=16)
        plt.subplots_adjust(top=0.8, wspace=0.1, hspace=0.4)

        # ploting charged wire ZY plane
        ax1.set_title(f'ZY Plane')
        ax1.imshow(phi[:, half_size, :], cmap='gnuplot')
        ax1.set_xlabel('Z-Axis')
        ax1.set_ylabel('Y-Axis')

        # ploting charged wire XY plane
        ax2.set_title(f'XY Plane')
        im = ax2.imshow(phi[:, :, half_size], cmap='gnuplot')
        ax2.set_xlabel('X-Axis')
        ax2.set_ylabel('Y-Axis')

        # ploting color bars for both plots
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax1)
        plt.colorbar(im, cax=cax2)

    # plots XY plane for charged point
    elif (condition=='electric'): 

        charge_title='Point Charge'

        # set ups single plot
        fig, ax1 = plt.subplots()
        fig.suptitle(f'{charge_title} Potential Slice, N={N}', fontsize=15)
        plt.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)

        # ploting point charge XY plane
        ax1.set_title(f'XY Plane')
        im = ax1.imshow(phi[:, half_size, :], cmap='gnuplot')
        ax1.set_xlabel('X-Axis')
        ax1.set_ylabel('Y-Axis')

        # ploting color bar for plot
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

    # saving plot
    plt.savefig(f'Plots/Jacobi_{condition}_{N}N_Slice.png')
    plt.show()
    
    return 0

def jacobi_field(data, condition):

    # reshaping stored 2D matrix data back into 3D matrix
    phi = data.reshape(N, N, N)

    # calculating vector field for either point charge or charged wire
    if (condition=='electric'):
        x, y, z = Grad(phi, dx)
        scale1 = 0.7

    elif (condition=='magnetic'): 
        x, y, z = Curl(phi, dx)
        scale1 = 1

    # ploting vector field
    plt.title(f'Normalised Jacobi {condition} Field, N=100', pad=16, fontsize=16)
    plt.quiver(x, y, scale_units='xy', scale=scale1)
    plt.xlabel('X-Axis', fontsize=12)
    plt.ylabel('Y-Axis', fontsize=12)
    plt.xlim([40,60])
    plt.ylim([40,60])
    
    # saving plot
    plt.savefig(f'Plots/Jacobi_{condition}_{N}N_Field.png')
    plt.show()

    return 0

def hilliard_freeEnergy():

    # toggle plots for phi=0.0 or phi=0.5
    # filename = 'Data_good/hilliard_100N_phi0.0.txt'
    filename = 'Data_good/hilliard_100N_phi0.5.txt'

    # extracting phi value
    phi = float(filename[27:30])

    # extracting data
    rawData = np.loadtxt(filename)
    time = rawData[:,0]
    freeEnergy = rawData[:,1]

    # setting up plot figure/titles/labels
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.set_title(fr'Hilliard Free Energy Density $\phi$={phi}, N={N}', pad=16)
    ax.errorbar(time, freeEnergy, marker='o', markersize = 4, linestyle='--', color='black')
    ax.set_xlabel('Time [sweeps]')
    ax.set_ylabel('Free Energy Density')

    # saving to plot to file
    plt.savefig(f'Plots/hilliard_freeEnergy_phi{phi}_{N}N.png')
    plt.show()

    return 0


def plotSORS():

    filename = 'Data/seidel_electric_50N_phi0.0.txt'

    rawData = np.loadtxt(filename)
    iterations = rawData[:,0]
    omega = rawData[:,1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # finding omega which minimises the number of interations for convergence
    min_iterations = np.min(iterations)
    min_iter_idx = np.where(iterations == min_iterations)
    min_omega = omega[min_iter_idx][0]

    # setting figure title
    ax.set_title(fr'Point Charge Gauss-Seidel SORS Algorithm Convergence, N=50', pad=16)
    ax.errorbar(omega, iterations, marker='o', markersize = 4, linestyle='', color='black', label=f'Min $\omega$={min_omega}')
    ax.set_ylabel('Number of Iterations [-]')
    ax.set_xlabel('$\omega$ [-]')

    plt.legend()
    plt.savefig(f'Plots/seidel_electric_{N}N.png')
    plt.show()

    return 0

def main():

    global N, dx

    N = 100
    dx = 1

    # comment/uncomment desired file for plotting
    filename = "Data_good/Jacobi_magneticField_100N_phi0.5.txt"
    # filename = "Data_good/Jacobi_electricField_100N_phi0.5.txt"

    # sets condition as electric or magnetic data, i.e., uses point charge or charged wire data
    condition = filename[17:25]
    data = np.loadtxt(filename) 

    # uncomment to use functions, they are all plotting functions
    # hilliard_freeEnergy()
    # jacobi_field(data, condition)
    # jacobi_slice(data, condition)
    # ExtractRadialData(data, condition)
    # plotRadial(condition)
    plotSORS()

    return 0

main()