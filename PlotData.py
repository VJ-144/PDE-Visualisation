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


def PlotAverage(data):

    # extracting data
    time = data[:,0]
    phi_avg = data[:,1]
    m_avg = data[:,2]


    # setting up plot figure/titles/labels
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(fr'{Algorithm} Algorithm - Average Solution Evolution, {N}x{N} Matrix, $\alpha$={alpha}', fontsize=16)
    # fig.suptitle(fr'{Algorithm} Algorithm - Average Solution Evolution, {N}x{N} Matrix', fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.3, hspace=0.4)

    ax1.set_title(fr'$\phi$={phi0} Solution Evolution')
    ax1.errorbar(time, phi_avg, marker='o', markersize = 4, linestyle='--', color='black')
    ax1.set_xlabel('Time [sweeps]')
    ax1.set_ylabel('Solution Average [-]')

    ax2.set_title(fr'm Solution Evolution, $\chi$={chi}')
    ax2.errorbar(time, m_avg, marker='o', markersize = 4, linestyle='--', color='black')
    ax2.set_xlabel('Time [sweeps]')
    ax2.set_ylabel('Solution Average [-]')

    # saving to plot to file
    plt.savefig(f'Plots/{Algorithm}Average_{N}N_phi{phi0}_chi_{chi}_Time{sweeps}.png')
    plt.savefig(f'Plots/{Algorithm}Average_{N}N_phi{phi0}_chi_{chi}_Time{sweeps}_{alpha}.png')
    plt.show()


def PlotPhase(alpha, avg, var, symbol):

    # setting up plot figure/titles/labels
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(fr'{Algorithm} Algorithm - $\alpha$ Evolution, {N}x{N} Matrix', fontsize=16)
    # fig.suptitle(fr'{Algorithm} Algorithm - Average Solution Evolution, {N}x{N} Matrix', fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.3, hspace=0.4)

    ax1.set_title(fr'{symbol} Solution Average')
    ax1.errorbar(alpha, avg, marker='o', markersize = 4, linestyle='--', color='black')
    ax1.set_xlabel(fr'$\alpha$ [-]')
    ax1.set_ylabel('Solution Average [-]')

    ax2.set_title(fr'{symbol} Solution Varience')
    ax2.errorbar(alpha, var, marker='o', markersize = 4, linestyle='--', color='black')
    ax2.set_xlabel(fr'$\alpha$ [-]')
    ax2.set_ylabel('Solution Varience [-]')

    # saving to plot to file
    plt.savefig(f'Plots/VariedAlpha_{N}N_{symbol}.png')
    plt.show()

def plotContour(matrix1, matrix2):
        
    # set ups two plots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(fr'{Algorithm} Algorithm - Average Solution Evolution, {N}x{N} Matrix, $\alpha$={alpha}', fontsize=16)
    # fig.suptitle(f'Standard Algorithm {N}x{N} Matrix', fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.1, hspace=0.4)

    # ploting charged wire ZY plane
    ax1.set_title(fr'$\phi$={phi0} Solution', pad=15)
    ax1.imshow(matrix1, cmap='gnuplot')
    ax1.set_xlabel('X-Axis')
    ax1.set_ylabel('Y-Axis')

    # ploting charged wire XY plane
    ax2.set_title(f'm Solution, $\chi$={chi}', pad=15)
    im = ax2.imshow(matrix2, cmap='gnuplot')
    ax2.set_xlabel('X-Axis')
    ax2.set_ylabel('Y-Axis')

    # ploting color bars for both plots
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax1)
    plt.colorbar(im, cax=cax2)

    # plt.savefig(f'Plots/{Algorithm}Snapshot_{N}N_phi{phi0}_chi_{chi}_Time{sweeps}.png')
    plt.savefig(f'Plots/{Algorithm}Snapshot_{N}N_phi{phi0}_chi_{chi}_Time{sweeps}_alpha{alpha}.png')
    plt.show()

def main():

    global N, dx, phi0, chi, sweeps, Algorithm, alpha

    N = 50
    dx = 1
    phi0=0.5
    chi=0.3
    sweeps = 2000
    alpha=0.005

    # Algorithm = 'Standard'
    Algorithm = 'Reaction'


    # data1 = np.loadtxt('Data/standardPhi_Mat_50N_phi0.5_chi_0.3_Time5000.txt')
    # data2 = np.loadtxt('Data/standardMag_Mat_50N_phi0.5_chi_0.3_Time5000.txt')

    data1 = np.loadtxt('Data/reaction/reactionPhi_Mat_50N_phi0.5_chi_0.3_Time2500_alpha0.005.txt')
    data2 = np.loadtxt('Data/reaction/reactionMag_Mat_50N_phi0.5_chi_0.3_Time2500_alpha0.005.txt')

    # data1 = np.loadtxt('Data/Varied_Alpha/reaction_Evol_50N_Varied_Alpha_Time2500_feedbackIn.txt')
    # alpha = data1[:,0]
    # m_avg = data1[:,3]
    # m_var = data1[:,4]
    # symbol = 'm'

    # uncomment to use functions, they are all plotting functions
    # PlotAverage(data1)
    plotContour(data1, data2)    
    # ExtractRadialData(data)
    # plotRadial()
    # PlotPhase(alpha, m_avg, m_var, symbol)


    return 0

main()