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
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks

def correlation_prob(tao):

    correlation_1 = np.zeros(shape=(int(N/2)))
    correlation_2 = np.zeros(shape=(int(N/2)))

    # diff_list = []

    for ijk in itertools.product(range(int(N/2)), repeat=3):

        k,i,j = ijk

        cell1 = tao[i,k]
        cell2 = tao[j,k]

        diff = np.abs(j - i)
        # diff_list.append(diff)

        if (cell1==cell2):
            correlation_1[diff] +=1
            correlation_2[diff] +=1


    prob = correlation_1/correlation_2

    # setting up plot figure/titles/labels
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    diff1 = np.linspace(0, 25, 25)

    ax.set_title(fr'Matching Species Proabilites {N}x{N} Matrix', pad=16)
    ax.errorbar(diff1, prob, marker='o', markersize = 4, linestyle='--', color='red')
    ax.set_xlabel('Radius [-]')
    ax.set_ylabel('Proability [-]')
    plt.legend()

    # saving to plot to file
    plt.savefig(f'Plots/MatchingSpecies_{N}N.png')
    plt.show()


def PlotContour(matrix):

    cmap = ListedColormap(['grey', 'red', 'green', 'dodgerblue'])
    vmin=0
    vmax=4

    # set ups single plot
    fig, ax = plt.subplots()
    fig.suptitle(f'N={N} D={D} q={q} p={p} Time={sweeps}', fontsize=15)
    plt.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)

    # ploting point charge XY plane
    ax.set_title(f'Matrix Snapshot')
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')

    # ploting color bar for plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    # saving plot
    plt.savefig(f'Plots/Tao_Snapshot_{N}N_D{D}_q{q}_p{p}_sweep{sweeps}.png')
    plt.show()
    
    return 0


def PlotEvolution(data):

    time = data[:,0]
    a = data[:,1]
    b = data[:,2]
    c = data[:,3]

    # setting up plot figure/titles/labels
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.set_title(fr'Evolution of Species {N}x{N} Matrix', pad=16)
    ax.errorbar(time, a, marker='o', markersize = 4, linestyle='--', color='red', label='a')
    ax.errorbar(time, b, marker='o', markersize = 4, linestyle='--', color='green', label='b')
    ax.errorbar(time, c, marker='o', markersize = 4, linestyle='--', color='blue', label='c')
    ax.set_xlabel('Time [sweeps]')
    ax.set_ylabel('Fractional Concentration [-]')
    plt.legend()

    # saving to plot to file
    plt.savefig(f'Plots/SpeciesEvolution_{N}N.png')
    plt.show()

def PlotPoints(data):

    time = data[:,0]
    a1 = data[:,1]
    a2 = data[:,2]

    # setting up plot figure/titles/labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(f'N={N} D={D} q={q} p={p} Time={sweeps}', fontsize=15)
    ax1.set_title(fr'Evolution of Points in A Species')
    ax1.errorbar(time, a1, marker='o', markersize = 4, linestyle='--', color='black', label='point 1')
    ax1.errorbar(time, a2, marker='o', markersize = 4, linestyle='--', color='orange', label='point 2')
    ax1.set_xlabel('Time [sweeps]')
    ax1.set_ylabel('A Matrix Species Value [-]')

    peak_idx = find_peaks(a1)[0]
    peak_values = a1[peak_idx]
    peak_time = time[peak_idx]


    ax2.set_title(fr'Best Fit for A Species Point')
    ax2.errorbar(peak_time, peak_values, marker='o', markersize = 4, linestyle='--', color='black', label='point 1')
    # ax2.errorbar(time, a2, marker='o', markersize = 4, linestyle='--', color='orange', label='point 2')
    ax2.set_xlabel('Time [sweeps]')
    ax2.set_ylabel('A Matrix Species Peaks [-]')



    plt.legend()

    frequency = 1200/4

    print(f'Period={np.round(1/frequency, 2)}')

    # saving to plot to file
    plt.savefig(f'Plots/A_Species_PointEvolution_{N}N.png')
    plt.show()

    return 0


def CalcAbsorbingTime(time):

    AbsorbTime = np.mean(time)
    AbsorbTime_err = np.std(time)

    data=open(f'Calculated_AbsorbingTime.txt','w')
    data.write('{0:5.20s} {1:5.20s}\n'.format('Absorbing-Time', 'Error'))
    data.write('{0:5.5e} {1:5.5e}\n'.format(AbsorbTime, AbsorbTime_err))
    data.close()

    print(fr'Absorbing Time = {np.round(AbsorbTime,2)}, Error={np.round(AbsorbTime_err,2)}')


def main():

    global N, D, q, p, sweeps

    N = 50
    D = 0.5
    q = 1
    p = 2.5
    sweeps = 2000

    # comment/uncomment desired file for plotting
    # data = np.loadtxt("AbsorbtionTimeData_good.txt")
    # data = np.loadtxt("Data/tao_frac_50N_D1.0_q1.0_p0.5_sweep1200.txt")
    data = np.loadtxt("Data/Tao_Mat_50N_D0.5_q1.0_p2.5_good2.txt")
    # data = np.loadtxt("Data/PointsTao_50N_D0.5_q1.0_p2.5_good.txt")

    
    # uncomment to use functions, they are all plotting functions
    # PlotEvolution(data)
    # PlotContour(data)
    # CalcAbsorbingTime(data)
    # PlotPoints(data)
    correlation_prob(data)

    return 0

main()