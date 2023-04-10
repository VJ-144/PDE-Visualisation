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

    # if (total==True): return np.sqrt((xGrad_comp)**2 + (yGrad_comp)**2)
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

    phi = data.reshape(N, N, N)
    half_size = int(N/2)
    phi_slice = phi[:, :, half_size]

    charge_pos = np.sqrt(half_size**2 + half_size**2)

    data=open(f'Data/Radius_{condition}_Dependence_{N}N.txt','w')

    iteration=0
    
    if (condition=='electric'): x, y, z = Grad(phi, dx, total=True)
    elif (condition=='magnetic'): x, y, z = Curl(phi, dx, total=True)
    
    for ij in itertools.product(range(N), repeat=2):

        i, j = ij

        idx_pos = np.sqrt(i**2 + j**2)

        xx = np.abs(i-half_size)
        yy = np.abs(j-half_size)

        Radius = np.sqrt(xx**2 + yy**2)

        E_potential = phi_slice[i, j]
     
        E_field_idx = np.sqrt(x[i, j]**2 + y[i, j]**2)

        data.write('{0:5.5e} {1:10.10e} {2:10.10e}\n'.format(Radius, E_potential, E_field_idx))

        iteration+=1
        print(f'iteration={iteration}', end='\r')        

    data.close()

    return 0

def line(x, m, c):
    return m * x + c

def plotRadial(condition):

    data = np.loadtxt(f'Data_good/Radius_{condition}_Dependence_100N.txt')

    if (condition=='magnetic'): charge_title='Charged Wire'
    elif (condition=='electric'): charge_title='Gaussian Charge'

    radius = data[:,0]
    potential = data[:,1]
    electric = data[:,2]

    order_idx = np.argsort(electric)

    radius = radius[order_idx]
    potential = potential[order_idx]
    electric = electric[order_idx]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(f'{charge_title} Radial Dependence Analysis, N={N}', fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.4, hspace=0.4)

    ax1.set_title(f'{condition} Vector Potential')
    ax1.set_xlabel('Log Radius - ln(R)')
    ax1.set_ylabel('Potential Field  Strength - P')
    ax1.plot(np.log(radius), potential, marker='x', ls='')

    ax2.set_title(f'{condition} Field Strength')
    ax2.set_xlabel('Log Radius - ln(R)')
    ax2.set_ylabel(f'Log {condition} Field Strength - ln(E)')
    ax2.plot(np.log(radius), np.log(electric), marker='x', ls='')

    popt, pcov = curve_fit(line, np.log(radius)[8500:9980], np.log(electric)[8500:9980])
    m, c = popt

    xfit = np.linspace(0, 4, 100)
    ax2.plot(xfit, line(xfit, m, c), marker='', ls='-', color='black', label=f'm = {np.round(m,3)}')

    plt.legend()
    # plt.show()
    plt.savefig(f'Plots/RadiusDep_{condition}_{N}N.png')


    return 0 

def jacobi_slice(data, condition):

    phi = data.reshape(N, N, N)
    half_size = int(N/2)

    if (condition=='magnetic'): 

        charge_title='Charged Wire'
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
        fig.suptitle(f'{charge_title} Slice Potential Strength, N={N}', fontsize=16)
        plt.subplots_adjust(top=0.8, wspace=0.1, hspace=0.4)

        ax1.set_title(f'ZY Plane')
        ax1.imshow(phi[:, half_size, :], cmap='gnuplot')
        ax1.set_xlabel('Z-Axis')
        ax1.set_ylabel('Y-Axis')

        ax2.set_title(f'XY Plane')
        im = ax2.imshow(phi[:, :, half_size], cmap='gnuplot')
        ax2.set_xlabel('X-Axis')
        ax2.set_ylabel('Y-Axis')

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax1)
        plt.colorbar(im, cax=cax2)

    elif (condition=='electric'): 

        charge_title='Gaussian Charge'

        fig, ax1 = plt.subplots()
        fig.suptitle(f'{charge_title} Slice Potential Strength, N={N}', fontsize=15)
        plt.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)

        ax1.set_title(f'XY Plane')
        im = ax1.imshow(phi[:, half_size, :], cmap='gnuplot')
        ax1.set_xlabel('X-Axis')
        ax1.set_ylabel('Y-Axis')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

    plt.savefig(f'Plots/Jacobi_{condition}_{N}N_Slice.png')
    
    return 0

def jacobi_field(data, condition):

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

    N = 100
    dx = 1

    # filename = "Data_good/Jacobi_magneticField_100N_phi0.5.txt"
    filename = "Data_good/Jacobi_electricField_100N_phi0.5.txt"

    # sets condition as electric or magnetic data, i.e., charged wire of guassian charge 
    condition = filename[17:25]
    data = np.loadtxt(filename) 

    # hilliard_freeEnergy()
    # jacobi_field(data, condition)
    # jacobi_slice(data, condition)
    # ExtractRadialData(data, condition)
    # plotRadial(condition)

    return 0

main()