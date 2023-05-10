"""
This file is used to run the the checkpoint.

This file must be run through terminal with the input arguments as follows:

'python IVP_Run_new.py N phi algorithm field'

A full explanation of the parameters can be found in the README.txt file
"""


import Functions as func
import numpy as np
import time
import sys
import random


def main():

    # number of times the simulation is run, default is 1
    NumOfRun = 1

    # run conditions read in from terminal + offset noise
    N, conditions = func.initialise_simulation()
    nstep = 2500
    noise = 0.01

    phi0, chi, algorithm, alpha, Batch = conditions

    phi = np.random.choice([phi0-noise, phi0+noise], size=[N,N])
    m = np.random.choice([-noise, noise], size=[N,N])

    if (Batch=='False'):

        # runs simulation
        phi_convg, m_convg = func.diff_equation(N, conditions, phi, m, nstep)

        np.savetxt(f'Data/{algorithm}/{algorithm}Mag_Mat_{N}N_phi{phi0}_chi_{chi}_Time{nstep}_alpha{alpha}.txt', m_convg)
        np.savetxt(f'Data/{algorithm}/{algorithm}Phi_Mat_{N}N_phi{phi0}_chi_{chi}_Time{nstep}_alpha{alpha}.txt', phi_convg)

    elif (Batch=='True'):

        alpha_list = np.linspace(0.0005, 0.005, 10)

        new_phi = phi.copy()
        new_m = m.copy()

        data=open(f'Data/Varied_Alpha/{algorithm}_Evol_{N}N_Varied_Alpha_Time{nstep}.txt','w')

        for i, alpha1 in enumerate(alpha_list):

            alpha1 = np.round(alpha1, 6)
            
            conditions_new = (phi0, chi, algorithm, alpha1, Batch)

            phi_convg, m_convg = func.diff_equation(N, conditions_new, new_phi, new_m, nstep)
            new_phi = phi_convg.copy()
            new_m = m_convg.copy()

            print(f'Complete Simulation @ alpha={alpha1}')

            phi_avg = np.mean(phi_convg)
            phi_var = np.var(phi_convg)

            m_avg = np.mean(m_convg)
            m_var = np.var(m_convg)

            data.write('{0:5.5e} {1:5.5e} {2:5.5e} {3:5.5e} {4:5.5e}\n'.format(alpha1, phi_avg, phi_var, m_avg, m_var))

        data.close()

main()