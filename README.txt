INSTRUCTIONS

To run checkpoint 3 we require conditions that the correct directories are created
for output data and plots to be stored.


The simulation conditions are as follows:

1 -- There are directories /Data/ and /Plots/ in the same directory as the IVP_Run_new.py, 
     IVP_Functions_new.py and IVP_PlotData.py files.

     This is nessesary as data is stored to the files in these directories.
     The names of these directories are case sensitive.

FURTHER CODEBASE EXPLANATION 

PYTHON FILES 

----    The IVP_Run_new.py runs the checkpoint 3 code. The following parameters must be 
	specified in terminal when running the code in this order.

        'python IVP_Run_new.py N phi algorithm field'

        Parameters:
            N - axis length of square cube used in simulation
            phi - order parmeter for the cahn-hilliard equation
            algorithm - must specify either 'hilliard' or 'jacobi' or 'seidel'
            field - must specify either 'electric' or 'magnetic'


            The field parameter runs the simulation for either a point charge (electric parameter) or
 	    a charged wire (magnetic parameter).

            example: 'python IVP_Run_new.py 50 0.0 jacobi electric'
                      This would run the jacobi simulation of a point charge in a 3D cube
		      until converged.


----    The IVP_PlotData.py file uses the stored data files in the pre-simulated data directory /Data_good/ to 
	plot graphs of vector fields, potential field strength, contour plots, hilliard free energy convergence
	and Gauss-Seidel SORS omega convergence. The plotting is controlled by a main() function located at the 
	bottom of the file.



----    The IVP_Functions_new.py contains all the functions used in checkpoint 3.



DATA AND GRAPHS FOR MARKING


----    All raw data used in plotting is stored in the directory /Data_good/. The filenames are labeled as 'electric' or 'magnetic'
	which corresponds to the point charge or the charged wire. The following data files contain:

	hilliard files contains 1st coloumn - Time/sweeps and 2nd coloumn - free energy density
	jacobi files contains converged 3D matrix of charged system (either point charge or charged wire)
	seidel file contains 1st coloumn - omega value used and 2nd coloumn - number of iterations for convergence
	RadiusDep files contain radial field strengths of converged 3D systems: 1st coloumn - radius from center charge, 2nd coloumn 
	- potential strength, 3rd coloumn - (electric or magnetic) field strength



----    The plotted graphs for grading can be located within the directory called /Plots/
