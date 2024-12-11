# CSCI597_Final_Project

A preliminary modification to the Adaptive Ising model proposed in the Lombardi paper to use local connectivity of spins and a hex lattice. https://www.nature.com/articles/s43588-023-00410-9
The code available from the Lombardi paper was used as a base and converted into Python code. The code was used to regenerate figures from the paper to test the implementation and verify the results. That code was then modified to support a hex lattice and local connectivity of spins.

adaptive_ising.py is the driver code for the simulation, where as simulation_functions.pyx contains the code to create the simulation data. simulation_functions.pyx must be compiled with Cython before running adaptive_ising.py. After installing Cython, run the following command in the command line.

``python setup.py build_ext --inplace``

Then we can run the simulation. To get the animation data and magnetization data, run adaptive_ising.py true.

``python adaptive_ising.py true`` 

Then to generate the animation:

``movie_maker.py``

simulation_functions.pyx contains options for adjusting the beta and c values, number of nodes to simulate, duration of simulation, etc., which can be played with in adaptive_ising.py.
