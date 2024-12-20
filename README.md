# Adaptive Ising Model

A preliminary modification to the Adaptive Ising model proposed in the Lombardi paper to use local connectivity of spins and a hex lattice https://www.nature.com/articles/s43588-023-00410-9.  The Ising Model is a model of magnetism, but can be applied to many fields including neural excitation. The Lombardi paper introduces a feedback element to the model which simulates an inhibitory effect (such as with an inhibitory neuron population). The original paper discusses the effects of the adaptive model in the context of an all-to-all connected neuron population. Our modifications allow us to observe the dynamics in a locally connected model through data and animations. The code available from the Lombardi paper was used as a base and converted into Python code. The code was used to regenerate figures from the paper to test the implementation and verify the results. That code was then modified to support a hex lattice and local connectivity of spins.

# Setup
The environment YML file included should allow easy installation of any dependencies 

``conda env create -f environment.yml``

adaptive_ising.py is the driver code for the simulation, where as simulation_functions.pyx contains the code to create the simulation data. simulation_functions.pyx must be compiled with Cython before running adaptive_ising.py. After installing Cython, run the following command in the command line.

``python setup.py build_ext --inplace``

Then we can run the simulation. To get the animation data and magnetization data, run adaptive_ising.py true.

``python adaptive_ising.py true`` 

Then to generate the animation:

``python movie_maker.py``

simulation_functions.pyx contains options for adjusting the beta and c values, number of nodes to simulate, duration of simulation, etc., which can be played with in adaptive_ising.py.

## All-to-all connectivity slightly above the critical point
https://github.com/user-attachments/assets/cb73673a-92ca-4574-9bc4-42e6bd401abf

## Local connectivity (18 neighbors) slightly above the critical point
https://github.com/user-attachments/assets/e360a2c2-afbf-46ce-b20d-73a2cfc61fe1

## Local connectivity (60 neighbors) slightly above the critical point
https://github.com/user-attachments/assets/0d6b76bf-3bf7-44ae-a497-58dd99f6be44




