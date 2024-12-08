# Nicholas J Uhlhorn & Cameron Henderson
# December 2024
# Adaptive Ising model
# Converted from C code from https://github.com/demartid/stat_mod_ada_nn/blob/main/adaptive_ising.cpp

import sys
import numpy as np
import random
from bitstring import BitArray
import simulation_functions as sf
import old_sim_functions as osf

# Main
random.seed(0)
t = 0
wait = 100
total = 700 #10000000

save_s = False
if len(sys.argv) > 1:
    save_s = sys.argv[1].lower() == 'true'
if len(sys.argv) > 2:
    total = int(sys.argv[2])
print(f'save_s:{save_s} | total:{total}')

# sim = sf.Simulation()
sim = osf.Simulation(total, N=250)
sim.init()

file = open('simulation_output.txt', 'w')
states_file = open('simulation_spins.bin', 'bw')
# print(f"0/{total}", end='\r')
# while t <= wait + total:
#     if(t > wait):
#         file.write(sim.get_values_printout())

#         if save_s:
#             s_flat = sim.s.flatten()
#             binary_str_array = np.where(s_flat == -1, 0, s_flat).astype(int).astype(str)
#             binary_string = ''.join(binary_str_array)
#             binary = BitArray(bin=binary_string)
#             binary.tofile(states_file)

#     sim.update()
#     # sim.vectorized_update()
                             
#     t += 1
#     if (t % 100 == 0):
#         print(f"{t}/{total}", end='\r')
sim.one_go()
(m, H) = sim.get_saved_mH()
for i in range(len(m)):
    file.write(f"{m[i]} {H[i]}\n")

file.close()
