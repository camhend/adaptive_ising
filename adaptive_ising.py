# Nicholas J Uhlhorn
# December 2024
# Adaptive Ising model
# Converted from C code from https://github.com/demartid/stat_mod_ada_nn/blob/main/adaptive_ising.cpp

import sys
import numpy as np
import random
from bitstring import BitArray
from lattice import *

class Simulation:
    N = 10000
    shape = (100,100)
    s = np.ones(shape)
    H = np.zeros(shape)
    # H = 0
    P = hex_coord(shape) # hex grid positions
    m = 0
    beta = 0.025
    dt = 1 / N
    c = 0.01
    even_filter = gen_filter(7, center_row_is_odd=False)
    odd_filter = gen_filter(7, center_row_is_odd=True)

    def casual(self):
        '''Returns a random number in (0,1).'''
        return random.random()

    # init no longer initializes H to 0 as we do so when we create a Simulation.
    def init(self):
        '''initalizes temperature configuration.'''
        for idx in np.ndindex(self.shape):
            # s[idx]=1 #| we already populated s with all ones 
            if(idx[0] % 2 == 0):
                self.s[idx]=-1
            self.m += self.s[idx] / self.N

    def single_update(self, n: tuple):
        '''update of the spin n according to the heat bath method'''
        news = 0
        # find sum of states m for all neighbors
        m = 0 
        filter = None
        if (n[1] % 2 == 0):
            filter = self.even_filter
        else:
            filter = self.odd_filter

        filter_center = np.array((len(filter) // 2, len(filter) // 2 ))
        n_vector = np.array(n)
        for idx in np.ndindex(filter.shape):
            idx = np.array(idx)
            (x,y) = (n_vector + idx - filter_center)
            (x,y) = (x % self.shape[0], y % self.shape[1])
            m += self.s[(x,y)]

        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (m + self.H[n]))))

        if (self.casual() < pp):
            news = 1
        else:
            news = -1

        if (news != self.s[n]):
            self.s[n] = -self.s[n]
            self.m += 2.0 * self.s[n] / self.N

        num_neighbors = filter.sum()
        self.H[n] -= self.c * (m / num_neighbors) * self.dt

    def update(self):
        '''one sweep over N spins'''
        for i in range(0,self.N):
            n = int(self.N * self.casual())
            n = ((n // self.shape[0]), (n % self.shape[1])) # convert to 2D tuple index
            self.single_update(n)

    def vectorized_update(self):
        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (self.m + self.H))))

        # update_indexes = (np.random.rand(self.N) * self.N).astype(int)
        random_numbers = np.random.rand(*self.shape)

        # if we want to flip multiply spin by -1, otherwize 1 to keep the same
        new_spins = np.where(random_numbers < pp, 1, -1)
        changing_spins = np.where(self.s != new_spins, 1, 0)
        self.s = new_spins 
        
        self.m += 2.0 * (np.sum(np.multiply(self.s, changing_spins)) / self.N)
        self.H -= (self.c * self.m * self.dt) * self.N

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

sim = Simulation()
sim.init()

file = open('simulation_output.txt', 'w')
states_file = open('simulation_spins.bin', 'bw')


print(f"0/{total}", end='\r')
while t <= wait + total:
    if(t > wait):
        file.write(f"{sim.m} {sim.H}\n")

        s_flat = sim.s.flatten()
        if save_s:
            binary_str_array = np.where(s_flat == -1, 0, s_flat).astype(int).astype(str)
            binary_string = ''.join(binary_str_array)
            binary = BitArray(bin=binary_string)
            binary.tofile(states_file)

    sim.update()
    # sim.vectorized_update()
                             
    t += 1
    if (t % 100 == 0):
        print(f"{t}/{total}", end='\r')

file.close()
