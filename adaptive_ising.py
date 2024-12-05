# Nicholas J Uhlhorn
# December 2024
# Adaptive Ising model
# Converted from C code from https://github.com/demartid/stat_mod_ada_nn/blob/main/adaptive_ising.cpp

import numpy as np
import random
from bitstring import BitArray

class Simulation:
    N = 10000
    s = np.ones(N)
    m = 0
    H = 0
    beta = 2
    dt = 1 / N
    c = 0.01

    def casual(self):
        '''Returns a random number in (0,1).'''
        return random.random()

    # init no longer initializes H to 0 as we do so when we create a Simulation.
    def init(self):
        '''initalizes temperature configuration.'''
        for i in range(0, self.N):
            # s[i]=1 #| we already populated s with all ones 
            if(i % 2 == 0):
                self.s[i]=-1
            self.m += self.s[i] / self.N

    def single_update(self, n):
        '''update of the spin n according to the heat bath method'''
        news = 0
        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (self.m + self.H))))

        if (self.casual() < pp):
            news = 1
        else:
            news = -1

        if (news != self.s[n]):
            self.s[n] = -self.s[n]
            self.m += 2.0 * self.s[n] / self.N

        self.H -= self.c * self.m * self.dt

    def update(self):
        '''one sweep over N spins'''
        for i in range(0,self.N):
            n = int(self.N * self.casual())
            self.single_update(n)

    def vectorized_update(self):
        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (self.m + self.H))))

        # update_indexes = (np.random.rand(self.N) * self.N).astype(int)
        random_numbers = np.random.rand(self.N)

        # if we want to flip multiply spin by -1, otherwize 1 to keep the same
        new_spins = np.where(random_numbers < pp, 1, -1)
        changing_spins = np.where(self.s != new_spins)
        self.s = new_spins 
        
        self.m += 2.0 * (np.sum(self.s[changing_spins]) / self.N)
        self.H -= (self.c * self.m * self.dt) * self.N

# Main
random.seed(0)
t = 0
wait = 100
total = 700 #10000000

sim = Simulation()
sim.init()

file = open('simulation_output.txt', 'w')
states_file = open('simulation_spins.bin', 'bw')

print(f"0/{total}", end='\r')
while t <= wait + total:
    if(t > wait):
        file.write(f"{sim.m} {sim.H}\n")
        binary_str_array = np.where(sim.s == -1, 0, sim.s).astype(int).astype(str)
        binary_string = ''.join(binary_str_array)
        binary = BitArray(bin=binary_string)
        binary.tofile(states_file)

    # sim.update()
    sim.vectorized_update()
                             
    t += 1
    if (t % 100 == 0):
        print(f"{t}/{total}", end='\r')

file.close()
