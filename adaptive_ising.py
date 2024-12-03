# Nicholas J Uhlhorn
# December 2024
# Adaptive Ising model
# Converted from C code from https://github.com/demartid/stat_mod_ada_nn/blob/main/adaptive_ising.cpp

import numpy as np
import random

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

# Main
random.seed(0)
t = 0
wait = 100
total = 10000000

sim = Simulation()

sim.init()

while t <= wait + total:
    if(t > wait):
        print(sim.m, sim.H)
    sim.update()
    t += 1
