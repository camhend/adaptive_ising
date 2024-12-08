# Nicholas J Uhlhorn & Cameron Henderson 
# December 2024
# cython implementation of simulation code (vecorizing broke the dynamics :( )

import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport rand, RAND_MAX

import cython
@cython.boundscheck(False)
@cython.wraparound(False)

cdef class Simulation:
    cdef int N, total
    cdef int [:] s
    cdef cython.double [:] m_out, H_out
    cdef cython.double H, m, beta, dt, c 

    def __init__(self, int total= 10000, int N=10000, cython.double beta=1, cython.double c=0.01):
        self.total = total
        self.N = N
        self.H = 0.0
        self.s = np.ones(N, dtype=np.int32)
        self.m_out = np.zeros(total, dtype=np.double)
        self.H_out = np.zeros(total, dtype=np.double)
        self.m = 0.0
        self.beta = beta
        self.c = c
        self.dt = 1 / N
    
    cdef casual(self):
        '''Returns a random number in (0,1).'''
        return rand() / RAND_MAX

    cpdef init(self):
        '''initalizes temperature configuration.'''
        cdef int i

        for i in range(0, self.N):
            # s[i]=1 #| we already populated s with all ones 
            if(i % 2 == 0):
                self.s[i]=-1
            self.m += self.s[i] / self.N

    cdef single_update(self, int n):
        '''update of the spin n according to the heat bath method'''
        cdef int news
        cdef cython.double pp

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

    cdef c_update(self):
        cdef int n, i
        '''one sweep over N spins'''
        for i in range(0,self.N):
            n = int(self.N * self.casual())
            self.single_update(n)

    cpdef update(self):
        self.c_update()

    cdef c_one_go(self):
        cdef int n, i, count
        count = 0
        for i in range(0, self.total):
            count += 1
            print(f'{count}/{self.total}', end='\r')
            for j in range(0, self.N):
                n = int(self.N * self.casual())
                self.single_update(n)
            self.m_out[i] = self.m
            self.H_out[i] = self.H
        print(f'{count}/{self.total}: DONE')

    cpdef one_go(self):
        self.c_one_go()

    cpdef get_values_printout(self):
        return f'{self.m} {self.H}\n'

    cpdef get_saved_mH(self):
        return (self.m_out, self.H_out)
    
    cpdef get_s_matrix(self):
        return self.s