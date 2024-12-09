# Nicholas J Uhlhorn & Cameron Henderson 
# December 2024
# cython implementation of simulation code (vecorizing broke the dynamics :( )

import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport rand, RAND_MAX

# return a matrix that contains a circular filter on a hex grid
# The shape of the filter after transformation onto a hex grid 
# depends on whether the center of the filter is in an even or odd row. 
# Assumes filter is in a hex grid where odd rows are shifted to the right.
def gen_filter(length: int, center_row_is_odd: bool): 
    filter = np.zeros((length, length))
    norm = np.zeros((length, length, 2))
    pos = hex_coord(filter.shape)
    center = (length // 2, length // 2)
    for idx in np.ndindex(filter.shape):
        if np.array_equal(idx,center): # skip center
            continue 
        connect_radius = 1 * (3/2) * (length // 2) + .0001 # extra decimal to cover for rounding error
        if (np.linalg.norm(pos[idx] - pos[center]) <= connect_radius): 
             filter[idx] = 1
        norm[idx] = pos[idx] - pos[center]

    # fix rows if center is on an even row
    if (not center_row_is_odd):
        for row in filter:
            row[0: len(row) - 1] = row[1: (len(row))]
            row[-1] = 0
    return filter

# return a matrix of 2D grid coordinates for creating a hex lattice
# each element in the matrix corresponds to a position to place a hexagon
# The radius of the bounding circle around a hexagon is 1.
# Odd rows are shifted to the right.
def hex_coord(shape: tuple):
    size = 1 # radius of bounding circle
    horizontal_spacing = size * np.sqrt(3)
    vertical_spacing = size * (3/2)
    shift = size * .5
    grid = np.zeros((*shape,2))
    # shift odd rows to the right
    for idx in np.ndindex(shape):
        (x,y) = idx
        if y % 2 == 1:
            grid[idx] = [(x + shift), y]
        else: 
            grid[idx] = [x, y]
    # scale X and Y positions to fit hex lattice
    for idx in np.ndindex(shape):
        grid[idx][0] *= horizontal_spacing
        grid[idx][1] *= vertical_spacing
    return grid

cdef class Simulation:
    cdef int N, total, shape_x, shape_y
    cdef np.ndarray s, h, P, even_filter, odd_filter
    cdef cython.double [:] m_out, H_out
    cdef cython.double H, m, beta, dt, c 

    def __init__(self, int total=10000, int N=10000, int shape_x=100, int shape_y=100, cython.double beta=0.025, cython.double c=0.01, int filter_length=7):
        assert(shape_x * shape_y == N,'Simulation N value should equal shape_x * shape_y')

        self.total = total
        self.N = N    
        cdef cython.double [:] m_out, H_out
        self.H = 0
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.s = np.ones((shape_x, shape_y))
        self.h = np.zeros((shape_x, shape_y))
        self.P = hex_coord((shape_x, shape_y))
        self.m = 0.0
        self.beta = beta
        self.c = c
        self.dt = 1 / N
        self.even_filter = gen_filter(filter_length, center_row_is_odd=False)
        self.odd_filter = gen_filter(filter_length, center_row_is_odd=True)
        self.m_out = np.zeros(total, dtype=np.double)
        self.H_out = np.zeros(total, dtype=np.double)

    cdef casual(self):
        '''Returns a random number in (0,1).'''
        return rand() / RAND_MAX

    # init no longer initializes H to 0 as we do so when we create a Simulation.
    def init(self):
        '''initalizes temperature configuration.'''
        for idx in np.ndindex((self.shape_x, self.shape_y)):
            # s[idx]=1 #| we already populated s with all ones 
            if(idx[0] % 2 == 0):
                self.s[idx]=-1
            self.m += self.s[idx] / self.N

    cdef single_update(self, int n_x, int n_y):
        '''update of the spin n according to the heat bath method'''
        cdef int news, m 
        cdef np.ndarray sim_filter, filter_center, n_vector

        news = 0
        # find sum of states m for all neighbors
        m = 0 
        sim_filter = np.empty(0)
        if (n_y % 2 == 0):
            sim_filter = self.even_filter
        else:
            sim_filter = self.odd_filter

        sim_filter_center = np.array((len(sim_filter) // 2, len(sim_filter) // 2 ))
        n_vector = np.array((n_x, n_y))
        for idx in np.ndindex(np.shape(sim_filter)):
            idx = np.array(idx)
            (x,y) = (n_vector + idx - sim_filter_center)
            (x,y) = (x % self.shape_x, y % self.shape_y)
            m += self.s[(x,y)]

        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (m + self.h[(n_x, n_y)]))))

        if (self.casual() < pp):
            news = 1
        else:
            news = -1

        if (news != self.s[(n_x, n_y)]):
            self.s[(n_x, n_y)] = -self.s[(n_x, n_y)]
            self.m += 2.0 * self.s[(n_x, n_y)] / self.N

        num_neighbors = sim_filter.sum()
        self.h[(n_x, n_y)] -= self.c * (m / num_neighbors) * self.dt

    cdef c_update(self):
        '''one sweep over N spins'''
        for i in range(0,self.N):
            n = int(self.N * self.casual())
            n_x = (n // self.shape_x) # convert to 2D tuple index
            n_y = (n % self.shape_y) # convert to 2D tuple index
            self.single_update(n_x, n_y)

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
                n_x = n // self.shape_x # convert to 2D tuple index
                n_y = n % self.shape_y # convert to 2D tuple index
                self.single_update(n_x, n_y)
            self.m_out[i] = self.m
            self.H_out[i] = self.H
        print(f'{count}/{self.total}: DONE')

    cpdef one_go(self):
        self.c_one_go()

    def get_values_printout(self):
        return f'{self.m} {self.H}\n'
    
    cpdef get_saved_mH(self):
        return (self.m_out, self.H_out)
    
    def get_s_matrix(self):
        return self.s