# Nicholas J Uhlhorn & Cameron Henderson 
# December 2024
# cython implementation of simulation code (vecorizing broke the dynamics :( )

import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport rand, RAND_MAX
import random

# Return a matrix that contains a circular filter on a hex grid
# The shape of the filter after transformation onto a hex grid 
# depends on whether the center of the filter is in an even or odd row. 
# Assumes filter is in a hex grid where odd rows are shifted to the right.
def gen_filter(length: int, center_row_is_odd: bool): 
    filter = np.zeros((length, length))
    pos = hex_coord(filter.shape)
    center = (length // 2, length // 2)
    connect_radius = 1 * (3/2) * (length // 2) + .0001 # extra decimal to cover for rounding error
    for idx in np.ndindex(filter.shape):
        if np.array_equal(idx,center): # skip center
            continue 
        if (np.linalg.norm(pos[idx] - pos[center]) <= connect_radius): 
             filter[idx] = 1

    # fix rows if center is on an even row
    if (not center_row_is_odd):
        for idx, y in enumerate(filter.T):
            if (idx % 2 == 0):
                filter.T[idx] = np.roll(y, -1)
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

# Create a connectivity matrix with local connectivity.
# Connections are made using the filter from gen_filter.
# grid_shape: Shape of grid. The connectivity matrix will be a 
#             a square of sidelength grid_shape[0] * grid_shape[1]
# filter_size: Length of the filter matrix. The circular filter 
#              is bounded within this matrix.
def gen_connectivity_matrix(grid_shape: tuple, filter_size: int):
    print("Generating connectivity matrix...")
    C = np.zeros( (np.prod(grid_shape), np.prod(grid_shape)) )
    xlim, ylim = grid_shape
    even_filter = gen_filter(filter_size, center_row_is_odd=False)
    odd_filter = gen_filter(filter_size, center_row_is_odd=True)
    filter = None
    for grid_idx in np.ndindex(grid_shape):
        if (grid_idx[1] % 2 == 0): 
            filter = even_filter
        else:
            filter = odd_filter
        filter_center = np.array((len(filter) // 2, len(filter) // 2 ))
        grid_idx = np.array(grid_idx)
        C_row = grid_idx[0] * xlim + grid_idx[1]
        for filter_idx in np.ndindex(filter.shape):
            if filter[filter_idx] == 1: 
                filter_idx = np.array(filter_idx)
                (x,y) = (grid_idx + filter_idx - filter_center)
                (x,y) = (x % xlim, y % ylim)
                C_col = x * xlim + y
                C[C_row, C_col] = 1
    return C

cdef class Simulation:
    cdef int N, shape_x, shape_y, total
    cdef np.ndarray s, h, m_out, H_out, connectivity_matrix, num_neighbors
    cdef cython.double H, m, beta, dt, c 

    def __init__(self, int total= 1000, int N=10000, int shape_x=100, int shape_y=100, cython.double beta=0.025, cython.double c=0.01, int filter_length=7):
        assert(shape_x * shape_y == N,'Simulation N value should equal shape_x * shape_y')

        self.total = total
        self.N = N
        self.H = 0
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.s = np.ones(N)
        self.h = np.zeros(N)
        self.m = 0.0
        self.beta = beta
        self.c = c
        self.dt = 1 / N
        self.m_out = np.zeros(total, dtype=np.double)
        self.H_out = np.zeros(total, dtype=np.double)
        self.connectivity_matrix = gen_connectivity_matrix((shape_x, shape_y), filter_length) # locally connected
        # self.connectivity_matrix = np.ones((N, N)) # fully connected
        self.num_neighbors = np.sum(self.connectivity_matrix, axis=0)

    def init(self):
        '''initalizes temperature configuration.'''
        print("Number of neighbors: ", self.num_neighbors[0])
        for i in range(0, self.N):
            # s[i]=1 #| we already populated s with all ones 
            if(i % 2 == 0):
                self.s[i]=-1
            self.m += self.s[i] / self.N

    cdef single_update(self, int n):
        '''update of the spin n according to the heat bath method'''
        cdef int news 
        cdef cython.double m 
        news = 0
        # find sum of states s for all neighbors
        m = np.dot(self.s, self.connectivity_matrix[n]) / self.num_neighbors[n]

        pp = 1 / (1 + (1 / np.exp(2 * self.beta * (m + self.H)))) # locally connected
        # pp = 1 / (1 + (1 / np.exp(2 * self.beta * (self.m + self.H)))) # fully connected

        if (random.random() < pp):
            news = 1
        else:
            news = -1

        if (news != self.s[n]):
            self.s[n] = -self.s[n]
            self.m += 2.0 * self.s[n] / self.N
            m += 2.0 * self.s[n] / self.num_neighbors[n]

        self.H -= self.c * m * self.dt

    cdef c_update(self):
        '''one sweep over N spins'''
        rng = np.random.default_rng()
        rand_indexes = rng.integers(low=0, high=self.N, size=self.N)
        for i in rand_indexes:
            self.single_update(i)

    cpdef update(self):
        self.c_update()

    cdef c_one_go(self):
        cdef int n, i, count
        count = 0
        print(self.total)
        for i in range(0, self.total):
            count += 1
            print(f'{count}/{self.total}', end='\r')
            rng = np.random.default_rng()
            rand_indexes = rng.integers(low=0, high=self.N, size=self.N)
            for j in rand_indexes:
                self.single_update(j)
            self.m_out[i] = self.m
            self.H_out[i] = self.H
        print(f'{count}/{self.total}: DONE')

    cpdef one_go(self):
        self.c_one_go()

    def get_values_printout(self):
        return f'{self.m} {self.H}\n'
    
    def get_s_matrix(self):
        return self.s

    cpdef get_saved_mH(self):
        return (self.m_out, self.H_out)
    