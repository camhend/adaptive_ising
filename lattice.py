# Cameron Henderson
# December 2024
# Generate nearest neighbor filter for hex lattice

import numpy as np

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