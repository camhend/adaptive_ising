# Nicholas J Uhlhorn
# December 2024
# Creates a movie from a binary file given the size of each frame

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bitstring import BitStream

def hex_coord(data, radius):
    shape = np.shape(data)

    horizontal_spacing = radius * np.sqrt(3)
    vertical_spacing = radius * (3/2)
    shift = radius * .5
    points = []
    # shift odd rows to the right
    for idx in np.ndindex(shape):
        if not data[idx]:
            continue
        (x,y) = idx
        if y % 2 == 1:
            points.append([(x + shift) * horizontal_spacing, y * vertical_spacing])
        else: 
            points.append([x * horizontal_spacing, y * vertical_spacing])
    return np.vstack(points)

frame_size = 4096
img_dimentions = (64,64)
fps = 30
bytes_size = np.ceil(frame_size / 8).astype(int)

bin_path = 'simulation_spins.bin'
if len(sys.argv) > 1:
    bin_path = sys.argv[1]

print(f'bin_path:{bin_path}')

bin_file = open(bin_path, 'br')
frames = []

print("Reading bits")

raw_bit_string = bin_file.read(bytes_size)
counter = 0
while raw_bit_string:
    if counter % 100 == 0:
        print('.', end='', flush=True)
    counter += 1
    bit_stream = BitStream(raw_bit_string) 
    bin_string = bit_stream.read('bin')
    frames.append(np.array(list(bin_string), dtype=int).reshape(img_dimentions))
    raw_bit_string = bin_file.read(bytes_size)
print()

print("Generating Frames")
fig = plt.figure(figsize=(8,8))
a = frames[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

def animate_func(i):
    if i % fps == 0:
        print(f'{i}/{len(frames)}', end='\r', flush=True)

    im.set_array(frames[i])#[:,0], frames[i][:,1])
    return [im]
print()

anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames = len(frames),
    interval = 1000 / fps,
)

writer_video = animation.FFMpegFileWriter(fps=fps)
anim.save('movie.mp4', writer=writer_video)
print(flush=False)
