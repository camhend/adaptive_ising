# Nicholas J Uhlhorn
# December 2024
# Creates a movie from a binary file given the size of each frame

import bitstring
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bitstring import BitStream

frame_size = 10000
img_dimentions = (100,100)
fps = 30
bytes_size = np.ceil(frame_size / 8).astype(int)

bin_file = open('simulation_spins.bin', 'br')
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
        print('.', end='', flush=True)

    im.set_array(frames[i])
    return [im]
print()

anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames = len(frames),
    interval = 1000 / fps,
)

print("Saving Movie")
writer_video = animation.FFMpegFileWriter(fps=fps)
anim.save('movie.mp4', writer=writer_video)
