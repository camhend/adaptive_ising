# Nicholas J Uhlhorn
# December 2024
# Generate 1c and 1d (todo) figures given data files

import sys
import numpy as np
import matplotlib.pyplot as plt

bin_count = 10000
data_path = 'simulation_output.txt'
save_path = 'figure_c1.png'

if len(sys.argv) > 1:
    bin_count = int(sys.argv[1])
if len(sys.argv) > 2:
    data_path = sys.argv[2]
if len(sys.argv) > 3:
    save_path = sys.argv[3]
print(f"bin_count:{bin_count} | data_path:{data_path} | save_path:{save_path}")

data_file = open(data_path, 'r')
raw_data = data_file.readlines()

print('Reading Raw Data')
m_list = []
h_list = []
count = 0
for line in raw_data:
    if count % 10000 == 0:
        print(count, end='\r', flush=True)
    tokens = line.split(' ')
    m_list.append(float(tokens[0]))
    h_list.append(float(tokens[1]))
    count += 1
print(count)

# Calculate a_0 and P(a_0)
# a_0 is area under m(t) between two zero-crossing events
# P(a_0) is the distrabution of the zero-crossing area

signs = np.sign(m_list)
signs_offset = np.roll(signs, -1)
signs_offset[-1] = signs[-1]

zero_crossings = np.append([0], np.where(signs != signs_offset))
zero_crossings_offset = np.roll(zero_crossings, -1)
zero_crossings_offset[-1] = len(signs) - 1

print("Summing Areas Under Curve")
a_0s = []
last_index = 0
count = 0

for index in zero_crossings:
    if count % 10000 == 0:
        print(count, end='\r', flush=True)
    a_0s.append(np.sum(m_list[last_index:index])) 
    last_index = index
    count += 1
a_0s.append(np.sum(m_list[last_index:])) 

a_0s = np.array(a_0s)
a_0s = a_0s[np.where(a_0s >= 0)]

print('Calculating Distribution')
hist, bin_edges = np.histogram(a_0s, bin_count)
bin_edges[0] -= 1
bin_edges[-1] += 1

bin_indexes = np.subtract(np.digitize(a_0s, bin_edges, True), 1)
hist_probabilities = np.divide(hist, len(a_0s))

p_of_a_0s = hist_probabilities[bin_indexes]

print(f"Saving Figure as {save_path}")
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.001, 1500)
plt.ylim(0.0000001, 10)
plt.scatter(a_0s, p_of_a_0s, s=0.1, alpha=0.1)
plt.savefig(save_path)
