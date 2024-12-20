# Nicholas J Uhlhorn
# December 2024
# Generate 1c and 1d (todo) figures given data files

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

bin_count = 1000
save_path_1c = 'figure_1c.png'
save_path_1d = 'figure_1d.png'
data_paths = ['simulation_output.txt']

if len(sys.argv) > 1:
    bin_count = int(sys.argv[1])
if len(sys.argv) > 2:
    save_path_1c = sys.argv[2]
if len(sys.argv) > 3:
    save_path_1d = sys.argv[3]
if len(sys.argv) > 4:
    data_paths = sys.argv[4:]
print(f"bin_count:{bin_count} | save_path_1c:{save_path_1c} | save_path_1d:{save_path_1d}\ndata_paths:{data_paths}")
print(save_path_1d[-3:])
assert not (save_path_1c[-3:] == 'txt') and not (save_path_1d[-3:] == 'txt'), 'DO NOT SAVE OVER TEXT FILES (SAFTEY FOR DATA)'

m_lists = []

for data_path in data_paths:
    print(f'Reading Raw Data of {data_path}')

    m_list = np.loadtxt(data_path, delimiter=' ')[:,0]
    m_lists.append(m_list)

# Calculate a_0 and P(a_0)
# a_0 is area under m(t) between two zero-crossing events
# P(a_0) is the distrabution of the zero-crossing area
a_0_lists = []
t_lists = []

for i, m_list in enumerate(m_lists):
    signs = np.sign(m_list)
    signs_offset = np.roll(signs, -1)
    signs_offset[-1] = signs[-1]

    zero_crossings = np.append([0], np.where(signs != signs_offset))
    ts = np.diff(zero_crossings)

    print(f"Summing Areas Under Curve for list {i+1}")
    a_0s = []
    last_index = 0
    count = 0

    for index in zero_crossings:
        if index == 0:
            continue
        print(count, end='\r', flush=True)
        a_0s.append(np.sum(m_list[last_index:index])) 
        last_index = index
        count += 1
    print(count)
    a_0s.append(np.sum(m_list[last_index:])) 

    a_0s = np.array(a_0s)
    a_0s = a_0s[np.where(a_0s >= 0)]
    a_0_lists.append(a_0s)
    t_lists.append(ts)

p_of_a_0s_lists = []
for i, a_0s in enumerate(a_0_lists):
    print(f'Building a_0 Distribution {i+1}')

    hist, bin_edges = np.histogram(a_0s, bin_count)
    bin_edges[0] -= 1
    bin_edges[-1] += 1

    bin_indexes = np.subtract(np.digitize(a_0s, bin_edges, True), 1)
    hist_probabilities = np.divide(hist, len(a_0s))

    p_of_a_0s = hist_probabilities[bin_indexes]
    p_of_a_0s_lists.append(p_of_a_0s)

p_of_t_lists = []
for i, ts in enumerate(t_lists):
    print(f'Building t Distribution {i+1}')

    hist, bin_edges = np.histogram(ts, bin_count)
    bin_edges[0] -= 1
    bin_edges[-1] += 1

    bin_indexes = np.subtract(np.digitize(ts, bin_edges, True), 1)
    hist_probabilities = np.divide(hist, len(ts))

    p_of_t = hist_probabilities[bin_indexes]
    p_of_t_lists.append(p_of_t)

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

print(f"Saving Figure as {save_path_1c}")
plt.figure(figsize=(6,4.5))

plt.yscale('log')
plt.xscale('log')
plt.xlim(0.01, 2000)
plt.ylim(0.0000001, 10)

labels = [r'$\beta=1$', r'$\beta=0.99$', r'$\beta=0.9$', r'$\beta=0.7$']
colors = ['black', 'red', 'blue', 'yellow']
for i in range(len(a_0_lists)):
    plt.scatter(a_0_lists[i], p_of_a_0s_lists[i], s=0.1, alpha=0.25, label=labels[i], c=colors[i])

plt.xlabel('$a_0$')
plt.ylabel('$P(a_0)$')
leg = plt.legend(framealpha=1, fontsize=12, markerscale=2.5, loc='upper right')
for lh in leg.legend_handles: 
    lh.set_alpha(1)

plt.savefig(save_path_1c)

print(f"Saving Figure as {save_path_1d}")
plt.figure(figsize=(6,4.5))

plt.yscale('log')
plt.xscale('log')
plt.xlim(0.1, 20000)
plt.ylim(0.00001, 1)

labels = [r'$\beta=1$', r'$\beta=0.99$', r'$\beta=0.9$', r'$\beta=0.7$']
colors = ['black', 'red', 'blue', 'yellow']
for i in range(len(a_0_lists)):
    plt.scatter(a_0_lists[i], p_of_a_0s_lists[i], s=0.1, alpha=0.25, label=labels[i], c=colors[i])

plt.xlabel('$t$')
plt.ylabel('$P(t)$')
leg = plt.legend(framealpha=1, fontsize=12, markerscale=2.5, loc='upper right')
for lh in leg.legend_handles: 
    lh.set_alpha(1)

plt.savefig(save_path_1d)