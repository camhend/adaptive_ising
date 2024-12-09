# Nicholas J Uhlhorn
# December 2024
# Generate 1c and 1d (todo) figures given data files

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

bin_count = 10000
save_path = 'figure_c1.png'
data_paths = ['simulation_output.txt']

if len(sys.argv) > 1:
    bin_count = int(sys.argv[1])
if len(sys.argv) > 2:
    save_path = sys.argv[2]
if len(sys.argv) > 3:
    data_paths = sys.argv[3:]
print(f"bin_count:{bin_count} | save_path:{save_path}\ndata_paths:{data_paths}")

m_lists = []

for data_path in data_paths:
    print(f'Reading Raw Data of {data_path}')

    m_list = np.loadtxt(data_path, delimiter=' ')[:,0]
    # m_list = []
    # h_list = []
    # count = 0
    # for line in raw_data:
    #     print(count, end='\r', flush=True)
    #     tokens = line.split(' ')
    #     m_list.append(float(tokens[0]))
    #     # h_list.append(float(tokens[1]))
    #     count += 1
    # print(count)
    m_lists.append(m_list)

# Calculate a_0 and P(a_0)
# a_0 is area under m(t) between two zero-crossing events
# P(a_0) is the distrabution of the zero-crossing area
a_0_lists = []

for i, m_list in enumerate(m_lists):
    signs = np.sign(m_list)
    signs_offset = np.roll(signs, -1)
    signs_offset[-1] = signs[-1]

    zero_crossings = np.append([0], np.where(signs != signs_offset))

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

p_of_a_0s_lists = []

for i, a_0s in enumerate(a_0_lists):
    print(f'Building Distribution {i+1}')

    # samples = a_0s[np.random.choice(a_0s.shape[0], 1000, replace=False)]

    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2, algorithm='kd_tree', atol=1e-3, rtol=1e-2).fit(samples.reshape(-1,1))
    # print("Scoring values")
    # p_of_a_0s = np.exp(kde.score_samples(a_0s.reshape(-1,1)))

    hist, bin_edges = np.histogram(a_0s, bin_count)
    bin_edges[0] -= 1
    bin_edges[-1] += 1

    bin_indexes = np.subtract(np.digitize(a_0s, bin_edges, True), 1)
    hist_probabilities = np.divide(hist, len(a_0s))

    p_of_a_0s = hist_probabilities[bin_indexes]
    p_of_a_0s_lists.append(p_of_a_0s)

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

print(f"Saving Figure as {save_path}")
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
leg = plt.legend(framealpha=1, fontsize=12, markerscale=2.5)
for lh in leg.legend_handles: 
    lh.set_alpha(1)

plt.savefig(save_path)
