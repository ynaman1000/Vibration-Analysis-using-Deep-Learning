import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


y = np.ones([6, 100])*np.array([1,2,3,4,5,6]).reshape((6, 1))

subplot_rows = 3
subplot_cols = 2
num_plots = subplot_rows * subplot_cols

# create a figure with subplots
fig, ax = plt.subplots(subplot_rows, subplot_cols)

# intialize line objects (one in each axis)
line = []
clrs = [['k', 'k'], ['k', 'r'], ['r', 'r']]
cnt = 0
for i in range(subplot_rows):
    for j in range(subplot_cols):
        l, = ax[i][j].plot(np.linspace(1, 100, 100).reshape((100)), y[cnt, :].reshape((100)), lw=1, color=clrs[i][j])
        cnt += 1
        line.append(l)

# axes limit settings
for r in ax:
    for a in r:
        a.set_ylim(0, 10)
        a.set_xlim(0, 150)
        a.grid()

plt.savefig("out.png")
plt.show()


# closing current figure
plt.close('all')