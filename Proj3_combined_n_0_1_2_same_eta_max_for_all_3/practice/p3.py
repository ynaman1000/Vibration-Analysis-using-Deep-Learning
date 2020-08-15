import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initialization function 
def init(): 
    # creating an empty plot/frame 
    for i in range(6):
        line[i].set_data([], []) 
    return line

def data_gen():
    t = data_gen.t
    cnt = 0
    while cnt < 1000:
        cnt+=1
        t += 0.05
        y = np.sin(2*np.pi*t) * np.exp(-t/10.)*np.ones([6, 1])
        # adapted the data generator to yield both sin and cos
        yield t, y

data_gen.t = 0

# create a figure with two subplots
ax = []
fig, ax = plt.subplots(3,2)
# print(len(ax))
# print(ax)

# intialize two line objects (one in each axes)
line = []
clrs = [['b', 'r'], ['k', 'g'], ['y', 'c']]
for i in range(3):
    for j in range(2):
        l, = ax[i][j].plot([], [], lw=2, color=clrs[i][j])
        line.append(l)


# the same axes initalizations as before (just now we do it for both of them)
for r in ax:
    for a in r:
        a.set_ylim(-1.1, 1.1)
        a.set_xlim(0, 5)
        a.grid()

# initialize the data arrays 
xdata, ydata = np.zeros([1, 1]), np.zeros([6, 1])
print(ydata.shape)
def run(data):
    # update the data
    t, y = data
    # print(y)
    global xdata
    global ydata
    xdata = np.append(xdata, t)
    # print(y.shape)
    ydata = np.append(ydata, y, axis=1)
    print(xdata, ydata)

    # axis limits checking. Same as before, just for both axes
    for r in ax:
        for a in r:
            xmin, xmax = a.get_xlim()
            if t >= xmax:
                a.set_xlim(xmin, 2*xmax)
                a.figure.canvas.draw()

    # update the data of both line objects
    for i in range(6):
        print(ydata[i].shape, xdata.shape)
        line[i].set_data(xdata, ydata[i])
        # print(ydata[i].shape)
        # print(ydata.shape)

    return line

ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, blit=True, interval=10,
    repeat=False)

# save the animation as mp4 video file 
ani.save('anim_subplots.mp4', writer = 'ffmpeg', fps = 30) 

plt.show()