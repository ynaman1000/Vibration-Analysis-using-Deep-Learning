# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_anim(err_arr):
    plot_anim.err_arr = err_arr.transpose()
    subplot_rows = 3
    subplot_cols = 2
    num_plots = subplot_rows * subplot_cols

    # initialization function 
    def init(): 
        # creating an empty plot/frame 
        for i in range(num_plots):
            line[i].set_data([], []) 
        return line

    def data_gen():
        # global err_arr
        err_arr = plot_anim.err_arr
        print(err_arr.shape[0])
        for cnt in range(err_arr.shape[0]):
            yield cnt, err_arr[cnt]

    # create a figure with two subplots
    ax = []
    fig, ax = plt.subplots(subplot_rows, subplot_cols)
    # print(len(ax))
    # print(ax)

    # intialize two line objects (one in each axes)
    line = []
    clrs = [['b', 'r'], ['k', 'g'], ['y', 'c']]
    for i in range(subplot_rows):
        for j in range(subplot_cols):
            l, = ax[i][j].plot([], [], lw=2, color=clrs[i][j])
            line.append(l)


    # the same axes initalizations as before (just now we do it for both of them)
    for r in ax:
        for a in r:
            a.set_ylim(0, 100)
            a.set_xlim(0, 5)
            a.grid()

    # initialize the data arrays 
    plot_anim.xdata, plot_anim.ydata = np.zeros([1, 1]), np.zeros([num_plots, 1])
    # print(ydata.shape)
    def run(data):
        # update the data
        t, y = data
        # print(y)
        xdata = np.append(plot_anim.xdata, t)
        plot_anim.xdata = xdata
        # print(y.shape, plot_anim.ydata.shape)
        ydata = np.append(plot_anim.ydata, y.reshape((6, 1)), axis=1)
        plot_anim.ydata = ydata
        # print(y.shape, plot_anim.ydata.shape)
        # print(xdata, ydata)
        # print(xdata.shape, ydata.shape)

        # axis limits checking. Same as before, just for both axes
        for r in ax:
            for a in r:
                xmin, xmax = a.get_xlim()
                if t >= xmax:
                    a.set_xlim(xmin, 2*xmax)
                    a.figure.canvas.draw()

        # update the data of both line objects
        for i in range(num_plots):
            print(ydata[i].shape, xdata.shape)
            line[i].set_data(xdata, ydata[i])
            # print(ydata[i].shape)
            # print(ydata.shape)

        return line

    ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, blit=True, interval=1, repeat=True, save_count=10000, cache_frame_data=False)

    # save the animation as mp4 video file 
    ani.save('anim_subplots.mp4', writer = 'ffmpeg', fps = 1)




# np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf, sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("CUDA Available:", torch.cuda.is_available())

############################################### Loading Data ###############################################

def load_data():
    data_file = open("train_data.csv", 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        for i in range(len(row_list)):
            row_list[i] = float(row_list[i])
        data_list.append(row_list)
    data_file.close()   
    shuffle(data_list)
    return np.array(data_list, dtype = np.float32, ndmin = 2)
    

train_data_frac = 0.9
data_arr  = load_data()
train_ind = int(train_data_frac*data_arr.shape[0])
data_arr = data_arr.transpose()
# print(data_arr)
train_data_input_arr = data_arr[0:1].transpose()
train_data_input_arr = train_data_input_arr[:train_ind]
train_data_output_arr = data_arr[1:].transpose()
train_data_output_arr = train_data_output_arr[:train_ind]
val_data_input_arr = data_arr[0:1].transpose()
val_data_input_arr = val_data_input_arr[train_ind:]
val_data_output_arr = data_arr[1:].transpose()
val_data_output_arr = val_data_output_arr[train_ind:]
# print(train_data_output.shape)
# print(train_data_input.shape)
# print(val_data_input.shape)
# print(val_data_output.shape)
train_data_input = torch.from_numpy(train_data_input_arr).to(device)
train_data_output = torch.from_numpy(train_data_output_arr).to(device)
val_data_input = torch.from_numpy(val_data_input_arr).to(device)
val_data_output = torch.from_numpy(val_data_output_arr).to(device)
# print(train_data_input.size())
# print(train_data_output.size())
# print(val_data_input.size())
# print(val_data_output.size())

############################################################################################################

############################################### Defining Model ###############################################

dim_input = train_data_input.shape[1]
dim_output = train_data_output.shape[1]
no_hidden_layers = 3
hidden_layer_dims  = np.array([110, 110, 110])
learning_rate = 1e-3

""" Neural Network """
model = torch.nn.Sequential(
    torch.nn.Linear(dim_input, hidden_layer_dims[0]),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden_layer_dims[0], hidden_layer_dims[1]),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden_layer_dims[1], hidden_layer_dims[2]),
    torch.nn.PReLU(),
    torch.nn.Linear(hidden_layer_dims[2], dim_output)
)
model.to(device)

""" Loss function """
# loss_fn = torch.nn.MSELoss(reduction='sum')
def loss_fn(pred, act):
    # fun = torch.sum
    # # fun = torch.mean
    # err = torch.add(pred, -act)
    # b_sq_err = torch.mul(err.narrow(1, 0, 3), err.narrow(1, 0, 3))
    # x_sq_err = torch.mul(err.narrow(1, 3, 3), err.narrow(1, 3, 3))
    # l1 = fun(torch.sum(b_sq_err, dim=1, keepdim=True), dim=0)
    # l2 = fun(torch.sum(x_sq_err, dim=1, keepdim=True), dim=0)
    # # print(l1.item(), l2.item())
    # loss = l1 + 100*l2

    # fun = torch.mean
    fun = torch.sum
    err = torch.add(pred, -act)
    per_err = torch.div(err, act)*100
    sq_per_err = torch.mul(per_err, per_err)
    loss = fun(torch.sum(sq_per_err, dim=1, keepdim=True), dim=0)
    # print(fun(sq_per_err, dim=0))

    # fun = torch.mean
    # fun = torch.sum
    # err = torch.add(pred, -act)
    # per_err = torch.div(err, act)*100
    # abs_per_err = torch.abs(per_err)
    # loss = fun(torch.sum(abs_per_err, dim=1, keepdim=True), dim=0)
    # print(fun(sq_per_err, dim=0))
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############################################################################################################


############################################### Training Model ###############################################

start = time.time()
# test = [100, 500, 1000, 5000, 10000]
max_percent_err_arr = np.zeros([dim_output, 1])
test = [5000]    # 5000 is optimum
for i in test:
    for t in range(i):
        # Forward pass: compute predicted output by passing training input to the model.
        train_pred = model(train_data_input)
        # print(train_pred.shape)

        # Compute and print loss.
        train_loss = loss_fn(train_pred, train_data_output)
        # print(t, train_loss.item())

        # zero all of the gradients
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        train_loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        val_pred = model(val_data_input)
        max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
        max_percent_err = max_percent_err.values.detach().cpu().numpy().reshape((1, 6))*100
        # print(max_percent_err.shape, max_percent_err_arr.shape)
        if t%100==99:
            max_percent_err_arr = np.append(max_percent_err_arr, max_percent_err.transpose(), axis=1)
        
        # print(max_percent_err)

    print(max_percent_err_arr.shape)
    plot_anim(max_percent_err_arr)

end = time.time()
print("Total Time taken by GPU: ", end-start)
############################################################################################################
