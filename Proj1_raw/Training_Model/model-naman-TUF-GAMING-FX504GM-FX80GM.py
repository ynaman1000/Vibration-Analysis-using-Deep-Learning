# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

############################################### Manual Setting ###############################################

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf, sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("CUDA Available:", torch.cuda.is_available())

dim_input = 1
dim_output = 6
train_data_frac = 0.9

learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
num_layers = [2, 3, 4]
num_units = [30, 50, 70, 90, 110]
num_iter = [500, 1000, 1500]
# num_iter = [1,2,3]
# num_iter = [9000]
isSum = [True, False]       # is the loss function 'sum' or 'mean'
isAbs = [True, False]       # is the error 'absolute' or 'squared'
isPer = [True, False]       # 'Percent error' or 'Regularized error'
Reg = [1, 10, 100]          # regulariztions
do_anim = False             # for making animation video
# do_anim = True
if do_anim:
    fps = 1               # frames per second in animation, error after every fps iteration will be plotted
    num_subplot_rows = 3    # number of rows of subplot array in animation
    num_subplot_cols = 2    # number of columns of subplot array in animation

############################################################################################################



############################################### Loading Data ###############################################

# function for loading data
def load_data():
    ''' Returns numpy array of shape (total data points, 7) '''
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


data_arr  = load_data()
train_ind = int(train_data_frac*data_arr.shape[0])
data_arr = data_arr.transpose()

# defining data variables as numpy arrays
train_data_input_arr = data_arr[0:1].transpose()
train_data_input_arr = train_data_input_arr[:train_ind]
train_data_output_arr = data_arr[1:].transpose()
train_data_output_arr = train_data_output_arr[:train_ind]
val_data_input_arr = data_arr[0:1].transpose()
val_data_input_arr = val_data_input_arr[train_ind:]
val_data_output_arr = data_arr[1:].transpose()
val_data_output_arr = val_data_output_arr[train_ind:]

# converting data variables to torch tensors
train_data_input = torch.from_numpy(train_data_input_arr).to(device)
assert train_data_input.shape[1] == dim_input
train_data_output = torch.from_numpy(train_data_output_arr).to(device)
assert train_data_output.shape[1] == dim_output
val_data_input = torch.from_numpy(val_data_input_arr).to(device)
assert val_data_input.shape[1] == dim_input
val_data_output = torch.from_numpy(val_data_output_arr).to(device)
assert val_data_output.shape[1] == dim_output

############################################################################################################



############################################### Defining Model Class ###############################################

class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, n, l):
        super(Model, self).__init__()
        self.hidden_layer_dim  = n
        self.num_hidden_layers = l
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, n), torch.nn.PReLU())
        for i in range(l-1):
            self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, n), torch.nn.PReLU())
        self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, output_size))

    def forward(self, input_data):
        pred = self.layers(input_data)
        return pred

#####################################################################################################################



############################################### Defining Loss Function ###############################################

""" Loss function """
def loss_fn(pred, act, isSum, isAbs, isPer, reg):
    ''' pred, act- torch.tensor of shape (train_data_input.shape[0], dim_output)
        isSum, isAbs, isPer- bool
        reg- number
        returns- number '''
    if isSum:
        fun = torch.sum
    else:
        fun = torch.mean

    err = torch.add(pred, -act)

    if isAbs:
        err = torch.abs(err)
    else:
        err = torch.mul(err, err)
        act2 = torch.mul(act, act)
    
    if isPer:
        try:
            err = torch.div(err, act2)*1e4
        except:
            err = torch.div(err, act)*1e2
    else:
        err = torch.cat((err.narrow(1, 0, 3), reg*err.narrow(1, 3, 3)), dim=1)

    loss = fun(torch.sum(err, dim=1, keepdim=True), dim=0)
    return loss

#####################################################################################################################



############################################### Defining Animation function ###############################################

def animate(err_arr, vd_nm, subplot_rows, subplot_cols):
    ''' err_arr- numpy array of shape (dim_output, number of frames)
        vd_nm- string '''
    animate.err_arr = err_arr.transpose()

    num_plots = subplot_rows * subplot_cols

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # intialize line objects (one in each axis)
    line = []
    clrs = [['k', 'k'], ['k', 'r'], ['r', 'r']]
    for i in range(subplot_rows):
        for j in range(subplot_cols):
            l, = ax[i][j].plot([], [], lw=1, color=clrs[i][j])
            line.append(l)

    # axes limit settings
    for r in ax:
        for a in r:
            a.set_ylim(0, 100)
            a.set_xlim(0, 4)
            a.grid()

    # initialization function 
    def init(): 
        # creating an empty plot/frame 
        for i in range(num_plots):
            line[i].set_data([], []) 
        return line

    # data generator for frame_fn function
    def data_gen():
        for cnt in range(animate.err_arr.shape[0]):
            yield cnt*np.ones([1, 1]), animate.err_arr[cnt].reshape((num_plots, 1))

    # initialize the data arrays 
    animate.xdata, animate.ydata = np.empty([1, 1]), np.empty([num_plots, 1])

    # function to generate frame data, will run on every frame
    def frame_fn(data):
        # update the data arrays
        t, y = data
        if t != [[0]]:
            animate.xdata = np.append(animate.xdata, t, axis=1)
            animate.ydata = np.append(animate.ydata, y, axis=1)
        else:
            animate.xdata = t
            animate.ydata = y

        # axis limits checking
        for r in ax:
            for a in r:
                xmin, xmax = a.get_xlim()
                if t >= xmax:
                    a.set_xlim(xmin, 2*xmax)
                    a.figure.canvas.draw()

        # update the data of both line objects
        for i in range(num_plots):
            line[i].set_data(animate.xdata.reshape((1, -1)), animate.ydata[i].reshape((1, -1)))

        return line

    ani = animation.FuncAnimation(fig, frame_fn, data_gen, init_func=init, blit=True, interval=10, save_count=err_arr.shape[1]+10)

    # save the animation as mp4 video file 
    ani.save(vd_nm + ".mp4", writer = 'ffmpeg', fps = 1)

    # closing current figure
    plt.close('all')

############################################################################################################



count = 1   # for counting model number

for s in isSum:
    for a in isAbs:
        for p in isPer:
            if p:
                reg = [0]
            else:
                reg = Reg
            for R in reg:

                start = time.time()     # timer for calculating execution time

                min_percent_err = 100*np.ones([6, 1])   

                # creating analysis file
                present = datetime.datetime.now()
                file_name = "Analysis_files/new/isSum-" + repr(s) + "_isAbs-" + repr(a) + "_isPer-" + repr(p) + "_reg-" + repr(R) + "_" \
                            + repr(present.year) + "_" + repr(present.month) + "_" + repr(present.day) + "_" + repr(present.hour) + "_" + repr(present.minute)
                print("Working on ", file_name, ".............................................")
                analysis_file = open(file_name + ".csv", 'w+')
                analysis_file.write(','.join(["Learning Rate"] + [repr(e) for e in learning_rates] + ["\n"]))
                analysis_file.write(','.join(["Number of units per hidden layer"] + [repr(e) for e in num_layers] + ["\n"]))
                analysis_file.write(','.join(["Number of hidden layers"] + [repr(e) for e in num_units] + ["\n"]))
                analysis_file.write(','.join(["Number of iterations"] + [repr(e) for e in num_iter] + ["\n"]))
                analysis_file.write(','.join(["Regularisation", repr(R), "\n"]))
                analysis_file.write(','.join(["Loss function", "isSum", repr(s), "\n,"]))
                analysis_file.write(','.join(["isAbsolute", repr(a), "\n,"]))
                analysis_file.write(','.join(["isPercent", repr(p)] + ["\n", "\n", "\n"]))
                analysis_file.write(','.join(["Learning Rate", "Number of hidden layers", "Number of units per hidden layer", "Number of iterations", "e1", "e2", "e3", "e4", "e5", "e6", "\n"]))
                
                for r in learning_rates:
                    for l in num_layers:
                        for n in num_units: 
                            ############################################### Defining model ###############################################

                            learning_rate = r
                            no_hidden_layers = l
                            num_units_per_layer = n
                            
                            model = Model(dim_input, dim_output, num_units_per_layer, no_hidden_layers)
                            model.to(device)

                            my_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                            ############################################################################################################


                            for i in num_iter:
                                parameters_arr = np.array([[r, l, n, i]])

                                ############################################### Training Model ###############################################

                                for t in range(i):
                                    # Forward pass: compute predicted output by passing training input to the model.
                                    train_pred = model(train_data_input)
                                    assert train_pred.shape == train_data_output.shape, train_pred.shape

                                    # Compute loss.
                                    train_loss = loss_fn(train_pred, train_data_output, s, a, p, R)

                                    # zero all of the gradients
                                    my_optimizer.zero_grad()

                                    # Backward pass: compute gradient of the loss with respect to model parameters
                                    train_loss.backward()

                                    # Calling the step function on an Optimizer makes an update to its parameters
                                    my_optimizer.step()

                                    # collecting data for animation
                                    if do_anim and i == num_iter[-1] and t%fps==0:
                                        # calculating error on validation data
                                        val_pred = model(val_data_input)
                                        max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
                                        max_percent_err = max_percent_err.values.detach().cpu().numpy().reshape((6, 1))*100
                                        
                                        try:
                                            max_percent_err_arr = np.append(max_percent_err_arr, max_percent_err, axis=1)
                                        except:
                                            max_percent_err_arr = max_percent_err
                                        assert (max_percent_err_arr.shape[0] == dim_output) and (max_percent_err.shape[0] == dim_output), (max_percent_err_arr.shape,max_percent_err.shape)

                                ############################################################################################################


                                print(count, end=" ")

                                # making animation
                                if do_anim and i == num_iter[-1]:
                                    video_name = "Analysis_files/new/isSum-" + repr(s) + "_isAbs-" + repr(a) + "_isPer-" + repr(p) + "_reg-" + repr(R) \
                                                 + "_r-" + repr(r) + "_l-" + repr(l) + "_n-" + repr(n) + "_i-" + repr(i) + "_" \
                                                 + repr(present.year) + "_" + repr(present.month) + "_" + repr(present.day) + "_" + repr(present.hour) + "_" + repr(present.minute)
                                    assert max_percent_err_arr.shape == (dim_output, i/fps), max_percent_err_arr.shape
                                    print("Creating animation for: ", count)
                                    animate(max_percent_err_arr, video_name, num_subplot_rows, num_subplot_cols)
                                    max_percent_err_arr = None

                                # calculating error on validation data
                                if not (do_anim and i == num_iter[-1]):
                                    val_pred = model(val_data_input)
                                    max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
                                    max_percent_err = max_percent_err.values.detach().cpu().numpy().reshape((6, 1))*100

                                # writing max error data of each model
                                nxt_row = np.append(parameters_arr, max_percent_err.transpose())
                                analysis_file.write(','.join([repr(e) for e in nxt_row.tolist()] + ["\n"]))

                                # updating best model parameters for each output attribute
                                min_percent_err_parameters = np.zeros([6, 4])
                                for j in range(6):
                                    if max_percent_err[j][0] < min_percent_err[j][0]:
                                        min_percent_err_parameters[j] = parameters_arr[0]
                                        min_percent_err[j][0] = max_percent_err[j][0]

                                # updaing best models overall
                                flag1 = True
                                flag2 = True
                                for j in range(6):
                                    if flag1:
                                        flag1 = max_percent_err[j][0] < 1
                                    if flag2:
                                        flag2 = max_percent_err[j][0] < 2
                                if flag1:
                                    try:
                                        less_than_1.append(nxt_row.tolist())
                                    except:
                                        less_than_1 = [nxt_row.tolist()]
                                elif flag2:
                                    try:
                                        less_than_1.append(nxt_row.tolist())
                                    except:
                                        less_than_1 = [nxt_row.tolist()]

                                count += 1


                # writing best model parameters for each output attribute to analysis file
                analysis_file.write("\n")
                for i in range(6):
                    analysis_file.write(','.join([repr(e) for e in min_percent_err_parameters[i].tolist()] + [repr(min_percent_err[i][0]), "e"+repr(i+1) , "\n"]))

                # writing best models overall to analysis file
                try:
                    assert less_than_1
                    assert less_than_2 
                except:
                    analysis_file.write("\nNo Top Performers found!!!!\n")
                else:
                    analysis_file.write("\nAll errors less than 1\n")
                    for ele in less_than_1:
                        analysis_file.write(','.join([repr(e) for e in ele] + ["\n"]))
                    analysis_file.write("\nAll errors less than 2\n")
                    for ele in less_than_2:
                        analysis_file.write(','.join([repr(e) for e in ele] + ["\n"]))
                
                # calculating running time and closing analysis file
                end = time.time()
                print("\n")
                print("Total Time taken by GPU: ", end-start, "\n")
                analysis_file.write(','.join(["\nTotal Time taken by GPU(in seconds)", repr(end-start), "\n"]))
                analysis_file.close()
            
