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
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()

datafile_folder = "Training_Data/"
datafile_name = "train_data"    # name of data file
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 6      # output dimension
numDataPts = 15000*301     # total number of data points
case = 1
assert case in [1, 2, 3], case
if case == 1:
    miniBatchSize = 2**12     # size of mini-batch used for training
    numBatch = 1              # number of mini-batches in which training data will be divided for training
    valid_size = 500               # size of validation dataset
elif case == 2:
    miniBatchSize = 2**13     # size of mini-batch used for training
    numBatch = 5              # number of mini-batches in which training data will be divided for training
    valid_size = 500               # size of validation dataset
elif case == 3:
    miniBatchSize = 2**16     # size of mini-batch used for training
    numBatch = 6              # number of mini-batches in which training data will be divided for training
    valid_size = 500               # size of validation dataset
numTrainEx = numBatch * miniBatchSize     # number of datapoints used for training
if valid_size:
    numValEx = valid_size                       # number of datapoints used for validation
else:
    numValEx = numDataPts - numTrainEx   # number of datapoints used for validation
numIter = 40000             # number of iterations for training
doLayerNorm = True       # for performing layer normalization on each layer

actFuns = [torch.nn.PReLU, torch.nn.SELU]        # activation functions
# act_fns = [torch.nn.PReLU(), torch.nn.ReLU(), torch.nn.ELU(), torch.nn.SELU(), torch.nn.Hardtanh()]     # activation functions
# act_fns = [torch.nn.ELU(), torch.nn.SELU(), torch.nn.ReLU()]                                          # activation functions
# act_fns = [torch.nn.Softplus(), torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.ReLU6()]     # activation functions
optimizers = [torch.optim.Adam, torch.optim.ASGD, torch.optim.SGD]

numHypParamSamples = 1
if numHypParamSamples == 1:
    initLR = 1e-2           # initial learning rates
    lrUpdateCoeff = 2        # number by which learning rate will be divided on every updation
    lrUpdatePeriod = 2000       # number of iterations after which learning rate will be updated
    L1 = 4                   # number of hidden layers with activation function
    N1 = 170                 # number of units per hidden layer with activation function
    L2 = 2                   # number of hidden layers without activation function
    N2 = 2                # hidden layers without activation function will have units = dimOutput*(N2**?)

# For doing post-processing/analysis
do_analysis = True           # for making analysis file
plt_max_percent_dev = True   # for plotting max percent deviation in training and validation data vs time step
y_ax_lim_per_dv = 5          # max value on y-axis for max_dev plots
plt_loss = True              # for plotting loss plots
y_ax_lim_loss = 15000        # max value on y-axis for loss plots
save_best_models = True     # for saving best models
do_anim = False              # for making animation video
if do_anim or plt_max_percent_dev:
    fps = 100               # frames per second in animation, error after every fps iterations will be plotted
    num_subplot_rows = 3    # number of rows of subplot array in animation
    num_subplot_cols = 2    # number of columns of subplot array in animation

############################################################################################################



############################################### Loading Data ###############################################

# Function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, 7) '''
    data_file = open(fl_nm, 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        for i in range(len(row_list)):
            row_list[i] = float(row_list[i])
        data_list.append(row_list)
    data_file.close()   
    shuffle(data_list)
    return np.array(data_list, dtype = np.float32, ndmin = 2)


data_arr  = load_data(datafile_folder+datafile_name+datafile_ext)
assert data_arr.shape == (numDataPts, dimInput+dimOutput), data_arr.shape
data_arr = data_arr.transpose()

# defining data variables as numpy arrays
train_data_input_arr = data_arr[0:dimInput].transpose()
train_data_input_arr = train_data_input_arr[:numTrainEx]
train_data_output_arr = data_arr[dimInput:].transpose()
train_data_output_arr = train_data_output_arr[:numTrainEx]
val_data_input_arr = data_arr[0:dimInput].transpose()
val_data_input_arr = val_data_input_arr[numTrainEx:numTrainEx+numValEx]
val_data_output_arr = data_arr[dimInput:].transpose()
val_data_output_arr = val_data_output_arr[numTrainEx:numTrainEx+numValEx]

# converting data variables to torch tensors
train_data_input = torch.from_numpy(train_data_input_arr).to(device)
assert train_data_input.shape[1] == dimInput
train_data_output = torch.from_numpy(train_data_output_arr).to(device)
assert train_data_output.shape[1] == dimOutput
val_data_input = torch.from_numpy(val_data_input_arr).to(device)
assert val_data_input.shape[1] == dimInput
val_data_output = torch.from_numpy(val_data_output_arr).to(device)
assert val_data_output.shape[1] == dimOutput

############################################################################################################



############################################### Defining Model Class ###############################################

class Model(torch.nn.Module):
    ''' Using activationFun activation function in all hidden layes '''
    def __init__(self, inputSize, outputSize, hlwaf, n1, hlwoaf, n2, ln, activationFun):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential()
        # Adding hidden layers without activation function
        if hlwoaf > 0:
            self.addLayer(inputSize, outputSize*(n2**1))
            for i in range(hlwoaf-1):
                self.addLayer(outputSize*(n2**(i+1)), outputSize*(n2**(i+2)))
        # Adding hidden layers with activation function
            self.addLayer(outputSize*(n2**hlwoaf), n1, ln, activationFun)
        else:
            self.addLayer(inputSize, n1, ln, activationFun)
        for _i in range(hlwaf-1):
            self.addLayer(n1, n1, ln, activationFun)
        # Adding hidden layers without activation function
        if hlwoaf > 0:
            self.addLayer(n1, outputSize*(n2**hlwoaf))
            for i in range(hlwoaf):
                self.addLayer(outputSize*(n2**(hlwoaf-i)), outputSize*(n2**(hlwoaf-i-1)))
        else:
            self.addLayer(n1, outputSize)

    def addLayer(self, inDim, outDim, LN=False, actFun=None):
        if LN:
            if actFun != None:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), torch.nn.LayerNorm(outDim), actFun())
            else:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), torch.nn.LayerNorm(outDim))
        else:
            if actFun != None:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), actFun())
            else:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim))

    def forward(self, inputData):
        pred = self.layers(inputData)
        return pred

#####################################################################################################################



############################################### Defining Loss Function ###############################################

def loss_fn(pred, act, isSum=True, isAbs=False, isPer=True, reg=0):
    ''' pred, act- torch.tensor of shape (train_data_input.shape[0], dimOutput)
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



############################################### For adjusting Learning Rate ###############################################

def adjust_learning_rate(optimizer, c):
    """ Divides learning rate by c whenever called """
    for param_group in optimizer.param_groups:
        param_group['lr'] /= c
        print(param_group['lr'])

#####################################################################################################################



############################################### Defining Animation function ###############################################

def animator(err_arr, vd_nm, subplot_rows, subplot_cols):
    ''' err_arr- numpy array of shape (dimOutput, number of frames)
        vd_nm- string '''
    animator.err_arr = err_arr.transpose()
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
            a.set_ylim(0, 10)
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
        for cnt in range(animator.err_arr.shape[0]):
            yield cnt*np.ones([1, 1]), animator.err_arr[cnt].reshape((num_plots, 1))

    # initialize the data arrays 
    animator.xdata, animator.ydata = np.empty([1, 1]), np.empty([num_plots, 1])

    # function to generate frame data, will run on every frame
    def frame_fn(data):
        # update the data arrays
        t, y = data
        if t != [[0]]:
            animator.xdata = np.append(animator.xdata, t, axis=1)
            animator.ydata = np.append(animator.ydata, y, axis=1)
        else:
            animator.xdata = t
            animator.ydata = y

        # axis limits checking
        for r in ax:
            for a in r:
                xmin, xmax = a.get_xlim()
                if t >= xmax:
                    a.set_xlim(xmin, 2*xmax)
                    a.figure.canvas.draw()

        # update the data of both line objects
        for i in range(num_plots):
            line[i].set_data(animator.xdata.reshape((1, -1)), animator.ydata[i].reshape((1, -1)))

        return line

    ani = animation.FuncAnimation(fig, frame_fn, data_gen, init_func=init, blit=True, interval=10, save_count=err_arr.shape[1]+10)

    # save the animation as mp4 video file 
    ani.save(vd_nm + ".mp4", writer = 'ffmpeg', fps = 1)

    # closing all figures
    plt.close('all')

############################################################################################################



############################################### Defining Animation function ###############################################

def plotter(err_arr, fig_nm, subplot_rows, subplot_cols, y_max):
    ''' err_arr- numpy array of shape (dimOutput, number of frames)
        fig_nm- string '''

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # plotting on each subplot
    cnt = 0
    if subplot_cols == 1 or subplot_rows == 1:
        if subplot_cols == 1 and subplot_rows == 1:
            ax.plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1])), err_arr[cnt, :].reshape((err_arr.shape[1])), lw=0.3)
            cnt += 1
        else:
            for a in ax:
                a.plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1])), err_arr[cnt, :].reshape((err_arr.shape[1])), lw=0.3)
                cnt += 1

                # axes limit settings
                a.set_ylim(0, y_max)
                a.set_xlim(0, err_arr.shape[1]+1)
                a.grid()
    else:
        clrs = [['k', 'k'], ['k', 'r'], ['r', 'r']]
        major_ticks = np.arange(0, y_max, 1)
        minor_ticks = np.arange(0, y_max, 0.1)
        for i in range(subplot_rows):
            for j in range(subplot_cols):
                ax[i][j].plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1])), err_arr[cnt, :].reshape((err_arr.shape[1])), lw=0.3, color=clrs[i][j])
                cnt += 1

                # axes limit settings
                ax[i][j].set_ylim(0, y_max)
                ax[i][j].set_xlim(0, err_arr.shape[1]+1)

                # y-axis ticks setting
                ax[i][j].set_yticks(major_ticks)
                ax[i][j].set_yticks(minor_ticks, minor=True)
                ax[i][j].grid(which='both', axis='y')
                ax[i][j].grid(which='minor', axis='y', alpha=0.2)
                ax[i][j].grid(which='major', axis='y', alpha=0.5)            

    plt.savefig(fig_nm + ".svg")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################



count = 1   # for counting model number

for opt in optimizers: 
    for aF in actFuns:

        start = time.time()     # timer for calculating execution time
        present = datetime.datetime.now()       # for creating files' name

        # creating analysis file
        if do_analysis:
            file_name = "Analysis_files/new/LN-" + repr(doLayerNorm) + "_opt-" + repr(opt) + "_actFun-" + repr(aF) + "_case-" + repr(case) \
                        + "_bSz-" + repr(miniBatchSize) + "_bNum-" + repr(numBatch) + "_i-" + repr(numIter) + "_" \
                        + repr(present.year) + "_" + repr(present.month) + "_" + repr(present.day) + "_" + repr(present.hour) + "_" + repr(present.minute)
            print("Working on ", file_name, ".............................................")
            analysis_file = open(file_name + ".csv", 'w+')
            analysis_file.write(','.join(["Number of Inputs", repr(dimInput), "\n"]))
            analysis_file.write(','.join(["Number of Outputs", repr(dimOutput), "\n"]))
            analysis_file.write(','.join(["Number of data points", repr(numDataPts), "\n"]))
            analysis_file.write(','.join(["Case", repr(case), "\n"]))
            analysis_file.write(','.join(["Size of mini-batch", repr(miniBatchSize), "\n"]))
            analysis_file.write(','.join(["Number of batches", repr(numBatch), "\n"]))
            analysis_file.write(','.join(["Number of Training examples", repr(numTrainEx), "\n"]))
            analysis_file.write(','.join(["Number of Validation examples", repr(numValEx), "\n"]))
            analysis_file.write(','.join(["Number of iterations", repr(numIter), "\n"]))
            analysis_file.write(','.join(["Layer Normalization", repr(doLayerNorm), "\n", "\n"]))
            analysis_file.write(','.join(["Optimizer", repr(opt), "\n"]))
            analysis_file.write(','.join(["Activation Function", repr(aF), "\n", "\n"]))
            analysis_file.write(','.join(["Regularisation", repr(None), "\n"]))
            analysis_file.write(','.join(["Loss function", "isSum", repr(True), "\n,"]))
            analysis_file.write(','.join(["isAbsolute", repr(False), "\n,"]))
            analysis_file.write(','.join(["isPercent", repr(True), "\n", "\n", "\n"]))
            analysis_file.write(','.join(["initLR", "lrUpdateCoeff", "lrUpdatePeriod", "L1", "N1", "L2", "N2", "e1", "e2", "e3", "e4", "e5", "e6", "\n"]))
            
            min_percent_err = 100*np.ones([dimOutput, 1])
            min_percent_err_parameters = np.zeros([dimOutput, 7]) 

        for _i in range(numHypParamSamples):
            if numHypParamSamples > 1:
                initLR = 10**np.random.uniform(-5, 0)           # initial learning rates
                lrUpdateCoeff = np.random.randint(1, 11)        # number by which learning rate will be divided on every updation
                lrUpdatePeriod = np.random.randint(2000, 10000)       # number of iterations after which learning rate will be updated
                L1 = np.random.randint(2, 6)                   # number of hidden layers with activation function
                N1 = np.random.randint(30, 200)                 # number of units per hidden layer with activation function
                L2 = np.random.randint(0, 5)                   # number of hidden layers without activation function
                if L2 == 0:
                    N2 = 0                                      
                else:
                    N2 = np.random.randint(2, 5)                # hidden layers without activation function will have units = dimOutput*(N2**?)
            
            ############################################### Defining model ###############################################
                                                     
            parameters_arr = np.array([initLR, lrUpdateCoeff, lrUpdatePeriod, L1, N1, L2, N2]).reshape((1, min_percent_err_parameters.shape[1]))
            print(parameters_arr)

            model = Model(dimInput, dimOutput, L1, N1, L2, N2, doLayerNorm, aF)
            model.to(device)
            myOptimizer = opt(model.parameters(), lr=initLR)

            ############################################################################################################


            ############################################### Training Model ###############################################

            for t in range(numIter):
                print(count, t)
                if t%lrUpdatePeriod == lrUpdatePeriod-1 and lrUpdateCoeff > 1:
                    adjust_learning_rate(myOptimizer, lrUpdateCoeff)
                
                b_ind = 0
                for j in range(numBatch):
                    if b_ind + miniBatchSize > train_data_input.shape[0]:
                        train_pred = model(train_data_input.narrow(0, b_ind, train_data_input.shape[0]-b_ind))     # Forward pass: compute predicted output.
                    else:
                        train_pred = model(train_data_input.narrow(0, b_ind, miniBatchSize))     # Forward pass: compute predicted output.
                    assert train_pred.shape == train_data_output.narrow(0, b_ind, miniBatchSize).shape, train_pred.shape
                    train_loss = loss_fn(train_pred, train_data_output.narrow(0, b_ind, miniBatchSize))   # Compute loss.
                    myOptimizer.zero_grad()    # zero all of the gradients
                    train_loss.backward()   # Backward pass: compute gradient of the loss with respect to model parameters
                    myOptimizer.step()     # Calling the step function on an Optimizer makes an update to its parameters
                    b_ind += miniBatchSize
                assert b_ind >= train_data_input.shape[0], b_ind

                # collecting data for animation and plots
                if (do_anim or plt_max_percent_dev or plt_loss) and t%fps==fps-1:
                    # calculating maximum percent deviation in training and validation data
                    if numBatch != 1: 
                        train_pred = model(train_data_input)
                    train_loss = loss_fn(train_pred, train_data_output)
                    train_loss = train_loss.detach().cpu().numpy().reshape((1, 1))
                    try:
                        train_loss_arr = np.append(train_loss_arr, train_loss, axis=1)
                    except:
                        train_loss_arr = np.array([train_loss]).reshape((1, 1))
                    assert (train_loss_arr.shape[0] == 1) and (train_loss.shape[0] == 1), (train_loss_arr.shape,train_loss.shape)

                    val_pred = model(val_data_input)
                    val_loss = loss_fn(val_pred, val_data_output)
                    val_loss = val_loss.detach().cpu().numpy().reshape((1, 1))
                    try:
                        val_loss_arr = np.append(val_loss_arr, val_loss, axis=1)
                    except:
                        val_loss_arr = np.array([val_loss]).reshape((1, 1))
                    assert (val_loss_arr.shape[0] == 1) and (val_loss.shape[0] == 1), (val_loss_arr.shape,val_loss.shape)

                    train_max_percent_err = torch.max(torch.div(torch.abs(train_pred - train_data_output), train_data_output), dim=0)   # not a tensor, see return type of torch.max
                    train_max_percent_err = train_max_percent_err.values.detach().cpu().numpy().reshape((dimOutput, 1))*100
                    try:
                        train_max_percent_err_arr = np.append(train_max_percent_err_arr, train_max_percent_err, axis=1)
                    except:
                        train_max_percent_err_arr = np.array([train_max_percent_err]).reshape(dimOutput, 1)
                    assert (train_max_percent_err_arr.shape[0] == dimOutput) and (train_max_percent_err.shape[0] == dimOutput), (train_max_percent_err_arr.shape,train_max_percent_err.shape)

                    val_max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
                    val_max_percent_err = val_max_percent_err.values.detach().cpu().numpy().reshape((dimOutput, 1))*100
                    try:
                        val_max_percent_err_arr = np.append(val_max_percent_err_arr, val_max_percent_err, axis=1)
                    except:
                        val_max_percent_err_arr = np.array([val_max_percent_err]).reshape((dimOutput, 1))
                    assert (val_max_percent_err_arr.shape[0] == dimOutput) and (val_max_percent_err.shape[0] == dimOutput), (val_max_percent_err_arr.shape,val_max_percent_err.shape)

                   
            ############################################################################################################

            fl_nm = "Analysis_files/new/initLR-" + repr(initLR) + "_lrUC-" + repr(lrUpdateCoeff) + "_lrUP-" + repr(lrUpdatePeriod) \
                    + "_L1-" + repr(L1) + "_N1-" + repr(N1) + "_L2-" + repr(L2) +"_N2-" + repr(N2) + "_" \
                    + repr(present.year) + "_" + repr(present.month) + "_" + repr(present.day) + "_" + repr(present.hour) + "_" + repr(present.minute)

            if (do_anim or plt_max_percent_dev or plt_loss):
                # making animation and plotting error
                assert train_max_percent_err_arr.shape == (dimOutput, numIter/fps), train_max_percent_err_arr.shape
                assert val_max_percent_err_arr.shape == (dimOutput, numIter/fps), val_max_percent_err_arr.shape
                assert train_loss_arr.shape == (1, numIter/fps), train_loss_arr.shape
                assert val_loss_arr.shape == (1, numIter/fps), val_loss_arr.shape
                fig_name = fl_nm
                if do_anim:
                    print("Creating animation for: ", count)
                    animator(train_max_percent_err_arr, fig_name+"_train", num_subplot_rows, num_subplot_cols)
                    animator(val_max_percent_err_arr, fig_name+"_val", num_subplot_rows, num_subplot_cols)
                if plt_max_percent_dev:
                    print("Plotting max percent deviation for: ", count)
                    plotter(train_max_percent_err_arr, fig_name+"_train_max_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv)
                    plotter(val_max_percent_err_arr, fig_name+"_val_max_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv)
                if plt_loss:
                    print("Plotting loss for: ", count)
                    loss_arr = np.append(train_loss_arr, val_loss_arr, axis=0)
                    plotter(loss_arr, fig_name+"_loss", 2, 1, y_ax_lim_loss)
                train_max_percent_err_arr = None
                val_max_percent_err_arr = None
                train_loss_arr = None
                val_loss_arr = None
            elif do_analysis:
                # calculating maximum percent deviation in training and validation data
                train_pred = model(train_data_input)
                train_max_percent_err = torch.max(torch.div(torch.abs(train_pred - train_data_output), train_data_output), dim=0)   # not a tensor, see return type of torch.max
                train_max_percent_err = train_max_percent_err.values.detach().cpu().numpy().reshape((dimOutput, 1))*100

                val_pred = model(val_data_input)
                val_max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
                val_max_percent_err = val_max_percent_err.values.detach().cpu().numpy().reshape((dimOutput, 1))*100

            if do_analysis:
                # writing max error data of each model
                nxt_row = np.append(parameters_arr, train_max_percent_err.transpose())
                analysis_file.write(','.join([repr(e) for e in nxt_row.tolist()] + ["\n"]))
                nxt_row = np.append(parameters_arr, val_max_percent_err.transpose())
                analysis_file.write(','.join([repr(e) for e in nxt_row.tolist()] + ["\n", "\n"]))

                # updating best model parameters for each output attribute
                for j in range(dimOutput):
                    if val_max_percent_err[j][0] < min_percent_err[j][0]:
                        min_percent_err_parameters[j] = parameters_arr[0]
                        min_percent_err[j][0] = val_max_percent_err[j][0]

                # updaing and saving best models overall
                flag1 = True
                flag2 = True
                for j in range(dimOutput):
                    if flag1:
                        flag1 = val_max_percent_err[j][0] < 1
                    if flag2:
                        flag2 = val_max_percent_err[j][0] < 2
                if flag1:
                    # saving model
                    if save_best_models:
                        model_name = fl_nm
                        torch.save(model.state_dict(), model_name+"_1"+".pth")

                    # writing max error data of each model
                    try:
                        less_than_1.append(nxt_row.tolist())
                    except:
                        less_than_1 = [nxt_row.tolist()]
                elif flag2:
                    # saving model
                    if save_best_models:
                        model_name = fl_nm
                        torch.save(model.state_dict(), model_name+"_2"+".pth")

                    # writing max error data of each model
                    try:
                        less_than_2.append(nxt_row.tolist())
                    except:
                        less_than_2 = [nxt_row.tolist()]

            count += 1

        if do_analysis:
            # writing best model parameters for each output attribute to analysis file
            analysis_file.write("\n")
            for i in range(dimOutput):
                analysis_file.write(','.join([repr(e) for e in min_percent_err_parameters[i].tolist()] + [repr(min_percent_err[i][0]), "e"+repr(i+1) , "\n"]))

            # writing best models overall to analysis file
            if 'less_than_1' in globals():
                analysis_file.write("\nAll errors less than 1\n")
                for ele in less_than_1:
                    analysis_file.write(','.join([repr(e) for e in ele] + ["\n"]))
                del less_than_1
            else:
                analysis_file.write("\nNo Top Performers found!!!!\n")

            if 'less_than_2' in globals():
                analysis_file.write("\nAll errors less than 2\n")
                for ele in less_than_2:
                    analysis_file.write(','.join([repr(e) for e in ele] + ["\n"]))
                del less_than_2
            else:
                analysis_file.write("\nNo Other Top Performers found!!!!\n")

        # calculating running time and closing analysis file
        end = time.time()
        print("\n")
        if torch.cuda.is_available():
            print("Total Time taken by GPU: ", end-start, "\n")
            if do_analysis:
                analysis_file.write(','.join(["\nTotal Time taken by GPU(in seconds)", repr(end-start), "\n"]))
                analysis_file.close()
        else:
            print("Total Time taken by CPU: ", end-start, "\n")
            if do_analysis:
                analysis_file.write(','.join(["\nTotal Time taken by CPU(in seconds)", repr(end-start), "\n"]))
                analysis_file.close()
                        
