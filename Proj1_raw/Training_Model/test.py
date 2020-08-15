# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle

############################################### Manual Setting ###############################################

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf, sci_mode=False)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
print("CUDA Available:", torch.cuda.is_available())

datafile_folder = "Test_Data/"
datafile_name = "test_data"    # name of data file
datafile_ext = ".csv"

modelfile_folder = "Analysis_files/new/"
# modelfile_folder = "Analysis_files/new_best/models/"
modelfile_name = "isSum-True_isAbs-False_isPer-True_reg-0_r-0.01_l1-4_n-170_l2-2_2019_6_26_17_18_2"
modelfile_ext = ".pth"

dimInput = 2
dimOutput = 6
numDataPts = 50*100

doLayerNorm = [True]       # for performing layer normalization on each layer
aF = torch.nn.PReLU        # activation functions
L1 = 4                   # number of hidden layers with activation function
N1 = 170                 # number of units per hidden layer with activation function
L2 = 2                   # number of hidden layers without activation function
N2 = 2                # hidden layers without activation function will have units = dimOutput*(N2**?)

make_report = True         # for making report

############################################################################################################


############################################### Loading Data ###############################################

# function for loading data
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
assert data_arr.shape == (numDataPts, dim_input+dim_output), data_arr.shape
data_arr = data_arr.transpose()

# defining data variables as numpy arrays
test_data_input_arr = data_arr[0:dim_input].transpose()
test_data_output_arr = data_arr[dim_input:].transpose()

# converting data variables to torch tensors
test_data_input = torch.from_numpy(test_data_input_arr).to(device)
assert test_data_input.shape[1] == dim_input
test_data_output = torch.from_numpy(test_data_output_arr).to(device)
assert test_data_output.shape[1] == dim_output

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


model = Model(dimInput, dimOutput, L1, N1, L2, N2, doLayerNorm, aF)
model.load_state_dict(torch.load(modelfile_folder+modelfile_name+modelfile_ext))
model.eval()
model.to(device)
test_pred = model(test_data_input)
test_per_dev = torch.div(torch.abs(test_pred - test_data_output), test_data_output)

test_max_per_dev = torch.max(torch.div(torch.abs(test_pred - test_data_output), test_data_output), dim=0)   # not a tensor, see return type of torch.max
test_max_per_dev = test_max_per_dev.values.detach().cpu().numpy().reshape(dim_output)*100

test_pred = test_pred.detach().cpu().numpy().reshape((numDataPts, dim_output))
test_per_dev = test_per_dev.detach().cpu().numpy().reshape((numDataPts, dim_output))*100
# print(test_per_dev)


if make_report:
    file_name = "Report_files/new_best/" + modelfile_name
    print("Working on ", file_name, ".............................................")
    report = open(file_name + ".csv", 'w+')
    report.write(','.join(["", "n", "eta", "beta1", "beta2", "beta3", "x21", "x31", "x32", "\n"]))
    for i in range(numDataPts):
        report.write(','.join(["Actual"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_data_output[i].tolist()] + ["\n"]))
        report.write(','.join(["Prediction"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_pred[i].tolist()] + ["\n"]))
        report.write(','.join(["Per Dev"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_per_dev[i].tolist()] + ["\n", "\n"]))

    report.write(','.join(["\nWorst", "", ""] + [repr(e) for e in test_max_per_dev.tolist()] + ["\n"]))

    print("Closing report file.")
    report.close()
else:
    for i in range(numDataPts):
        print(test_data_output[i])
        print(test_pred[i])
        print(test_per_dev[i])
        print("\n")