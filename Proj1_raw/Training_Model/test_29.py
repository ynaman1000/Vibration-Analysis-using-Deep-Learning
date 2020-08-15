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

dim_input = 2
dim_output = 6
num_data_pts = 50*100

num_hl_with_act = 4
num_units_with_act = 170
num_hl_without_act = 2
num_units_without_act_param = 2
pad_hl_without_act_front = True
act_fn = torch.nn.ReLU()

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
assert data_arr.shape == (num_data_pts, dim_input+dim_output), data_arr.shape
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
    ''' Using activation_fn activation function in all hidden layes '''
    def __init__(self, input_size, output_size, hlwa, n1, hlwoa, n2, pad_f, activation_fn):
        super(Model, self).__init__()
        if pad_f and hlwoa:
            self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, output_size*(n2**1)))
            for i in range(2, hlwoa+1):
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(output_size*(n2**(i-1)), output_size*(n2**i)))
        else:
            self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, n1), activation_fn)
        for i in range(hlwa-1):
            self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n1, n1), activation_fn)
        if hlwoa:
            self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n1, output_size*(n2**hlwoa)))
            for i in range(hlwoa):
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(output_size*(n2**(hlwoa-i)), output_size*(n2**(hlwoa-i-1))))
        else:
            self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n1, output_size))

    def forward(self, input_data):
        pred = self.layers(input_data)
        return pred

#####################################################################################################################


model = Model(dim_input, dim_output, num_hl_with_act, num_units_with_act, num_hl_without_act, num_units_without_act_param, pad_hl_without_act_front, act_fn)
model.load_state_dict(torch.load(modelfile_folder+modelfile_name+modelfile_ext))
model.eval()
model.to(device)
test_pred = model(test_data_input)
test_per_dev = torch.div(torch.abs(test_pred - test_data_output), test_data_output)

test_max_per_dev = torch.max(torch.div(torch.abs(test_pred - test_data_output), test_data_output), dim=0)   # not a tensor, see return type of torch.max
test_max_per_dev = test_max_per_dev.values.detach().cpu().numpy().reshape(dim_output)*100

test_pred = test_pred.detach().cpu().numpy().reshape((num_data_pts, dim_output))
test_per_dev = test_per_dev.detach().cpu().numpy().reshape((num_data_pts, dim_output))*100
# print(test_per_dev)


if make_report:
    file_name = "Report_files/new_best/" + modelfile_name
    print("Working on ", file_name, ".............................................")
    report = open(file_name + ".csv", 'w+')
    report.write(','.join(["", "n", "eta", "beta1", "beta2", "beta3", "x21", "x31", "x32", "\n"]))
    for i in range(num_data_pts):
        report.write(','.join(["Actual"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_data_output[i].tolist()] + ["\n"]))
        report.write(','.join(["Prediction"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_pred[i].tolist()] + ["\n"]))
        report.write(','.join(["Per Dev"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_per_dev[i].tolist()] + ["\n", "\n"]))

    report.write(','.join(["\nWorst", "", ""] + [repr(e) for e in test_max_per_dev.tolist()] + ["\n"]))

    print("Closing report file.")
    report.close()