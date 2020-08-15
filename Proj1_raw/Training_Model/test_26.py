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

data_filename = "Test_data/test_data.csv"
model_filename = ""
dim_input = 2
dim_output = 6
num_data_pts = 10000

num_hidden_layers = 4
num_units_per_layer = 170
act_fn = torch.nn.PReLU()
last_layer_act_fn = torch.nn.Softplus()

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


data_arr  = load_data(data_filename)
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
    ''' Using act_fn activation function in all but last hidden layes, last_layer_act_fn activation function used in last layer '''
    def __init__(self, input_size, output_size, l. n):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, n), act_fn)
        for i in range(l-1):
            if i < l-2:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, n), act_fn)
            else:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, n), last_layer_act_fn)
        self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, output_size))

    def forward(self, input_data):
        pred = self.layers(input_data)
        return pred

#####################################################################################################################


model = Model(dim_input, dim_output, num_hidden_layers, num_units_per_layer)
model.load_state_dict(torch.load(model_filename))
model.eval()
model.to(device)
test_pred = model(test_data_input)
test_per_dev = torch.div(torch.abs(train_pred - train_data_output), val_data_output).detach().cpu().numpy().reshape((6, 1))*100
print(test_per_dev)