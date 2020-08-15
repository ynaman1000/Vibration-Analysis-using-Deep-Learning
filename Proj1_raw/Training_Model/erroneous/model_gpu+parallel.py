# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle
import time
from multiprocessing import Pool
import datetime

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf, sci_mode=False)
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

train_data_input_arr = data_arr[0:1].transpose()
train_data_input_arr = train_data_input_arr[:train_ind]
train_data_output_arr = data_arr[1:].transpose()
train_data_output_arr = train_data_output_arr[:train_ind]
val_data_input_arr = data_arr[0:1].transpose()
val_data_input_arr = val_data_input_arr[train_ind:]
val_data_output_arr = data_arr[1:].transpose()
val_data_output_arr = val_data_output_arr[train_ind:]

train_data_input = torch.from_numpy(train_data_input_arr).to(device)
train_data_output = torch.from_numpy(train_data_output_arr).to(device)
val_data_input = torch.from_numpy(val_data_input_arr).to(device)
val_data_output = torch.from_numpy(val_data_output_arr).to(device)

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


############################################### Model Function ###############################################

def model_fn(input_list):
    ############################################### Defining model ###############################################

    dim_input = train_data_input.shape[1]
    dim_output = train_data_output.shape[1]
    learning_rate = input_list[0]
    no_hidden_layers = input_list[1]
    num_units_per_layer = input_list[2]

    model = Model(dim_input, dim_output, num_units_per_layer, no_hidden_layers)
    model.to(device)

    """ Loss function """
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    def loss_fn(pred, act):
        fun = torch.sum
        loss = torch.add(pred, -act)
        b_sq_loss = torch.mul(loss.narrow(1, 0, 3), loss.narrow(1, 0, 3))
        x_sq_loss = torch.mul(loss.narrow(1, 3, 3), loss.narrow(1, 3, 3))
        l1 = fun(torch.sum(b_sq_loss, dim=1, keepdim=True), dim=0)
        l2 = fun(torch.sum(x_sq_loss, dim=1, keepdim=True), dim=0)
        # print(l1.item(), l2.item())
        return l1 + 100*l2

    my_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ############################################################################################################


    ############################################### Training Model ###############################################

    num_iter = [500, 1000, 5000]
    for i in num_iter:
        for t in range(i):
            # Forward pass: compute predicted output by passing training input to the model.
            train_pred = model(train_data_input)
            # print(train_pred.shape)

            # Compute and print loss.
            train_loss = loss_fn(train_pred, train_data_output)
            # print(t, train_loss.item())

            # zero all of the gradients
            my_optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            train_loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            my_optimizer.step()


        val_pred = model(val_data_input)
        arr = np.array([r, l, n, i])
        max_err = torch.max(torch.abs(val_pred - val_data_output), dim=0)

        max_percent_err = np.zeros([1, 6])
        for j in range(6):
            max_percent_err[0][j] = (max_err.values[j].item() / val_data_output[max_err.indices[j].item()][j].item())*100
            if max_percent_err[0][j] < min_percent_err[0][j]:
                min_percent_err_parameters[j] = arr
                min_percent_err[0][j] = max_percent_err[0][j]
        nxt_row = np.append(arr, max_percent_err[0])
        
        analysis_file.write(','.join([repr(s) for s in nxt_row]))
        analysis_file.write("\n")
        print(count)
        count += 1

        ############################################################################################################

#####################################################################################################################



if __name__ == '__main__':
    count = 1
    min_percent_err = 100*np.ones([1, 6])
    min_percent_err_parameters = np.zeros([6, 4])

    present = datetime.datetime.now()
    file_name = "Analysis_files/analysis_" + repr(present.year) + "_" + repr(present.month) + "_" + repr(present.day) + "_" + repr(present.hour) + "_" + repr(present.minute) + ".csv"
    analysis_file = open(file_name, 'w+')
    analysis_file.write(','.join(["Learning Rate", "Number of hidden layers", "Number of units per hidden layer", "Number of iterations", "e1", "e2", "e3", "e4", "e5", "e6"]))
    analysis_file.write("\n")

    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    num_layers = [2, 3, 4]
    num_units = [30, 50, 70, 90, 110]
    model_fn_inputs = []
    for r in learning_rates:
        for l in num_layers:
            for n in num_units:
                model_fn_inputs.append([r, l, n])
    start = time.time()

    with Pool(6) as p:
        print(p.map(model_fn, model_fn_inputs))

    analysis_file.write("\n")
    for i in range(6):
        analysis_file.write(','.join([repr(s) for s in min_percent_err_parameters[i]]))
        analysis_file.write("\n")

    end = time.time()
    print("Total Time taken by GPU+parallel: ", end-start)
    analysis_file.write("\n")
    analysis_file.write(','.join["Total Time taken by GPU+parallel(in seconds)", repr(end-start)])
    analysis_file.write("\n")
    analysis_file.close()

            