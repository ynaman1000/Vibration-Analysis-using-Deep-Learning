# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle
import time
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf, sci_mode=False)

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
train_data_input = torch.from_numpy(train_data_input_arr)
train_data_output = torch.from_numpy(train_data_output_arr)
val_data_input = torch.from_numpy(val_data_input_arr)
val_data_output = torch.from_numpy(val_data_output_arr)
# print(train_data_input.size())
# print(train_data_output.size())
# print(val_data_input.size())
# print(val_data_output.size())

############################################################################################################



############################################### Defining Model ###############################################

""" Model Class """
class Model(torch.nn.Module):
# Our model
    def __init__(self, input_size, output_size, n, l):
        super(Model, self).__init__()
        # self.layers = []
        # for layer_num in range(self.num_hidden_layers):
        #     if layer_num==0:
        #         self.layers.append(torch.nn.Sequential(torch.nn.Linear(input_size, self.hidden_layer_dim),torch.nn.PReLU()))
        #     else:
        #         self.layers.append(torch.nn.Sequential(torch.nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),torch.nn.PReLU()))
        # self.layers.append(torch.nn.Linear(self.hidden_layer_dim, output_size))
        # self.l1 = torch.nn.Sequential(torch.nn.Linear(input_size, n), torch.nn.PReLU())
        # self.l2 = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.PReLU())
        # self.l3 = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.PReLU())
        # self.l4 = torch.nn.Linear(n, output_size)
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, n), torch.nn.PReLU())
        for i in range(l-1):
            self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, n), torch.nn.PReLU())
        self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(n, output_size))


    def forward(self, input_data):
        # for layer_num in range(self.num_hidden_layers):
        #     input_data = self.layers[layer_num](input_data)
        # pred = self.layers[-1](input_data)
        # input_data = self.l1(input_data)
        # input_data = self.l2(input_data)
        # input_data = self.l3(input_data)
        pred = self.layers(input_data)
        return pred


dim_input = train_data_input.shape[1]
# print(type(dim_input))
dim_output = train_data_output.shape[1]
num_units_per_layer = 110
num_hidden_layers = 3
hidden_layer_dims  = num_units_per_layer*np.ones(num_hidden_layers, dtype=np.int)
learning_rate = 1e-2

model = Model(dim_input, dim_output, num_units_per_layer, num_hidden_layers)
# model = Model(dim_input, dim_output)
# print(model.parameters())

# model = torch.nn.Sequential(
#     torch.nn.Linear(dim_input, hidden_layer_dims[0]),
#     torch.nn.PReLU(),
#     torch.nn.Linear(hidden_layer_dims[0], hidden_layer_dims[1]),
#     torch.nn.PReLU(),
#     torch.nn.Linear(hidden_layer_dims[1], hidden_layer_dims[2]),
#     torch.nn.PReLU(),
#     torch.nn.Linear(hidden_layer_dims[2], dim_output)
# )

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
    print(fun(sq_per_err, dim=0))

    # fun = torch.mean
    # fun = torch.sum
    # err = torch.add(pred, -act)
    # per_err = torch.div(err, act)*100
    # abs_per_err = torch.abs(per_err)
    # loss = fun(torch.sum(abs_per_err, dim=1, keepdim=True), dim=0)
    # print(fun(sq_per_err, dim=0))
    return loss

my_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############################################################################################################


############################################### Training Model ###############################################

start = time.time()
# test = [100, 500, 1000, 5000, 10000]
num_iter = [5]   # 5000 is optimum
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
        max_percent_err = torch.max(torch.div(torch.abs(val_pred - val_data_output), val_data_output), dim=0)   # not a tensor, see return type of torch.max
        max_percent_err = max_percent_err.values.detach().numpy()*100
        
        

end = time.time()
print("Total Time taken by CPU: ", end-start)

            ############################################################################################################
