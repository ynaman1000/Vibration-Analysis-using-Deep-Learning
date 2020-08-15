import numpy as np
import torch
from random import shuffle

q = torch.tensor([[1, 2, 3, 11], [4, 5, 6, 12], [7, 8, 9, 14]], dtype=torch.float32)
print(q[1][2].item())
# w = q.narrow(1, 2, 2)
# print(10*w)
# print(torch.mean(w, dim=0, keepdim=True))

# z = torch.tensor([2.5, 34.1, 56.5, 7.0])
# x = z.numpy()
# f = open("file.csv", 'w+')
# f.write(','.join(["asds", "dfv", "dfv", "bfv"]))
# f.write("\n")
# f.write(','.join([repr(s) for s in np.array([1, 2, 4.4512, 5.2])]))
# f.write("\n")
# f.write(','.join([repr(s) for s in x]))
# f.write("\n")
# # f.write(np.array([1, 2, 4, 5]))
# # f.write("\n")
# f.close()
# # y = -torch.rand(2, 3)
# # print(y)
# a = np.ones(4)*4
# print(a)

# hidden_layer_dims  = 3*np.ones(2)
# print(hidden_layer_dims)


# a = np.array([[1, 2, 3, 11], [4, 5, 6, 12], [7, 8, 9, 14]])
# b = a[:1]
# print(a)
# print(b)

# def load_data(train_data_input, train_data_output, val_data_input, val_data_output, val_data_frac):
#     data_file = open("train_data.csv", 'r')
#     data_list = []
#     for row in data_file.readlines():
#     	row_list = row.split(',')
#     	for i in range(len(row_list)):
#     		row_list[i] = float(row_list[i])
#     	data_list.append(row_list)
#     data_file.close()	
#     shuffle(data_list)
#     data_arr = np.array(data_list, dtype = np.double, ndmin = 2)
#     train_data_input = data_arr
#     train_data_output
#     val_data_input
#     val_data_output
#     print(type(data_arr))
#     print(data_arr.shape)
    

# train_data_input = []
# train_data_output = []
# val_data_input = []
# val_data_output = []
# val_data_frac = 0.1
# load_data(train_data_input, train_data_output, val_data_input, val_data_output, val_data_frac);




# # -*- coding: utf-8 -*-
# import torch

# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Use the nn package to define our model and loss function.
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )
# loss_fn = torch.nn.MSELoss(reduction='sum')

# # Use the optim package to define an Optimizer that will update the weights of
# # the model for us. Here we will use Adam; the optim package contains many other
# # optimization algoriths. The first argument to the Adam constructor tells the
# # optimizer which Tensors it should update.
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(500):
#     # Forward pass: compute predicted y by passing x to the model.
#     y_pred = model(x)

#     # Compute and print loss.
#     loss = loss_fn(y_pred, y)
#     print(t, loss.item())

#     # Before the backward pass, use the optimizer object to zero all of the
#     # gradients for the variables it will update (which are the learnable
#     # weights of the model). This is because by default, gradients are
#     # accumulated in buffers( i.e, not overwritten) whenever .backward()
#     # is called. Checkout docs of torch.autograd.backward for more details.
#     optimizer.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to model
#     # parameters
#     loss.backward()

#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()





# import torch

# x = torch.empty(5, 3)
# x = torch.rand(5, 3)
# x = torch.tensor([2, 435, 65, 7, 1.28])
# y = x.new_ones(2, 3)
# print(x)
# print(y)
# z = torch.rand_like(x, dtype=torch.double)
# print(z)
# print(torch.cuda.is_available())