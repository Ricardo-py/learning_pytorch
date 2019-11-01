import torch
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import pylab
from torch.autograd import variable

batch_n = 64
hidden_layer = 100
input_data = 1000
output_data = 10

models = torch.nn.Sequential(
    torch.nn.Linear(input_data,hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer,output_data)
)

epoch_n = 10
learning_rate = 1e-3
loss_fn = torch.nn.MSELoss()

x = variable(torch.randn(batch_n, input_data))
y = variable(torch.randn(batch_n, output_data))

optimzer = torch.optim.Adam(models.parameters(),lr=learning_rate)

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred,y)
    print("epoch:{}, loss:{}".format(epoch,loss.data))

    optimzer.zero_grad()
    loss.backward()

    optimzer.step()