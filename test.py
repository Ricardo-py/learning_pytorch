import torch

from torch.autograd import variable

# batch_n = 100
# input_data = 1000
# hidden_layer = 100
# output_data = 10
#
# x = variable(torch.randn(batch_n,input_data),requires_grad=False)
# y = variable(torch.randn(batch_n, output_data),requires_grad=False)
#
# w1 = variable(torch.randn(input_data,hidden_layer),requires_grad=True)
# w2 = variable(torch.randn(hidden_layer,output_data),requires_grad=True)
# #
# #设置学习率
# learning_rate = 1e-6
# #设置训练次数
# epochs = 100000
#
# #开始训练
# for epoch in range(epochs):
#     #计算神经网络的预测值
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)
#
#     #定义损失函数
#     loss = (y_pred - y).pow(2).sum()
#     #print("训练次数为",epoch,"训练的损失函数值：",loss)
#     #打印损失函数的值
#     #print(loss.data)
#     print("训练次数为:{},损失函数的值为:{:.4f}".format(epoch,loss.data))
#     #开始梯度下降
#     loss.backward()
#
#     #更新全值
#     w1.data -= learning_rate * w1.grad.data
#     w2.data -= learning_rate * w2.grad.data
#
#     #设置全职初始化
#     w1.grad.data.zero_()
#     w2.grad.data.zero_()

#定义一个类来完成前向传播
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#
#     def forward(self,input,w1,w2):
#         x = torch.mm(input,w1)
#         x = torch.clamp(x,min=0)
#         y_pred = torch.mm(x,w2)
#         return y_pred
#
#     def backward(self):
#         pass
#
# model = Model()
#
# for epoch in range(epochs):
#     #子类可以用父类的构造？
#     y_pred = model(x,w1,w2)
#
#     loss = (y_pred - y).pow(2).sum()
#
#     print("训练次数为：{},损失函数值为{:.4f}".format(epoch,loss.data))
#
#     loss.backward()
#
#     w1.data -= learning_rate * w1.grad.data
#     w2.data -= learning_rate * w2.grad.data
#
#     w1.grad.data.zero_()
#     w2.grad.data.zero_()

#使用torch.nn包来搭建
#这里的Input_data与output_data都是数据的维度
batch_n = 100
input_data = 1000
hidden_layer = 100
output_data = 10



x = variable(torch.randn(batch_n,input_data),requires_grad=False)
y = variable(torch.randn(batch_n,output_data),requires_grad=False)

#权值
#w1 = variable(torch.randn(input_data,hidden_layer),requires_grad=True)
#w2 = variable(torch.randn(hidden_layer,output_data),requires_grad=True)
models = torch.nn.Sequential(
    #输入层到隐藏层的线性变换
    torch.nn.Linear(input_data,hidden_layer),
    #激活函数
    torch.nn.ReLU(),
    #隐藏层到输出层的线性变换
    torch.nn.Linear(hidden_layer,output_data)
)
print(models)

learning_rate = 1e-3
epochs = 15
loss_fn = torch.nn.MSELoss()
#for epoch in range(epochs):
# for epoch in range(epochs):
#     y_pred = models(x)
#     loss = loss_fn(y_pred,y)
#     print("训练次数为:{},损失函数为{:.4f}".format(epoch,loss.data))
#
#     loss.backward()
#     for param in models.parameters():
#         param.data -= param.grad.data * learning_rate
#     models.zero_grad()

optimzer = torch.optim.Adam(models.parameters(),lr=learning_rate)

for epoch in range(epochs):
    y_pred = models(x)
    loss = loss_fn(y_pred,y)

    print("循环次数为{},损失函数为{:.4f}".format(epoch,loss.data))


    #for param in models.parameters():
    #    param.data -= param.grad.data * learning_rate
    optimzer.zero_grad()

    loss.backward()
    optimzer.step()

    #models.zero_grad()
