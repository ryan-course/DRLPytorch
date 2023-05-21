#------------------------
#线性回归
#图中的点表示样本点，线是拟合出来的，也可以说，我们观测到的是图中的点，而通过学习得到的是 y=wx+b 这条线
#------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1      #一维
output_size = 1     #一维
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Convert numpy arrays to torch tensors
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# Linear regression model

#case1: 自定义实现
# class LinearRegression(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         out = self.linear(x)
#         return out
 
# model = LinearRegression(input_size, output_size)

#case2: 直接使用使用PyTorch的nn包中的Linear类实现
model = nn.Linear(input_size, output_size)  #FLAG-Ryan:线性回归函数 y=wx+b

# Loss and optimizer
loss_fn = nn.MSELoss()    #损失函数   均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  #优化器(学习率在此用)

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward and optimize 常规写法
    optimizer.zero_grad()   #调整模型的参数
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Predicted
predicted = model(inputs).detach().numpy()

# Plot the graph
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
