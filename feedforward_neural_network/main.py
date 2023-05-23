import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#全连接网络

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784    #把一幅28*28像素的图片处理成一个1×784的向量（即1行784列的矩阵）
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100    #一次随机挑选100个样本
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (28*28像素的图片)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #定义网络的层
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #全连接层,w是 [784,500](有hidden_size个神经元,其实是一个784×500的矩阵)
        self.relu = nn.ReLU()   #激活函数 f(x)=max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) #全连接层w是 [500,10](有num_classes个神经元,相当于一个500×10的矩阵)
    
    def forward(self, x):
        #输入为一个784维（1×784）的矩阵,输出1×500的矩阵
        out = self.fc1(x)

        #输入为1×500的矩阵,输出1×500的矩阵
        out = self.relu(out)

        #输入为1×500的矩阵,输出1×10的矩阵    
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #Adam优化器

# Train the model 都是常规写法
total_step = len(train_loader)
for epoch in range(num_epochs): #epoch loop
    for i, (images, labels) in enumerate(train_loader):     #step loop, 一次随机挑选batch_size个样本
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass 正向传播一幅图片
        outputs = model(images) #得到 拟合值
        loss = criterion(outputs, labels)
        
        # Backward and optimize 常规写法
        optimizer.zero_grad()
        loss.backward() #进行一次反向传播
        optimizer.step()#梯度下降
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #返回指定列中值最大的那个元素并返回索引值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
