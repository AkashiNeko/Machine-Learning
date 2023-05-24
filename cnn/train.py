import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from my_conv_net import *


learn_rate = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, pin_memory=True)


# 模型、损失函数和优化器
model = ConvNet()
if device == 'cuda':
    model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

loss_list = []

# 训练模型
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
            
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))
            loss_list.append(loss.item())

# 保存模型
torch.save(model.state_dict(), model_path)
print('save model to {}'.format(model_path))

plt.plot(loss_list)
plt.title('loss')
plt.show()
