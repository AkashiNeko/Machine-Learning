import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from my_conv_net import *

device = 'cpu'

model = ConvNet()
if device == 'cuda':
    model = model.to(device)
model.load_state_dict(torch.load(model_path))

# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
test_dataset = datasets.MNIST(root=mnist_path, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, pin_memory=True)

# 测试模型（10000张手写数字图片分为100组进行测试）
with torch.no_grad():
    correct, total, epoch = 0, 0, 0
    for images, labels in test_loader:
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        accuracy = (predicted == labels).sum().item()
        epoch += 1
        print('[{}]\taccuracy: {} %'.format(epoch, accuracy))
        correct += accuracy

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
