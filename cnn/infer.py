import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from my_conv_net import *

model = ConvNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载图片并进行预处理
image = cv2.imread('./data/digit.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (28, 28))  # 将图片调整为MNIST数据集的大小

# plt.imshow(image)
# plt.show()

image_tensor = transforms.ToTensor()(image)
image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)
image_tensor = image_tensor.unsqueeze(0)  # 将数据增加一个维度，以符合模型的输入大小

# 将数据传输到GPU上进行预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_tensor = image_tensor.to(device)
# with torch.no_grad():
#     outputs = model(image_tensor)
#     _, predicted = torch.max(outputs.data, 1)
#     print("Predicted Number is:", predicted.item())
#     prob = F.softmax(outputs.cpu().reshape(10), dim=0)
#     plt.bar(range(10), prob)
#     plt.xlabel('digit')
#     plt.ylabel('probability')
#     plt.title('Predicted Number is: {}'.format(predicted.item()))
#     plt.show()


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
test_dataset = datasets.MNIST(root=mnist_path, train=False, transform=transform)

# 定义数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, pin_memory=True)

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    i = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        true_count = (predicted == labels).sum().item()
        i += 1
        print('[{}]\ttrue count: [{}/{}]'.format(i, true_count, labels.size(0)))
        correct += true_count

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
