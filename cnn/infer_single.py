import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from my_conv_net import *


def draw_bar(outputs, lable):
    prob = F.softmax(outputs.cpu().reshape(10), dim=0)
    prob = (prob * 100).tolist()
    rects = plt.bar(range(10), prob)    
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2 - 0.5, height + 1, '{:.2f}%'.format(height))
    plt.xticks(range(10))
    plt.yticks(range(0, 101, 10))
    plt.xlabel('digit (0 ~ 9)')
    plt.ylabel('probability (%)')
    plt.title('Predicted Number is: {}'.format(lable.item()))
    plt.show()

def infer_single(device: str, image_path: str):
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载图片 预处理
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))

    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)
    
    # 增加一个维度，以符合模型的输入大小
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        print("Predicted Number is:", predicted.item())
        draw_bar(outputs, predicted)

infer_single('cpu', './data/digit.png')
