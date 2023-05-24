import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from my_conv_net import *

def crop(image):
    height = image.shape[0]
    weight = image.shape[1]
    
    top, bottom, left, right = 0, 0, 0, 0
    
    for i in range(image.shape[0]):
        if cv2.countNonZero(image[i, :]) > 0:
            top = i
            break
    for i in range(image.shape[0]-1, -1, -1):
        if cv2.countNonZero(image[i, :]) > 0:
            bottom = i
            break
    for i in range(image.shape[1]):
        if cv2.countNonZero(image[:, i]) > 0:
            left = i
            break
    for i in range(image.shape[1]-1, -1, -1):
        if cv2.countNonZero(image[:, i]) > 0:
            right = i
            break

    top += height
    bottom += height
    left += weight
    right += weight
    
    center1 = (top + bottom) // 2
    center2 = (left + right) // 2
    
    a = int(max(abs(top - bottom) // 2, abs(right - left) // 2) * 1.5)
    
    # 填充黑色边框
    image = cv2.copyMakeBorder(image, height, height, weight, weight, cv2.BORDER_CONSTANT, value=0)
    
    return image[center1 - a:center1 + a, center2 - a:center2 + a]

def draw_bar(outputs, lable, image):
    prob = F.softmax(outputs.cpu().reshape(10), dim=0)
    prob = (prob * 100).tolist()
    with plt.style.context('dark_background'):
        fig = plt.figure()
        fig.set_size_inches(11, 5)
        plt.subplot(1, 2, 1)
        rects = plt.bar(range(10), prob)
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2 - 0.5, height + 1, '{:.2f}'.format(height))
        plt.xticks(range(10))
        plt.yticks(range(0, 101, 10))
        plt.xlabel('digit (0 ~ 9)')
        plt.ylabel('probability (%)')
        plt.title('Predicted Number is: {}'.format(lable.item()))
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.show()

def infer_single(device: str, image_path: str):
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载图片 预处理
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop(image)
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
    draw_bar(outputs, predicted, image)


infer_single('cpu', './data/digit.png')
