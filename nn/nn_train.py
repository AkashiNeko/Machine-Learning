import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from my_net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'nn.pth'

# training sample
train_inputs = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                            dtype=torch.float32).to(device)

train_outputs = torch.tensor([[0, 0], [1, 0], [1, 0], [0, 1],
                              [1, 0], [0, 1], [0, 1], [1, 1]],
                             dtype=torch.float32).to(device)

# net
net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# train
epochs = 100000
epoch_list = []
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(train_inputs)
    loss = criterion(outputs, train_outputs)
    loss.backward()
    optimizer.step()
    # print info
    if (epoch + 1) % 5000 == 0:
        losses.append(loss.item())
        epoch_list.append(epoch + 1)
        for i in range(8):
            print('input: {}, prediction: [{:.4f}, {:.4f}]'.format(
                train_inputs[i].int().tolist(), outputs[i][0].item(), outputs[i][1].item()))
        print('Epoch: {}, Loss: {:f}'.format(epoch + 1, loss))

# save model
torch.save(net.state_dict(), model_path)
print('Save model', model_path)

# draw losses
plt.plot(epoch_list, losses)
plt.xlabel('Loss')
plt.ylabel('epoch')
plt.show()
