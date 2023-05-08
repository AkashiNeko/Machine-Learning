import torch
from my_net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'nn.pth' # 'nn_epoch_100M.pth'

test_inputs = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                           dtype=torch.float32).to(device)

print('input:', test_inputs.int().tolist())

net = Net().to(device)
net.load_state_dict(torch.load(model_path))

result = net(test_inputs)
result_round = torch.round(result).int().tolist()

print('output:', result.tolist())
print('result:', result_round)
