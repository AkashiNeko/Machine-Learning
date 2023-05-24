import matplotlib.pyplot as plt
import torch

with plt.style.context('dark_background'):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(torch.sin(torch.arange(10)))
    axs[1].plot(torch.cos(torch.arange(10)))
    plt.show()