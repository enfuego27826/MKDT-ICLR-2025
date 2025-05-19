import matplotlib.pyplot as plt
import torchvision
import torch
from dataloader import get_ssl_dataloaders

def show(x1,x2):
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(torchvision.utils.make_grid(x1).permute(1,2,0))
    axs[0].set_title("View 1")
    axs[0].axis('off')
    axs[1].imshow(torchvision.utils.make_grid(x2).permute(1,2,0))
    axs[1].set_title("View 2")
    axs[1].axis('off')
    plt.show()

loader = get_ssl_dataloaders(dataset_name='cifar100',batch_size=4)

for (x1,x2),_ in loader:
    show(x1[0],x2[0])
    break