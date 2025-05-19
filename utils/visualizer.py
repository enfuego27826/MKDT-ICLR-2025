import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

"""
Usage:
    save_image_grid(D_syn,"visuals/synthetic_epoch_10.png")
"""

def save_image_grid(tensor_batch,path,nrow=8,normalize=True):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    grid = vutils.make_grid(tensor_batch,nrow=nrow,normalize=normalize)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.savefig(path)
    plt.close()