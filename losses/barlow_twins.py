import torch
import torch.nn as nn

def off_diagonal(x):
    n,m = x.shape
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self,lambda_bt=5e-3):
        super().__init__()
        self.lambda_bt = lambda_bt
    

    def forward(self,z1,z2):
        z1_norm = (z1-z1.mean(0))/z1.std(0)
        z2_norm = (z2-z2.mean(0))/z2.std(0)

        N,D = z1.size()
        c = (z1_norm.T @ z2_norm)/N

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c ** 2).sum() - torch.diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_bt*off_diag
        return loss  