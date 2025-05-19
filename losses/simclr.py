import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self,tau=0.5):
        super().__init__()
        self.tau = tau\
    
    def forward(self,z1,z2):
        """
        Returns NTXentLoss of two views

        Args:
        z1,z2 : tensor embeddings of two views

        Returns:
        Loss : NTXentLoss of two views
        """
        B,D = z1.shape
        z = torch.cat([z1,z2],dim=0)
        z = F.normalize(z,dim=1)

        sim = torch.matmul(z,z.T)
        sim = sim/self.tau

        labels = torch.arange(B,device=z.device)
        labels = torch.cat([labels+B,labels],dim=0)

        mask = torch.eye(2*B,device=z.device).bool()
        sim.masked_fill_(mask,float('-inf'))

        loss = F.cross_entropy(sim,labels)
        return loss