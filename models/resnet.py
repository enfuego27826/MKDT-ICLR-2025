import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    """
    ResNet18 backbone that returns features.
    """

    def __init__(self,pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.projection_head = nn.Sequential(
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )

    def forward(self,x,project=False):
        x = self.features(x)
        x = x.view(x.shape[0],-1)
        if project:
            return self.projection_head(x)

        return x

    def get_representation(self,x):
        return self.forward(x)