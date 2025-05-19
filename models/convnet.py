import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    Lightweight ConvNet for the student.
    """

    def __init__(self,depth=3,num_channels=3,num_classes=100,hidden_dim=128,teacher_dim=512):
        super().__init__()
        
        layers = []
        in_channels = num_channels
        out_channels = 64

        for i in range(depth):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels *= 2

        self.encoder = nn.Sequential(*layers)
        final_feat_dim = in_channels
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.proj_head = nn.Linear(final_feat_dim,teacher_dim)
        self.classifier = nn.Linear(teacher_dim,num_classes)

    def forward(self,x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size[0],-1)
        return self.classifier(x)

    def get_representation(self,x):
        """
        Extract intermediate representation before classifier (used for KD)
        """
        x = self.encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size[0],-1)
        return self.proj_head(x)