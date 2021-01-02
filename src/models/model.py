import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms as T



class Resnet152(nn.Module):
    def __init__(self, emb_dim, num_classes, LOAD_ORIGIN=True):
        super(Resnet152, self).__init__()
        self.resnet_152 = models.resnet152(pretrained=LOAD_ORIGIN)
        self.modules = list(self.resnet_152.children())[:-1]
        self.feature_extractor = nn.Sequential(*self.modules)

        self.classifier = nn.Sequential(
                nn.Linear(emb_dim, num_classes),
                nn.BatchNorm1d(num_classes),
                nn.Tanh()
            ) 

    def forward(self, x):
        x_embedding = torch.squeeze(self.feature_extractor(x))
        predict = self.classifier(x_embedding)
        return predict
