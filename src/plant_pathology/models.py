from torchvision import models
import torch.nn as nn


def get_resnet(train_labels):
    resnet18 = models.resnet18(pretrained=True)
    num_filters = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_filters, train_labels.shape[1])


def get_densenet(train_labels):
    densenet = models.densenet161(pretrained=True)
    num_filters = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(nn.Linear(num_filters, train_labels.shape[1]))
    return densenet


def get_effecientnet():
    pass