from torchvision import models
import torch.nn as nn
import torch


def load_pretrained_model(model, path):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path,  map_location=lambda storage, loc: storage))
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_resnet(train_labels, model_path=None):
    resnet34 = models.resnet34(pretrained=True)
    num_filters = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_filters, train_labels.shape[1])
    return load_pretrained_model(resnet34, model_path) if model_path else resnet34


def get_densenet(train_labels, pretrained=True, model_path=None):
    densenet = models.densenet161(pretrained=pretrained)
    num_filters = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(nn.Linear(num_filters, train_labels.shape[1]))
    return load_pretrained_model(densenet, model_path) if model_path else densenet


def get_effecientnet():
    pass
