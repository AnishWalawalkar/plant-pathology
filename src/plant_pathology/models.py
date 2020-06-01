from torchvision import models
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


def load_pretrained_model(model, path):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
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


def get_vggnet(train_labels, pretrained=True, model_path=None):
    model_ft = models.vgg16(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, train_labels.shape[1])
    return load_pretrained_model(model_ft, model_path) if model_path else model_ft


def get_squeezenet(train_labels, pretrained=True, model_path=None):
    """ Squeezenet
    """
    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(512, train_labels.shape[1],
                                       kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = train_labels.shape[1]
    return load_pretrained_model(model_ft, model_path) if model_path else model_ft


def get_effecientnet(train_labels, pretrained=True, model_path=None):
    # https://pypi.org/project/efficientnet-pytorch/
    # https://www.kaggle.com/ateplyuk/pytorch-efficientnet
    # https://www.kaggle.com/akasharidas/plant-pathology-2020-in-pytorch
    model_ft = EfficientNet.from_pretrained('efficientnet-b5')

    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, train_labels.shape[1])
    # model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 1000, bias=True),
    #                              nn.ReLU(),
    #                              nn.Dropout(p=0.5),
    #                              nn.Linear(1000, num_classes, bias=True))
    return load_pretrained_model(model_ft, model_path) if model_path else model_ft


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == 'effecientnet':
        # https://pypi.org/project/efficientnet-pytorch/
        # https://www.kaggle.com/ateplyuk/pytorch-efficientnet
        # https://www.kaggle.com/akasharidas/plant-pathology-2020-in-pytorch
        model_ft = EfficientNet.from_pretrained('efficientnet-b5')

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)
        # model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 1000, bias=True),
        #                              nn.ReLU(),
        #                              nn.Dropout(p=0.5),
        #                              nn.Linear(1000, num_classes, bias=True))

        input_size = 224

    else:
        print("Invalid model name, exiting...")
        # exit()

    return model_ft, input_size
