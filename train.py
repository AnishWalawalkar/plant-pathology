import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.special import softmax
from functools import partial
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data
import torch.nn as nn

from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms as T, models
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")


from src.plant_pathology.leaf_dataset import LeafDataset
from src.plant_pathology.model_loops import training, validation, testing
from src.plant_pathology.models import get_resnet, get_densenet, get_effecientnet
from src.plant_pathology.visualizations import show_saliency_maps, create_class_visualization
from src.plant_pathology.models import initialize_model, load_pretrained_model


USE_GPU = True

dtype = torch.float32  # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

print('using device:', device)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./plant-pathology-2020-fgvc7/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception, effecientnet]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 30

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False
print('Model name: {}, batch_size: {}'.format(model_name, batch_size))

IMAGE_PATH = Path(data_dir + 'images')


def image_path(file_stem):
    return IMAGE_PATH / f'{file_stem}.jpg'


train_df = pd.read_csv(data_dir + 'train.csv')
test_df = pd.read_csv(data_dir + 'test.csv')

train_paths = train_df['img_file'] = train_df['image_id'].apply(image_path)
test_paths = test_df['img_file'] = test_df['image_id'].apply(image_path)

train_labels = train_df[['healthy', 'multiple_diseases', 'rust', 'scab']]


train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=23, stratify=train_labels)
train_paths.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
valid_paths.reset_index(drop=True, inplace=True)
valid_labels.reset_index(drop=True, inplace=True)

TRAIN_SIZE = train_labels.shape[0]
VALID_SIZE = valid_labels.shape[0]


train_dataset = LeafDataset(train_paths, train_labels)
trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

valid_dataset = LeafDataset(valid_paths, valid_labels, train=False)
validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

test_dataset = LeafDataset(test_paths, train=False, test=True)
testloader = Data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

weighted_f1 = partial(f1_score, labels=[0, 1, 2, 3], average='weighted')
macro_f1 = partial(f1_score, labels=[0, 1, 2, 3], average='macro')
micro_f1 = partial(f1_score, labels=[0, 1, 2, 3], average='micro')
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# micro_roc_auc_score = partial(roc_auc_score, labels=[0, 1, 2, 3], average='macro', multi_class='ovr')
# acc_fns = [accuracy_score, weighted_f1, macro_f1, micro_f1, roc_auc_score]
acc_fns = [accuracy_score, weighted_f1, macro_f1, micro_f1]

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Send the model to GPU
model_ft = model_ft.to(device)


# optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay = 1e-3)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=8e-4, weight_decay=1e-3)
num_train_steps = int(len(train_dataset) / batch_size * num_epochs)
scheduler = get_cosine_schedule_with_warmup(
    optimizer_ft, num_warmup_steps=len(train_dataset) / batch_size * 5, num_training_steps=num_train_steps)

loss_fn = torch.nn.CrossEntropyLoss()


since = time.time()
train_loss = []
valid_loss = []
train_acc = []
val_acc = []

for epoch in range(num_epochs):

    tl, ta = training(model_ft, trainloader, optimizer_ft, scheduler, loss_fn, acc_fns, device, TRAIN_SIZE)
    vl, va, conf_mat = validation(model_ft, validloader, loss_fn, acc_fns, confusion_matrix, device, VALID_SIZE)
    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)

    if (epoch + 1) % 10 == 0:
        checkpoint = Path('model_checkpoints/')
        checkpoint.mkdir(exist_ok=True)
        torch.save(model_ft.state_dict(), checkpoint / f'{model_ft.__class__.__name__}_epoch_{epoch}.pt')

    printstr = f'Epoch: {epoch} , Train loss: {round(tl, 3)}, Val loss: {round(vl, 3)}, Train accs: {[round(t_a, 3) for t_a in ta]}, Val accs: {[round(v_a, 3) for v_a in va]}'
    tqdm.write(printstr)


time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


train_acc = np.array(train_acc)
val_acc = np.array(val_acc)


train_results = pd.DataFrame(train_acc)
train_results[4] = train_loss
train_results.columns = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'F1 Micro', 'Loss']
train_results.index.name = 'Epoch'

val_results = pd.DataFrame(val_acc)
val_results[4] = valid_loss
val_results.columns = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'F1 Micro', 'Loss']
val_results.index.name = 'Epoch'

train_results.to_csv(f'model_results/{model_ft.__class__.__name__}_train_results.csv')
val_results.to_csv(f'model_results/{model_ft.__class__.__name__}_valid_results.csv')


plt.figure(figsize=(14, 8))
plt.ylim(0, 1.5)
sns.lineplot(list(range(len(train_loss))), train_loss)
sns.lineplot(list(range(len(valid_loss))), valid_loss)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(['Train', 'Val'], fontsize=16)
plt.title('Loss vs Epoch', fontsize=20)
plt.savefig(f'{model_ft.__class__.__name__}_loss.jpg')


acc_names = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'F1 Micro']
for idx, acc_name in enumerate(acc_names):
    plt.figure(figsize=(14, 8))
    sns.lineplot(list(range(len(train_acc[:, idx]))), train_acc[:, idx])
    sns.lineplot(list(range(len(val_acc[:, idx]))), val_acc[:, idx])
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(acc_name, fontsize=18)
    plt.legend(['Train', 'Val'], fontsize=16)
    plt.title(f'{acc_name} vs Epoch', fontsize=20)
    plt.savefig(f'{model_ft.__class__.__name__}_accuracy.jpg')

labels = ['Healthy', 'Multiple', 'Rust', 'Scab']
plt.figure(figsize=(15, 10))
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.title('Confusion Matrix', fontsize=20)
plt.savefig(f'{model_ft.__class__.__name__}_confusion.jpg')
