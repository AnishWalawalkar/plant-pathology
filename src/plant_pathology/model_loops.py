import numpy as np
import torch
from tqdm import tqdm


def training(model, data_loader, optim, scheduler, loss_fn, acc_fns, device, train_size):
    running_loss = 0
    preds_for_acc = np.array([]).reshape(0, 4)
    labels_for_acc = []
    lrs = []

    pbar = tqdm(total=len(data_loader), desc='Training')

    for idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optim.zero_grad()
        scores = model(images)
        loss = loss_fn(scores, labels)
        loss.backward()
        optim.step()
        scheduler.step()
        lr_step = optim.state_dict()["param_groups"][0]["lr"]
        lrs.append(lr_step)

        running_loss += loss.item() * labels.shape[0]

        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
        preds_for_acc = np.concatenate((preds_for_acc, scores.cpu().detach().numpy()), 0)

        pbar.update()
    
    pbar.close()
    return running_loss / train_size, [acc_fn(labels_for_acc, preds_for_acc) for acc_fn in acc_fns], lrs


def validation(model, data_loader, loss_fn, acc_fns, confusion_matrix, device, valid_size):
    running_loss = 0
    preds_for_acc = np.array([]).reshape(0, 4)
    labels_for_acc = []

    pbar = tqdm(total=len(data_loader), desc='Validation')

    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            model.eval()
            scores = model(images)
            loss = loss_fn(scores, labels)

            running_loss += loss.item() * labels.shape[0]
            labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
            preds_for_acc = np.concatenate((preds_for_acc, scores.cpu().detach().numpy()), 0)

            pbar.update()

        conf_mat = confusion_matrix(labels_for_acc, np.argmax(preds_for_acc, axis=1))

    pbar.close()
    return running_loss / valid_size, [acc_fn(labels_for_acc, preds_for_acc) for acc_fn in acc_fns], conf_mat


def testing(model, data_loader, device):
    preds_for_output = np.zeros((1, 4))

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader))
        for _, images in enumerate(data_loader):
            images = images.to(device)
            model.eval()
            scores = model(images)
            preds_for_output = np.concatenate((preds_for_output, scores.cpu().detach().numpy()), 0)
            pbar.update()

    pbar.close()
    return preds_for_output
