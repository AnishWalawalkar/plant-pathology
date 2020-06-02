from sklearn.metrics import roc_auc_score
import numpy as np


def comp_metric(targs, preds, labels=range(4)):
    # One-hot encode targets
    targs = convert_to_one_hot(targs)
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])


def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])


def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])


def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])


def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])


def convert_to_one_hot(X):
    X = X.astype('int').reshape(X.shape[0], 1)
    z = np.max(X) + 1
    return np.squeeze(np.eye(z)[X])