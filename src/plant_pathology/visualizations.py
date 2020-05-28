import torch


def compute_saliency_maps(X, y, model):
    model.eval()
    X.requires_grad_()
    scores = model(X).gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones_like(scores))
    saliency, _ = torch.max(X.grad.abs(), 1)
    return saliency


def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    scores = model(img)
    s_y = scores[:, target_y] - l2_reg * img.square().sum()
    s_y.backward()

    with torch.no_grad():
        img += learning_rate * img.grad
        img.grad.zero_()
