import torch


def clip_grad_by_global_norm(gradients, max_norm):
    total_norm = 0.0

    for g in gradients:
        total_norm += torch.sum(g**2)

    global_norm = torch.sqrt(total_norm)
    scaling_factor = 1.0

    if global_norm.item() > max_norm:
        scaling_factor = max_norm / global_norm.item()

    return [g * scaling_factor for g in gradients]