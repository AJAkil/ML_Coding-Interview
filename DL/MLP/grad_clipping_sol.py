import torch

def clip_gradients_by_global_norm(gradients: list[torch.Tensor], max_norm: float) -> list[torch.Tensor]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient tensors
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient tensors
    """

    total_norm = 0.0

    for g in gradients:
        total_norm += torch.sum(g**2)   
    
    global_norm = torch.sqrt(total_norm)

    scaling_factor = 1.0

    if global_norm.item() > max_norm:
        scaling_factor = max_norm / (global_norm.item())
    
    return [g * scaling_factor for g in gradients]

def grad_clipping_with_torch(model: nn.Module, max_norm: float) -> None:
    """
    Clip gradients by global norm using torch.nn.utils.clip_grad_norm_.
    
    Args:
        model: PyTorch model
        max_norm: Maximum allowed global norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)