import torch


class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.

        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p = p
        self.mask = None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.

        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        if not training:
            return x

        # we create the mask here
        self.mask = (torch.rand_like(x) > self.p).float()

        # scaling factor
        scale = 1 / (1 - self.p)

        not_dropped_neurons = x * self.mask
        scaled_neurons = not_dropped_neurons * scale
        return scaled_neurons

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer."""
        return grad * self.mask * (1 / (1 - self.p))