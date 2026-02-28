import torch


class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, training):
        if not training:
            return x

        # if we are at training mode, we have to make the mask first
        self.mask = (torch.rand_like(x) > self.p).float()

        # scaling factor
        scale = 1 / (1 - self.p)

        not_dropped_nuerons = x * self.mask
        scaled_neurons = not_dropped_nuerons * scale
        return scaled_neurons


    def backward(self, grad):
        scale = 1 / (1 - self.p)
        return grad * self.mask * scale