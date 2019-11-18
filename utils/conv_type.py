from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

DenseConv = nn.Conv2d

from args import args as parser_args


class ChooseEdges(autograd.Function):
    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class MaskedCeil(autograd.Function):
    @staticmethod
    def forward(ctx, mask, prune_rate):
        output = mask.clone()
        _, idx = mask.flatten().abs().sort()
        p = int(prune_rate * mask.numel())

        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class SignedMaskedCeil(autograd.Function):
    @staticmethod
    def forward(ctx, mask, prune_rate):
        output = mask.clone()
        _, idx = mask.flatten().abs().sort()
        p = int(prune_rate * mask.numel())

        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1

        return output * mask.sign()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


# Not learning weights, learning mask
class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_mask(self):
        return self.mask.abs()

    def forward(self, x):
        mask = MaskedCeil.apply(self.clamped_mask, self.prune_rate)
        w = self.weight * mask
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


# Not learning weights, learning mask
class SignedMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_mask(self):
        return self.mask

    def forward(self, x):
        mask = SignedMaskedCeil.apply(self.clamped_mask, self.prune_rate)
        w = self.weight * mask
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        output = (torch.rand_like(mask) < mask).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


# Not learning weights, learning mask
class SampleMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.mask_init_constant is not None:
            self.mask.data = torch.ones_like(self.mask) * c
        else:
            nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

    @property
    def clamped_mask(self):
        return torch.sigmoid(self.mask)

    def forward(self, x):
        mask = StraightThroughBinomialSample.apply(self.clamped_mask)
        w = self.weight * mask
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


class FixedMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_mask(self):
        output = self.clamped_mask().clone()
        _, idx = self.clamped_mask().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_mask().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.mask = torch.nn.Parameter(output)
        self.mask.requires_grad = False

    def clamped_mask(self):
        return self.mask.abs()

    def get_weight(self):
        return self.weight * self.mask

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class FixedSignedMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_mask(self):
        with torch.no_grad():
            output = self.mask.clone()
            _, idx = self.mask.flatten().abs().sort()
            p = int(self.prune_rate * self.clamped_mask().numel())
            flat_oup = output.flatten()
            flat_oup[idx[:p]] = 0
            flat_oup[idx[p:]] = 1
            output *= self.mask.sign()

        self.mask = torch.nn.Parameter(output)
        self.mask.requires_grad = False

    def get_weight(self):
        return self.weight * self.mask

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
