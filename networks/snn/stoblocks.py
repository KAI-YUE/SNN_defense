import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

noise_std = 1.e-3

class StoConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(StoConv2d, self).__init__(*kargs, **kwargs)
        self._init_std()
    
    def _init_std(self):
        self.weight._std = torch.randn_like(self.weight, requires_grad=True)
        nn.init.normal_(self.weight._std, 0., noise_std)

        return self

    def _apply(self, fn):
        self.weight.data = fn(self.weight.data)
        self.weight._std.data = fn(self.weight._std.data)
        self.bias.data = fn(self.bias.data)

    def forward(self, input):
        mu = F.conv2d(input, self.weight, self.bias,
                      self.stride, self.padding, self.dilation)
        sigma_square = F.conv2d(input**2, self.weight._std**2, None, 
                      self.stride, self.padding, self.dilation)
        
        epsilon = torch.randn_like(mu)
        out = mu + (F.relu(sigma_square)).sqrt()*epsilon

        # out = mu

        return out

class StoLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(StoLinear, self).__init__(*kargs, **kwargs)
        self._init_std()

    def _init_std(self):
        self.weight._std = torch.randn_like(self.weight, requires_grad=True)
        nn.init.normal_(self.weight._std, 0., noise_std)

        return self

    def _apply(self, fn):
        self.weight.data = fn(self.weight.data)
        self.weight._std.data = fn(self.weight._std.data)

        if self.bias is not None:
            self.bias.data = fn(self.bias.data)

    def forward(self, input):

        mu = F.linear(input, self.weight, self.bias)
        sigma_square = F.linear(input**2, self.weight._std**2, None)
        
        epsilon = torch.randn_like(mu)
        out = mu + (F.relu(sigma_square)).sqrt()*epsilon
        out = mu

        return out