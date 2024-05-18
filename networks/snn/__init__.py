from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn as nn

class StoNN(ABC, nn.Module):
    def __init__(self):
        super(StoNN, self).__init__()

    def all_params(self):
        for module in self.modules():
            if hasattr(module, "bias") and module.bias is not None:
                yield module.bias
            if hasattr(module, "weight") and module.weight is not None:
                yield module.weight
                if hasattr(module.weight, "_std"):
                    yield module.weight._std
            

    def std_dict(self):
        std_dict = OrderedDict()
        named_modules = self.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "_std"):
                continue

            std_dict[module_name + ".weight.std"] = module.weight._std

        return std_dict

    def load_std_dict(self, std_dict):
        for std_name, std in std_dict.items():
            exec("self.{:s} = std".format(std_name))

    def noise_power(self):
        pow = 0
        counter = 0
        named_modules = self.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "_std"):
                continue

            pow += ((module.weight._std)**2).mean().item()
            counter += 1

        pow /= counter

        return pow