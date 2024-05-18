import copy

import torch

from networks import nn_registry
from src.fedalg import FedAlg

class FedAvg_SNN(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
        self.fed_lr = config.fed_lr
        self.tau = config.tau

        self.init_state = copy.deepcopy(self.model.state_dict())

        # archive the configuration
        self.config = config
    
    def client_grad(self, x, y):
        net_optimizer = torch.optim.SGD(self.model.all_params(), lr=self.fed_lr)
        for t in range(self.tau):
            out = self.model(x)
            risk = self.criterion(out, y)
            
            net_optimizer.zero_grad()
            risk.backward()
            net_optimizer.step()

        # now we simulate sampling a snn from a learned distribution
        sampled_snn = nn_registry[self.config.model](self.config)
        sampled_snn.load_state_dict(self.model.state_dict())
        sampled_snn = sampled_snn.to(self.config.device)
        
        sampled_snn_state_dict = sampled_snn.state_dict()

        named_modules = self.model.named_modules()
        next(named_modules)
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "_std"):
                continue
            
            noise = module.weight._std * torch.randn_like(sampled_snn_state_dict[module_name + ".weight"])
            sampled_snn_state_dict[module_name + ".weight"] += noise

        sampled_snn.load_state_dict(sampled_snn_state_dict)
        sampled_snn = sampled_snn.to(self.config.device)
            
        grad = []
        st = sampled_snn.state_dict()
        for w_name, w_val in st.items():
            grad.append((self.init_state[w_name] - w_val)/self.fed_lr)

        return grad

