import os

# PyTorch libraries
import torch
import torch.nn as nn


from .resnet import ResNet9, ResNet18
from .vgg import VGG7, Lenet5
from .snn.initialize import snn_registry

nn_registry = {
    "resnet9": ResNet9,
    "resnet18": ResNet18,

    "vgg7": VGG7,
    "vgg7_vb": VGG7,

    "lenet": Lenet5
}



def init_snn(config, logger):
    # initialize the bnn_model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.model](config)
    # snn = nn_registry[config.snn_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    snn = snn_registry[config.snn_model](config)

    # if os.path.exists(config.full_weight_dir):
    #     logger.info("--- Load pre-trained full precision model. ---")
    #     state_dict = torch.load(config.full_weight_dir)
    #     full_model.load_state_dict(state_dict)
    # else:
    logger.info("--- Train model from scratch. ---")
    full_model.apply(init_weights)

    snn.load_state_dict(full_model.state_dict())

    snn = snn.to(config.device)
    return snn


def init_full_model(config, logger):
    # initialize the qnn_model
    logger.info("--- Train full precision model from scratch. ---")
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    full_model.apply(init_weights)

    if config.freeze_fc:
        full_model.freeze_final_layer()

    full_model = full_model.to(config.device)
    return full_model


def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data, a=-1, b=1)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight.data)

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)


