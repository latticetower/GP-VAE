import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
# Decoders
from utils import make_nn


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = make_nn(input_size, output_size, hidden_sizes)
        
    def __call__(self, x):
        pass

class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return torch.distributions.Bernoulli(logits=mapped)
    

class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
    def __call__(self, x):
        mean = self.net(x)
        var = torch.ones_like(mean)
        return torch.distributions.Normal(mean, var)