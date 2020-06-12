import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ann

class LinearMasked(nn.Linear):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearMasked, self).__init__(in_features, out_features, bias=True)

    def forward(self, input, mask):
        full_batch = []
        for x, m in zip(input.split(1), mask.split(1)):
            ids = ann.Variable(m.nonzero()[:, 1])
            X = x.index_select(1, ids)
            W = self.weight.index_select(1, ids)
            #print(X.shape, W.shape, self.weight.shape)
            full_batch.append(F.linear(X, W, self.bias))
        return torch.cat(full_batch, 0)
        #return F.linear(input, self.weight, self.bias)



def make_nn(input_size, output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = []
    insize = input_size
    for outsize in hidden_sizes:
        layers.append(nn.Linear(insize, outsize))
        layers.append(nn.LeakyReLU())
        insize = outsize
    if isinstance(output_size, tuple):
        return nn.Sequential(*layers), [ nn.Linear(insize, k) for k in output_size]
    return nn.Sequential(*layers)
    #layers.append(nn.Linear(insize, output_size))
    #return nn.Sequential(*layers)


def MultivariateNormalDiag(mu, scale_diag):
    if len(scale_diag.size)==1:
        scale = torch.diag(scale_diag)
    elif len(scale_diag.size)==2:
        scale = torch.stack([
            torch.diag(xs.squeeze(0)) for xs in scale_diag.split(1, dim=0)])
    else:
        raise Exception("Not implemented")
    return torch.distributions.MultivariateNormal(mu, scale_tril=scale)
