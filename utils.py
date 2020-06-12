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
        return [nn.Sequential(*layers)] + [ nn.Linear(insize, k) for k in output_size]
    layers.append(nn.Linear(insize, output_size))
    return nn.Sequential(*layers)
    #layers.append(nn.Linear(insize, output_size))
    #return nn.Sequential(*layers)

def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of
          one 1d-convolutional layer that utilizes temporal dependences,
          fully connected network
        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """
    cnn_layer = [torch.nn.Conv1d(input_size, hidden_sizes[0], kernel_size=kernel_size, padding=1)]
    layers = [cnn_layer]
    for i, h in zip(hidden_sizes, hidden_sizes[1:]):
        layes.extend([
            nn.Linear(i, h),
            nn.ReLU()
        ])
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [ nn.Linear(hidden_sizes[-1], o) for o in output_size ]
    
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return nn.Sequential(*layers)
    #cnn_layer = [tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size,
    #                                    padding="same", dtype=tf.float32)]
    #layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
    #          for h in hidden_sizes[1:]]
    #layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    #return tf.keras.Sequential(cnn_layer + layers)


def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = torch.max(x, axis=axis, keepdim=True)
    return torch.log(torch.mean(
            torch.exp(x - x_max), axis=axis, keepdim=True) + eps) + x_max


class MultivariateNormalDiag(torch.distributions.MultivariateNormal):
    def __init__(self, mu, scale_diag):
        if len(scale_diag.shape) == 1:
            scale = torch.diag(scale_diag)
        elif len(scale_diag.shape) == 2:
            scale = torch.stack([
                torch.diag(xs.squeeze(0)) for xs in scale_diag.split(1, dim=0)])
        else:
            shape = scale_diag.shape
            scale = torch.stack([
                torch.diag(xs.squeeze())
                for xs in scale_diag.reshape(-1, shape[-1]).split(1)
            ]).reshape(shape + shape[-1:])

        super(MultivariateNormalDiag, self).__init__(mu, scale_tril=scale)
