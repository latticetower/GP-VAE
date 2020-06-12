import torch
import torch.nn as nn

import tensorflow as tf
from utils import make_nn, MultivariateNormalDiag


class DiagonalEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(64, 64), **kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(DiagonalEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_nn(input_size, (z_size, z_size), hidden_sizes)

    def __call__(self, x):
        output = self.net(x)
        mu = self.mu_layer(output)
        logvar = self.logvar_layer(output)
        return MultivariateNormalDiag(mu, F.softplus(logvar))


class JointEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(64, 64), window_size=3, transpose=False, **kwargs):
        """ Encoder with 1d-convolutional network and factorized Normal posterior
            Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size), hidden_sizes, window_size)
        self.transpose = transpose

    def __call__(self, x):
        output = self.net(x)
        mu = self.mu_layer(output)
        logvar = self.logvar_layer(output)
        if self.transpose:
            num_dim = len(x.shape)
            mu = torch.transpose(mu, num_dim-1, num_dim-2)
            logvar = torch.transpose(logvar, num_dim-1, num_dim-2)
            return MultivariateNormalDiag(mu, F.softplus(logvar))
        return MultivariateNormalDiag(mu, F.softplus(logvar))


class BandedJointEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(64, 64), window_size=3, data_type=None, **kwargs):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            Used by GP-VAE with proposed banded covariance matrix
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param data_type: needed for some data specific modifications, e.g:
                tf.nn.softplus is a more common and correct choice, however
                tf.nn.sigmoid provides more stable performance on Physionet dataset
        """
        super(BandedJointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size*2), hidden_sizes, window_size)
        self.data_type = data_type

    def __call__(self, x):
        mapped = self.net(x)
        batch_size = mapped.size[0]
        time_length = mapped.size[1]
        # Obtain mean and precision matrix components
        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        #perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_mean = torch.transpose(mu, num_dim - 1, num_dim - 2)
        mapped_covar = torch.transpose(logvar, num_dim - 1, num_dim - 2)
        if self.data_type == 'physionet':
            mapped_covar = F.sigmoid(mapped_covar)
        else:
            mapped_covar = F.softplus(mapped_covar)
        mapped_reshaped = mapped_covar.reshape(batch_size, self.z_size, 2*time_length)
        
        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)
        #unfinished!
        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = mapped_reshaped[:, :, :-1].reshape(-1)
            prec_sparse = torch.sparse.FloatTensor(
                idxs_all, mapped_values, dense_shape)
            #prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            #prec_sparse = tf.sparse.reorder(prec_sparse)
            #next
            prec_tril = torch.zeros(prec_sparse.dense_shape, dtype=tf.float32) + prec_sparse
            eye = torch.eye(prec_tril.shape[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            
            cov_tril = torch.triangular_solve(eye, prec_tril, lower=False)
            cov_tril = torch.where(torch.is_finite(cov_tril), cov_tril, torch.zeros_like(cov_tril))
            #prec_tril = tf.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            #eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            #prec_tril = prec_tril + eye
            #cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            #cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))
        num_dim = len(cov_tril.shape)
        #perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        #cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        cov_tril_lower = torch.transpose(cov_tril, num_dim - 1, num_dim - 2)
        ##TODO: wtf?
        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, scale_tril=cov_tril_lower)
        return z_dist