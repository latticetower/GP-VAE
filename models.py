import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ann
#from encoders import DiagonalEncoder

from utils import LinearMasked, reduce_logmeanexp


# class VAE(nn.Module):
#     def __init__(self, nsize):
#         super(VAE, self).__init__()
# 
#         self.fc1 = LinearMasked(nsize, nsize//2)
#         self.fc21 = nn.Linear(nsize//2, nsize//4)
#         self.fc22 = nn.Linear(nsize//2, nsize//4)
#         self.fc3 = nn.Linear(nsize//4, nsize//2)
#         self.fc4 = nn.Linear(nsize//2, nsize)
# 
#     def encode(self, x, xids):
#         xx = self.fc1(x, xids)
#         h1 = F.relu(xx)
#         return self.fc21(h1), self.fc22(h1)
# 
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
# 
#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))
# 
#     def forward(self, x, xids):
#         mu, logvar = self.encode(x, xids)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

class VAE(nn.Module):
    def __init__(self, latent_dim, data_dim, time_length,
                 encoder_sizes=(64, 64), encoder=DiagonalEncoder,
                 decoder_sizes=(64, 64), decoder=BernoulliDecoder,
                 image_preprocessor=None, beta=1.0, M=1, K=1, **kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            
            :param encoder_sizes: layer sizes for the encoder network
            :param encoder: encoder model class {Diagonal, Joint, BandedJoint}Encoder
            :param decoder_sizes: layer sizes for the decoder network
            :param decoder: decoder model class {Bernoulli, Gaussian}Decoder
            
            :param image_preprocessor: 2d-convolutional network used for image data preprocessing
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.time_length = time_length

        self.encoder = encoder(data_dim, latent_dim, encoder_sizes, **kwargs)
        self.decoder = decoder(latent_dim, data_dim, decoder_sizes)
        self.preprocessor = image_preprocessor

        self.beta = beta
        self.K = K
        self.M = M
        self.prior = None

    def encode(self, x):
        if self.preprocessor is not None:
            x_shape = x.shape
            new_shape = [x_shape[0] * x_shape[1]] + list(self.preprocessor.image_shape)
            x_reshaped = x.reshape(new_shape)
            x_preprocessed = self.preprocessor(x_reshaped)
            x = x_preprocessed.reshape(x_shape)
        return self.encoder(x)
        #x = tf.identity(x)  # in case x is not a Tensor already...
        
    def decode(self, z):
        #z = tf.identity(z)  # in case z is not a Tensor already...
        return self.decoder(z)

    def __call__(self, inputs):
        return self.decode(self.encode(inputs).sample()).sample()
    
    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = torch.normal(0, 1, size=(num_samples, self.latent_dim))
        return self.decode(noise)
    
    def _get_prior(self):
        if self.prior is None:
            self.prior = MultivariateNormalDiag(
                torch.zeros(self.latent_dim),
                torch.ones(self.latent_dim)
            )
        return self.prior
    
    def compute_nll(self, x, y=None, m_mask=None):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None:
            y = x
        z_sample = self.encode(x).sample()
        x_hat_dist = self.decode(z_sample)
        nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if m_mask is not None:
            m_mask = m_mask.astype(torch.bool)
            nll = torch.where(m_mask, nll, torch.zeros_like(nll))  # !!! inverse mask, set zeros for observed
        return nll.sum()

    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_mean = self.encode(x).mean()
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = torch.round(x_hat_mean)
        mse = (x_hat_mean - y)**2
        if m_mask is not None:
            m_mask = m_mask.astype(torch.bool)
            mse = torch.where(m_mask, mse, torch.zeros_like(mse))  # !!! inverse mask, set zeros for observed
        return mse.sum()
    
    def _compute_loss(self, x, m_mask=None, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        #x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
        x = x.repeat(self.M*self.K, 1, 1)
        if m_mask is not None:
            #m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = m_mask.repeat(self.M*self.K, 1, 1)
            m_mask = m_mask.astype(torch.bool)
            #m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
            #m_mask = tf.cast(m_mask, tf.bool)
        pz = self._get_prior()
        qz_x = self.encode(x)
        z = qz_x.sample()
        px_z = self.decode(z)
        print(x.shape, px_z, z.shape)
        nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if m_mask is not None:
            nll = torch.where(m_mask, torch.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
        nll = nll.sum(1).sum(1)
        #nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)
        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)
            
            weights = -nll - kl  # shape=(M*K*BS)
            weights = weights.reshape(self.M, self.K, -1)  # shape=(M, K, BS)
            
            elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = torch.mean(elbo)  # scalar
        else:
            # if K==1, compute KL analytically
            kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)  # shape=(M*K*BS)
            
            elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
            elbo = torch.mean(elbo)  # scalar
            
        if return_parts:
            nll = torch.mean(nll)  # scalar
            kl = torch.mean(kl)  # scalar
            return -elbo, nll, kl
        else:
            return -elbo

    def compute_loss(self, x, m_mask=None, return_parts=False):
        del m_mask
        return self._compute_loss(x, return_parts=return_parts)
    
    def kl_divergence(self, a, b):
        return torch.distributions.kl.kl_divergence(a, b)
    
    # def get_trainable_vars(self):
    #     #TODO: check if useful??
    #     self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
    #                       tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
    #     return self.trainable_variables


class HI_VAE(VAE):
    """ HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return)


class GP_VAE(HI_VAE):
    def __init__(self, *args, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
        """ Proposed GP-VAE model with Gaussian Process prior
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        if isinstance(self.encoder, JointEncoder):
            self.encoder.transpose = True

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None
        
    def decode(self, z):
        num_dim = len(z.shape)
        assert num_dim > 2
        return self.decoder(torch.transpose(z, num_dim - 1, num_dim - 2))
    
    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(
                    #tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
                    torch.expand_dims(kernel_matrices[i], 0).repeat(multiplier, 1, 1))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dim
            self.prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.latent_dim, self.time_length),
                covariance_matrix=kernel_matrix_tiled
            )

        return self.prior

    # def kl_divergence(self, a, b):