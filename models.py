import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ann
#from encoders import DiagonalEncoder

from utils import LinearMasked


class VAE(nn.Module):
    def __init__(self, nsize):
        super(VAE, self).__init__()

        self.fc1 = LinearMasked(nsize, nsize//2)
        self.fc21 = nn.Linear(nsize//2, nsize//4)
        self.fc22 = nn.Linear(nsize//2, nsize//4)
        self.fc3 = nn.Linear(nsize//4, nsize//2)
        self.fc4 = nn.Linear(nsize//2, nsize)

    def encode(self, x, xids):
        xx = self.fc1(x, xids)
        h1 = F.relu(xx)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, xids):
        mu, logvar = self.encode(x, xids)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

