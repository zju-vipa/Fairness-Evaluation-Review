"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize_slide(mu, logvar):
    z = mu
    std = logvar.div(2).exp()
    eps_plus = torch.arange(0, 3.0, 0.1)
    eps_minus = torch.arange(-3.0, 0, 0.1)
    z = torch.empty(mu.shape[0], mu.shape[1], eps_plus.shape[0])
    for i in range(eps_plus.shape[0]):  # z_dim
        z[:, :, i] = mu + std * eps_plus[i]
    return z

def reparametrize(mu, logvar):  # mu: B, z_dim      logvar: B, z_dim
    std = logvar.div(2).exp()  # B, z_dim    ∈（0.05,0.94）
    eps = Variable(std.data.new(std.size()).normal_())
    z = mu + std*eps
    return z


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def _encode(self, x):
        x = self.encoder(x)
        return x

    def _decode(self, z):
        return self.decoder(z)

    def forward(self, x, repara_name='reparametrize', num_eps = 10, decrement = 0.5, zi = 0, epsi = 0):                       # B, 3, img_size, img_size
        distributions = self._encode(x)         # B, z_dim*2
        mu = distributions[:, :self.z_dim]      # B, z_dim  
        logvar = distributions[:, self.z_dim:]  # B, z_dim
        if repara_name=='reparametrize':
            std = logvar.div(2).exp()  # B, z_dim    ∈（0.05,0.94）
            eps = Variable(std.data.new(std.size()).normal_())
            z = mu + std*eps
            x_recon = self._decode(z)          # B, 3, img_size, img_size
        elif repara_name=='reparametrize_slide':
            std = logvar.div(2).exp()
            eps = mu.clone().unsqueeze(-1).repeat(1, 1, 2 * num_eps)
            for m in range(mu.shape[0]):
                for n in range(mu.shape[1]):
                    eps_front = torch.arange(0 - num_eps * decrement * torch.abs(std[m, n]).item(), 0, decrement * torch.abs(std[m, n]).item())
                    eps_back = torch.arange(0, 0 + num_eps * decrement * torch.abs(std[m, n]).item(), decrement * torch.abs(std[m, n]).item())
                    row_eps = torch.cat((eps_front, eps_back))
                    eps[m, n, :] = row_eps  # B, z_dim, r_dim

            x_recon = torch.empty(mu.shape[0], self.z_dim, eps.shape[2], self.nc, 64, 64)
            z = mu.clone()
            for i in range(self.z_dim):  # z_dim
                for j in range(eps.shape[2]):  # r_dim
                    z = mu.clone()
                    z[:, i] = mu[:, i] + eps[:, i, j]   # B, z_dim
                    x_recon_ = self._decode(z)          # B, 3, img_size, img_size
                    x_recon[:, i, j, :, :, :] = F.sigmoid(x_recon_)  # B, z_dim, r_dim, 3, img_size, img_size
        elif repara_name=='reparametrize_pert':
            # for only one image
            x_recon = torch.empty(self.z_dim, self.nc, 64, 64)
            std = logvar.div(2).exp()
            eps = mu.clone().unsqueeze(-1).repeat(1, 1, 2 * num_eps)
            for m in range(mu.shape[0]):
                for n in range(mu.shape[1]):
                    eps_front = torch.arange(0 - num_eps * decrement * mu[m, n].item(), 0, decrement * mu[m, n].item())
                    eps_back = torch.arange(0, 0 + num_eps * decrement * mu[m, n].item(), decrement * mu[m, n].item())
                    row_eps = torch.cat((eps_front, eps_back))
                    eps[m, n, :] = row_eps  # B, z_dim, r_dim     middle->num_eps
            z = mu.clone()                  # B, z_dim
            z[0][zi] = mu[0][zi] + eps[0][zi][epsi]     # B, z_dim
            x_recon_ = self._decode(z)      # B, 3, img_size, img_size
            x_recon = F.sigmoid(x_recon_)   # B, 3, img_size, img_size
        else:  
            pass
            
        return x_recon, mu, logvar


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, reparametrize=reparametrize):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
