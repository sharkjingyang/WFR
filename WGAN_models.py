from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt

import lib.toy_data as toy_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import *



class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu=1):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.Tanh(True),
            nn.Linear(ngf, ngf),
            nn.Tanh(True),
            nn.Linear(ngf, ngf),
            nn.Tanh(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu=1):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)


class W1_D(nn.Module):
    def __init__(self, d, ndf, ngpu=1, nex=4096):
        super(W1_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            nn.Linear(d, ndf),
            nn.LeakyReLU(),
            nn.Linear(ndf, ndf),
            nn.LeakyReLU(),
            nn.Linear(ndf, ndf),
            nn.LeakyReLU(),
            nn.Linear(ndf, 1),
        )
        self.nex = nex
        self.main = main
        self.linear = nn.Linear(nex, 1)

    def forward(self, input):
        output = self.main(input)
        # output = self.linear(output.reshape(1, self.nex))

        return output

class W1_G(nn.Module):
    def __init__(self, d, ngf, ngpu=1):
        super(W1_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            nn.Linear(d, ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf),
            nn.ReLU(),
            nn.Linear(ngf, 1),
        )
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

def compute_gradient_penalty(D, real_samples, fake_samples, d,device):
    alpha = torch.rand((real_samples.size(0), d)).to(device)
    interpolates = (torch.mul(alpha, real_samples) + torch.mul((1 - alpha), fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gradient_penalty

if __name__ == "__main__":
    niter = 50000
    nSamples = 4096
    lambda0 = 0.1
    d=1
    w = [1 / sqrt(2 * pi) * exp(-x ** 2 / 2) for x in np.linspace(-10, 10, 1000)]
    prec = torch.float64
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    x0, rho_x0 = toy_data.inf_train_gen('1d', batch_size=nSamples, require_density=False)
    x0 = cvt(torch.from_numpy(x0))
    rho_x0 = cvt(torch.from_numpy(rho_x0))
    weights = cvt(torch.ones(nSamples, 1)/x0.shape[0])
    netD = W1_D(d=1, ndf=64)
    netD = netD.to(prec).to(device)
    netG = W1_G(d=1, ngf=64)
    netG = netG.to(prec).to(device)

    i = 0
    # optim = torch.optim.Adam(netD.parameters(), lr=0.001, weight_decay=0)
    # optimD = torch.optim.RMSprop(netD.parameters(), lr=0.0005)
    # optimG = torch.optim.RMSprop(netG.parameters(), lr=0.0005)
    optimD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=[0.0, 0.9])
    optimG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=[0.0, 0.9])


    x0 = x0.unsqueeze(-1)

    for epoch in range(niter):
        if epoch < 25 or epoch % 100 == 0:
            Diters = 5
        else:
            Diters = 5

        if epoch%100==0:
            x0, rho_x0 = toy_data.inf_train_gen('1d', batch_size=nSamples, require_density=False)
            x0 = cvt(torch.from_numpy(x0))
            x0 = x0.unsqueeze(-1)

        netG.zero_grad()
        x1 = netG(x0)

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False


        i = 0
        while i < Diters:
            i = i+1
            y = cvt(torch.randn(nSamples, d))
            # for p in netD.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            fx = netD(x1).squeeze()
            fy = netD(y).squeeze()
            gp = compute_gradient_penalty(netD, real_samples=y, fake_samples=x1, device=device)
            lossD = torch.mean(fx)-torch.mean(fy)+lambda0*gp
            optimD.zero_grad()
            lossD.backward(retain_graph=True)
            optimD.step()

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True


        fx1 = netD(x1).squeeze()
        lossG = torch.mean(-fx1)
        lossG.backward()
        optimG.step()

        if epoch%50==0:
            print("lossD:{}, lossG:{}".format(lossD, lossG))
            # with torch.no_grad():
            #     print(netD.main[0].weight) # 参数很多集中在两个极端1,-1上
            with torch.no_grad():
                test_x0, rho_x0 = toy_data.inf_train_gen('1d', batch_size=200000,require_density=False)
                test_x0 = cvt(torch.from_numpy(test_x0)).unsqueeze(-1)
                # test_x0 = cvt(torch.linspace(-5, 5, 4096))
                # test_x0 = test_x0.unsqueeze(-1)
                y0 = netG(test_x0).squeeze()

                # plt.hist(test_x0.detach().cpu().numpy(), bins=100, density=True)
                # plt.ylim(0, 0.5)
                # plt.xlim(-5, 5)
                # plt.show()
                plt.hist(y0.detach().cpu().numpy(), bins=100, density=True, label='target, epoch:{}'.format(epoch))
                plt.plot(np.linspace(-10, 10, 1000), w, 'r')
                # plt.ylim(0, 0.5)
                # plt.xlim(-5, 5)
                plt.legend()
                plt.show()

