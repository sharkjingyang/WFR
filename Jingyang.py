import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import math

# make data set
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cvt = lambda x: x.to(device, non_blocking=True)

nex = 1000
# x=torch.linspace(-5.0,5.0,nex).reshape(-1,1)
# x=cvt(x)
# y_target=torch.pow(x,3)-3*torch.pow(x,2)+torch.randn(x.size()).to(device)
# y_target=np.sqrt(2*math.pi)*torch.exp(-x**2/2)
# y_target=cvt(y_target)

x = torch.rand((nex, 1))
x = cvt(x)
y_target = torch.randn((nex, 1))
y_target = cvt(y_target)

parser = argparse.ArgumentParser('practice')
parser.add_argument("--nWlayer", type=int, default=1, help="num of Wasserstein_function layers")
parser.add_argument("--nWwidth", type=int, default=20, help="width of Wasserstein_function layers")
parser.add_argument("--lr", type=int, default=0.1, help="learning rate")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
args = parser.parse_args()


class Wasserstein_function(nn.Module):
    def __init__(self, nres, width, nex):
        super(Wasserstein_function, self).__init__()

        self.nres = nres
        self.nex = nex
        self.h = 1 / self.nres
        self.width = width
        self.act = nn.ReLU(True)
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(1, self.width, bias=True))
        for i in range(self.nres):
            self.layers.append(nn.Linear(self.width, self.width, bias=True))
        self.layers.append(nn.Linear(self.width, 1, bias=True))
        self.layers.append(nn.Linear(self.nex, 1, bias=True))

    def forward(self, x):

        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.nres):
            x = x + self.act(self.layers[i].forward(x))
        x = self.layers[self.nres + 1].forward(x)
        x = torch.reshape(x, (1, -1))
        x = self.layers[-1].forward(x)

        return x


class Generator(nn.Module):
    def __init__(self, nres, width):
        super(Generator, self).__init__()

        self.nres = nres
        self.h = 1 / self.nres
        self.width = width
        self.act = nn.Sigmoid()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(1, self.width, bias=True))
        for i in range(self.nres):
            self.layers.append(nn.Linear(self.width, self.width, bias=True))
        self.layers.append(nn.Linear(self.width, 1, bias=True))

    def forward(self, x):

        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.nres):
            x = x + self.act(self.layers[i].forward(x))
        x = self.layers[self.nres + 1].forward(x)
        return x


if __name__ == '__main__':
    # Initialize Wasserstein_function
    nres = args.nWlayer
    width = args.nWwidth
    nex = 1000
    W_dis = Wasserstein_function(nres=nres, width=width, nex=nex)
    W_dis = W_dis.to(device)

    # Initialize Generator
    nres = 1
    width = 20
    G = Generator(nres=nres, width=width)
    G = G.to(device)

    G_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.001)
    W_optimizer = torch.optim.RMSprop(W_dis.parameters(), lr=0.001)
    # Loss=nn.MSELoss()

    for G_itr in range(10000):

        for W_itr in range(10):
            W_optimizer.zero_grad()
            # loss_W=-torch.mean(W_dis(y_target)-W_dis(G(x)))
            loss_W = -W_dis(y_target) + W_dis(G(x))
            loss_W.backward()
            W_optimizer.step()
            for p in W_dis.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            if W_itr == 9:
                print("iter " + str(G_itr) + "is loss is " + str(loss_W))

        # loss_G=Loss(G(x),y_target)
        G_optimizer.zero_grad()
        # loss_G=-torch.mean(W_dis(G(x)))
        loss_G = -W_dis(G(x))
        loss_G.backward()
        G_optimizer.step()

        if G_itr % 10 == 0:
            plt.cla()
            plt.scatter(x.cpu(), y_target.cpu(), s=2)
            plt.plot(x.cpu().numpy(), G(x).data.cpu().numpy(), 'r-', lw=2)
            plt.show()
            plt.hist(G(x).data.cpu().numpy(), bins=100, density=True)
            plt.show()
            # plt.savefig("/home/chenjiaheng/OT flow/experiments/cnf/toy/")
            # plt.close()