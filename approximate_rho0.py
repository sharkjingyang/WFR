import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import *
import matplotlib.pyplot as plt
import lib.toy_data as toy_data
import time

class density_net(nn.Module):
    def __init__(self, d, width, ngpu=1):
        super(density_net, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            nn.Linear(d, width),
            nn.ReLU(True),
            nn.Linear(width, width),
            nn.ReLU(True),
            nn.Linear(width, width),
            nn.ReLU(True),
            nn.Linear(width, width),
            nn.ReLU(True),
            nn.Linear(width, 1),
        )
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

def get_density(x, d, niter, device):

    prec = torch.float64
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    x0 = cvt(torch.from_numpy(x))

    d_net = density_net(d=d, width=128)
    d_net = d_net.to(prec).to(device)

    optimizer = torch.optim.Adam(d_net.parameters(), lr=0.0001, weight_decay=0.0)

    for it in range(niter):
        # print(it)
        y = cvt(torch.randn(20000, d))
        l1 = torch.mean(torch.log(1.0 + torch.exp(d_net(y))))
        l2 = torch.mean(torch.log(1.0 + torch.exp(d_net(x0)))) - torch.mean(d_net(x0))
        loss = l1 + l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == niter-1:
            y0 = d_net(x0).detach().cpu().numpy()
            x1 = x0.detach().cpu().numpy()
            w = np.sum(np.power(x1, 2), 1, keepdims=True)
            p = np.exp(-w/2+y0)/((2*pi)**(d/2))
            d_net.eval()
            return p, d_net


if __name__ == "__main__":
    niter = 50
    nSamples = 4096
    d=2
    prec = torch.float64
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    x0 = toy_data.inf_train_gen('moons', batch_size=nSamples,require_density=False,device=device)
    plt.plot(x0[:,0],x0[:,1])
    plt.show()
    # start=time.time()
    # p = get_density(x0,d,niter,device)
    # end=time.time()
    # p1= p
    # print("Time is "+str(end-start))
