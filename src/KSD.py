import torch
import time
from math import *

device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')

def KSD(x, w, h=1):
    # RBF kernel
    d = x.shape[1]
    n = x.shape[0]

    f = 0
    for i in range(d):
        # |x_i-x_j|^2
        f_temp = (x[:, i].unsqueeze(1).T - x[:, i].unsqueeze(1))**2
        f = f + f_temp

    # med2 = torch.sum(f)/(n*(n-1))
    # h = sqrt(med2/2/log(n))
    h = 0.46
    h2 = 1/h**2
    h4 = 1/h**4
    # print(h)
    g = torch.mm(x, x.T)  # x_ix_j
    w = torch.mm(w, w.T)  # w_iw_j
    matrix1 = torch.exp(-h2/2*f)*w
    A = matrix1 * (g - (h2 + h4) * f)
    # ksd = torch.sum(A)-torch.trace(A)+d*h2*(torch.sum(matrix1)-torch.trace(matrix1))
    ksd = torch.sum(A) + d * h2 * (torch.sum(matrix1))
    return ksd

def KSD_IMQ(x, w, c=1,l=1,beta=-0.5):
    d = x.shape[1]
    n = x.shape[0]
    f0 = 0
    w = torch.mm(w, w.T)
    g = torch.mm(x, x.T)
    for i in range(d):
        # |x_i-x_j|^2
        f_temp = (x[:, i].unsqueeze(1) - x[:, i].unsqueeze(1).T)**2
        f0 = f0 + f_temp
    f = f0/l/l+c*c
    gamma = -4*beta*(beta-1)/(l**4)*torch.pow(f,beta-2)*f0 + \
            2*beta/l/l*torch.pow(f, beta-1)*f0- \
             2*beta*d/l/l*torch.pow(f, beta-1)+ \
            torch.pow(f, beta)*g

    gamma = gamma*w

    return torch.sum(gamma)

def KSD_kappa(x, w, h=1):
    # RBF kernel
    d = x.shape[1]
    print(d)
    h2 = 1/h**2
    h4 = 1/h**4
    f = 0
    for i in range(d):
        # |x_i-x_j|^2
        f_temp = (x[:, i].unsqueeze(1).T - x[:, i].unsqueeze(1))**2
        f = f + f_temp
    w = torch.mm(w, w.T)  # w_iw_j
    matrix1 = torch.exp(-h2/2*f)*w
    A = matrix1 * (2*h4 * f)
    # ksd = torch.sum(A)-torch.trace(A)+d*h2*(torch.sum(matrix1)-torch.trace(matrix1))
    ksd = torch.sum(A) + (1/4-2*h2)* (torch.sum(matrix1))
    return ksd

def FSSD_IMQ(x, w, J=10, c=1,l=1,beta=-0.5):
    # IMQ kernel, FSSD
    d = x.shape[1]
    # h2 = 1/h/h
    y = torch.randn(J, d).to(device)

    k=0
    for i in range(d):
        k_temp = (x[:, i].unsqueeze(1) - y[:, i].unsqueeze(1).T)**2
        k = k + k_temp   # N*J
    k = k/l/l+c*c
    k1 = 2*beta/l/l*torch.pow(k,beta-1)
    k2 = torch.pow(k,beta)
    fssd = 0
    for i in range(d):
        z = k1*(x[:,i].unsqueeze(1)-y[:,i].unsqueeze(1).T)-x[:,i]*k2
        z = z*w
        fssd = fssd + torch.sum(z**2)
    return fssd

def FSSD(x, w, J=10, h=1):
    # RBF kernel, FSSD
    d = x.shape[1]
    h2 = 1/h/h
    # y = torch.randn(J, d).to(device)
    y = (torch.rand(J,d)*6-3).to(device)
    k=0
    for i in range(d):
        k_temp = (x[:, i].unsqueeze(1) - y[:, i].unsqueeze(1).T)**2
        k = k + k_temp   # N*J
    k = torch.exp(-h2/2*k)*w
    fssd = 0
    for i in range(d):
        z1 = (-1 - 2 / h / h) * x[:,i].unsqueeze(1) + 2 / h / h * y[:,i].unsqueeze(1).T
        z = z1*k
        fssd = fssd + torch.sum(z**2)
    return fssd/J


if __name__ == '__main__':
    # device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')
    #
    # a = torch.randn((5000, 2)).to(device)
    # b = torch.randn((5000, 2)).to(device)
    #
    # begin = time.time()
    #
    # f1 = a[:, 0].unsqueeze(1).T-b[:, 0].unsqueeze(1)
    # f2 = a[:, 1].unsqueeze(1).T-b[:, 1].unsqueeze(1)
    # f = f1**2+f2**2
    # W = torch.mm(a,a.T)
    # c = torch.sum(torch.exp(-1/2/5/5*(a.T-b)**2)*(a.T*b)+1/5/5-(1/5/5+1/5/5/5/5)*(a.T-b)**2)
    # end = time.time()
    # print("cost time 1:", end-begin)
    y = torch.randn((3,1))
    Z1 = (y-y.T)**2
    Z2 = (y.T - y)**2
    x = torch.randn((10,1))
    h=0.1
    z = (-1-2/h/h)*x+2/h/h*y.T
    W = torch.randn((6,1))
    w2 = torch.randn((6,2))# 10*5
    A = w2*W
    a=0