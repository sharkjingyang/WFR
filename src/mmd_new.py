import torch
import numpy as np

def MMD(X, Q):
    """
    Gaussian kernel k(x,y)=exp(-1/2||x-y||^2)
    input:
    X: samples  shape: M*d
    Q: samples  shape: N*d
    """
    d = X.shape[1]
    f = 0
    g = 0
    h = 0
    for i in range(d):
        # |x_i-x_j|^2
        f_temp = (X[:, i].unsqueeze(1).T - X[:, i].unsqueeze(1))**2
        g_temp = (Q[:, i].unsqueeze(1).T - Q[:, i].unsqueeze(1))**2
        h_temp = (X[:, i].unsqueeze(1).T - Q[:, i].unsqueeze(1))**2
        f = f + f_temp
        g = g + g_temp
        h = h + h_temp
    mmd = torch.mean(torch.exp(-0.5*f))+torch.mean(torch.exp(-0.5*g))-2*torch.mean(torch.exp(-0.5*h))
    return mmd

def MMD_Weighted(X, Q, W):
    """
    Gaussian kernel k(x,y)=exp(-1/2||x-y||^2)
    input:
    X: samples  shape: N*d
    Q: samples  shape: M*d
    W: Q's weight   M*1
    """
    d = X.shape[1]
    x = 0
    q = 0
    xq = 0
    for i in range(d):
        # |x_i-x_j|^2
        x_temp = (X[:, i].unsqueeze(1).T - X[:, i].unsqueeze(1))**2
        q_temp = (Q[:, i].unsqueeze(1).T - Q[:, i].unsqueeze(1))**2
        xq_temp = (X[:, i].unsqueeze(1).T - Q[:, i].unsqueeze(1))**2
        x = x + x_temp
        q = q + q_temp
        xq = xq + xq_temp
    weight1 = torch.mm(W, W.T)
    weight2 = torch.diag_embed(W.squeeze())
    # wei
    mmd = torch.mean(torch.exp(-0.5*x))+torch.sum(weight1*torch.exp(-0.5*q))-2*torch.sum(torch.mm(weight2, torch.exp(-0.5*xq)))/X.shape[0]
    return mmd

if __name__ == "__main__":
    a = torch.randn((2,3))
    b = torch.randn((3,1))
    c = torch.mm(a,b)
    print(a)
    print(b)
    print(c)