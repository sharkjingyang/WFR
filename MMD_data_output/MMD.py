import numpy as np
import torch
import matplotlib.pyplot as plt

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
    W: Q's weight shape: M*1
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

print("--------------2d------------------")
##WFR MMD  2d 
print("WFR forward")
X_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_final.npy")
w_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_final.npy")
Gaussian_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/Gaussian_samples.npy")
print(MMD_Weighted(torch.tensor(Gaussian_p),torch.tensor(X_p),torch.tensor(w_p).reshape(-1,1)/Gaussian_p.shape[0]))

print("WFR inverse")
X_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_inverse.npy")
w_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_inverse.npy")
data_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/data_samples.npy")
print(MMD_Weighted(torch.tensor(data_p),torch.tensor(X_p_inverse),torch.tensor(w_p_inverse).reshape(-1,1)/Gaussian_p.shape[0]))

###OT MMD  2d
print("OT forward")
X_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/z_final.npy")
Gaussian_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/Gaussian_samples.npy")
print(MMD(torch.tensor(Gaussian_p),torch.tensor(X_p)))

print("OT inverse")
X_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/z_inverse.npy")
data_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/data_samples.npy")
print(MMD(torch.tensor(data_p),torch.tensor(X_p_inverse)))

print("--------------1d------------------")
##WFR MMD  1d
print("WFR forward")
X_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_final_1d.npy")
w_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_final_1d.npy")
Gaussian_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/Gaussian_samples_1d.npy")
print(MMD_Weighted(torch.tensor(Gaussian_p),torch.tensor(X_p),torch.tensor(w_p).reshape(-1,1)/Gaussian_p.shape[0]))

print("WFR inverse")
X_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_inverse_1d.npy")
w_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_inverse_1d.npy")
data_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/data_samples_1d.npy")
print(MMD_Weighted(torch.tensor(data_p).reshape(-1,1),torch.tensor(X_p_inverse),torch.tensor(w_p_inverse).reshape(-1,1)/Gaussian_p.shape[0]))


print("OT forward")
X_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/z_final_1d.npy")
Gaussian_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/Gaussian_samples_1d.npy")
print(MMD(torch.tensor(Gaussian_p),torch.tensor(X_p)))

print("OT inverse")
X_p_inverse=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/z_inverse_1d.npy")
data_p=np.load("C:/Users/shark/桌面/WFR-main/MMD_data_output/OT_output/data_samples_1d.npy")
print(MMD(torch.tensor(data_p),torch.tensor(X_p_inverse)))
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.hist2d(X_p[:,0],X_p[:,1],range=[[-4,4],[-4,4]],weights=w_p,bins=100)
# plt.title("f(x)")

# plt.subplot(1, 3, 2)
# plt.hist2d(Gaussian_p[:,0],Gaussian_p[:,1],range=[[-4,4],[-4,4]],bins=100)
# plt.title("Gaussian")

# plt.subplot(1, 3, 3)
# plt.hist2d(X_p_inverse[:,0],X_p_inverse[:,1],range=[[-4,4],[-4,4]],bins=100)
# plt.title("generate")

# plt.show()

