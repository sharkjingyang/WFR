import numpy as np
import matplotlib.pyplot as plt
import math

def Gaussian(x,mu,sigma):
  return 1/np.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)


X_prior,w=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy")
p=np.argsort(X_prior,axis=0)
X_prior=np.sort(X_prior,axis=0)
w=w[p].squeeze().reshape(-1,1)
w=w*Gaussian(X_prior,0.5,1)
# w=w/np.sum(w)*X_prior.shape[0]
X_inv,w_inv=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/inverse.npy")
print(X_inv.shape)
plt.plot(X_prior,w,"red",label="true posterior")
plt.hist(X_inv,bins=250,weights=w_inv,density=True,label="generated samples")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=15)
plt.ylabel("Density",fontsize=15)
plt.xlim(-1,3)
plt.ylim(0,3)
# plt.title("T=0.5",fontsize=30)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(fontsize=12)
plt.savefig("/home/liuchang/OT-Flow/single_plot_Baysian/inverse.png", dpi=300)
plt.close()
