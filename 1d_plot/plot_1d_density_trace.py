import numpy as np 
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st

def Gaussian(x,mu,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/2*sigma**2)

def mix_Gaussian(x):
    return 1/3*Gaussian(x,-3,1)+2/3*Gaussian(x,3,1)
X_space=np.linspace(-6,6,100)
# X_plot=np.random.randn(10000)
# w_1=np.ones(10000)
X_plot=np.load("MMD_data_output/output/z_final_1d.npy")
w_1=np.load("MMD_data_output/output/w_final_1d.npy")

p=np.argsort(X_plot,axis=0)
X_new=np.sort(X_plot,axis=0)
w_new=w_1[p].squeeze()

n0,b,pat=plt.hist(X_new,bins=30,color="white",density=True,weights=w_new)
mid=[]
height=[]
for i in range(b.shape[0]-1):
    mid.append((b[i]+b[i+1])/2)
    height.append(pat[i].get_height())
mid=np.array(mid)
height=np.array(height)

# plt.plot(mid,height, linewidth=5,color="darkblue")
# plt.hist(X_plot,weights=w_1,density=True,bins=50)
# plt.plot(X_space,Gaussian(X_space,0,1),linewidth=5,linestyle='--',color="red")
# plt.plot(X_space,mix_Gaussian(X_space),linewidth=5,linestyle='--',color="green")
# plt.xlim(-5,5)
# plt.show()

X_plot_full=np.load("MMD_data_output/output/z_full_1d.npy")[0:100].squeeze()
X_plot_inverse_full=np.load("MMD_data_output/output/z_inverse_full_1d.npy")[100:200].squeeze()
t_space=np.linspace(0,1,9)
t_space_inverse=np.linspace(1,0,9)
plt.figure(figsize=(10,8))
for i in range(30):
    plt.plot(X_plot_full[i,:],t_space,linewidth=2,marker="o",color="darkblue")
plt.grid()
plt.xlim(-5,5)
plt.ylim(0,1)
plt.xlabel("Position",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.ylabel("t",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.title("Forward",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.savefig("1d_plot/trace_forward.pdf",dpi=100)
plt.close()

plt.figure(figsize=(10,8))
for i in range(30):
    plt.plot(X_plot_inverse_full[i,:],t_space,linewidth=2,marker="o",color="darkblue")
plt.grid()
plt.xlim(-5,5)
plt.ylim(0,1)
plt.xlabel("Position",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.ylabel("t",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.title("Backward",fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.savefig("1d_plot/trace_inverse.pdf",dpi=100)
plt.close()