from cProfile import label
import seaborn 
import numpy as np
import matplotlib.pyplot as plt
import math

def Gaussian(x,mu,sigma):
  return 1/np.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)

X_data_1,w_data_1=np.load("posterior_data/T1.npy")
X_data_2,w_data_2=np.load("posterior_data/T2.npy")
X_data_3,w_data_3=np.load("posterior_data/T3.npy")
X_data_4,w_data_4=np.load("posterior_data/T4.npy")
X_data_5,w_data_5=np.load("posterior_data/T5.npy")
w_data_1=w_data_1*Gaussian(X_data_1,0.5,1)
w_data_2=w_data_2*Gaussian(X_data_2,0.5,1)
w_data_3=w_data_3*Gaussian(X_data_3,0.5,1)
w_data_4=w_data_4*Gaussian(X_data_4,0.5,1)
w_data_5=w_data_5*Gaussian(X_data_5,0.5,1)

p_1=np.argsort(X_data_1,axis=0)
X_data_1=np.sort(X_data_1,axis=0)
w_data_1=w_data_1[p_1].squeeze()

p_2=np.argsort(X_data_2,axis=0)
X_data_2=np.sort(X_data_2,axis=0)
w_data_2=w_data_2[p_2].squeeze()

p_3=np.argsort(X_data_3,axis=0)
X_data_3=np.sort(X_data_3,axis=0)
w_data_3=w_data_3[p_3].squeeze()

p_4=np.argsort(X_data_4,axis=0)
X_data_4=np.sort(X_data_4,axis=0)
w_data_4=w_data_4[p_4].squeeze()

p_5=np.argsort(X_data_5,axis=0)
X_data_5=np.sort(X_data_5,axis=0)
w_data_5=w_data_5[p_5].squeeze()

X1_inv,w1_inv=np.load("posterior_data/T1_inverse.npy")
X2_inv,w2_inv=np.load("posterior_data/T2_inverse.npy")
X3_inv,w3_inv=np.load("posterior_data/T3_inverse.npy")
X4_inv,w4_inv=np.load("posterior_data/T4_inverse.npy")
X5_inv,w5_inv=np.load("posterior_data/T5_inverse.npy")

my_x_ticks = np.arange(-1, 2.1,1)
my_y_ticks = np.arange(0, 7, 2)

fig = plt.figure(figsize=(30,7.5))

plt.subplot(1,4,1)
plt.hist(X1_inv,bins=100,weights=w1_inv,density=True,label="generated samples")
plt.plot(X_data_1,w_data_1,"red",label="true posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=30)
plt.ylabel("Density",fontsize=25)
plt.xlim(-1,2)
plt.ylim(0,7)
plt.title("T=0.5",fontsize=30)
plt.xticks(my_x_ticks,size=18)
plt.yticks(my_y_ticks,size=18)
plt.legend(fontsize=15)


plt.subplot(1,4,2)
plt.hist(X2_inv,bins=100,weights=w2_inv,density=True,label="generated samples")
plt.plot(X_data_2,w_data_2,"red",label="true posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=30)
plt.ylabel("Density",fontsize=25)
plt.xlim(-1,2)
plt.ylim(0,7)
plt.title("T=1",fontsize=30)
plt.xticks(my_x_ticks,size=18)
plt.yticks(my_y_ticks,size=18)
plt.legend(fontsize=15)

plt.subplot(1,4,3)
plt.hist(X3_inv,bins=100,weights=w3_inv,density=True,label="generated samples")
plt.plot(X_data_3,w_data_3,"red",label="true posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=30)
plt.ylabel("Density",fontsize=25)
plt.xlim(-1,2)
plt.title("T=2",fontsize=30)
plt.ylim(0,7)
plt.xticks(my_x_ticks,size=18)
plt.yticks(my_y_ticks,size=18)
plt.legend(fontsize=15)

# plt.subplot(1,5,4)
# plt.hist(X4_inv,bins=100,weights=w4_inv,density=True,label="generated samples")
# plt.plot(X_data_4,w_data_4,"red",label="real posterior")
# plt.grid(linestyle="--")
# plt.xlabel("$x_0$")
# plt.ylabel("Density")
# plt.xlim(-1,2)
# plt.title("T=4")
# plt.ylim(0,7)
# plt.legend()

plt.subplot(1,4,4)
plt.hist(X5_inv,bins=100,weights=w5_inv,density=True,label="generated samples")
plt.plot(X_data_5,w_data_5,"red",label="true posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=30)
plt.ylabel("Density",fontsize=25)
plt.xlim(-1,2)
plt.title("T=5",fontsize=30)
plt.ylim(0,7)
plt.xticks(my_x_ticks,size=18)
plt.yticks(my_y_ticks,size=18)
plt.legend(fontsize=15)

plt.savefig("posterior_data/online_algorithm.pdf", dpi=300)
plt.close()
