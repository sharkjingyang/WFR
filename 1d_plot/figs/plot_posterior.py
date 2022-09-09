from cProfile import label
import seaborn 
import numpy as np
import matplotlib.pyplot as plt
import math
# 查看data
def Gaussian(x,mu,sigma):
  return 1/np.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)

X_data_1,w_data_1=np.load("/home/liuchang/OT-Flow/posterior_data/T1.npy")
X_data_2,w_data_2=np.load("/home/liuchang/OT-Flow/posterior_data/T2.npy")
X_data_3,w_data_3=np.load("/home/liuchang/OT-Flow/posterior_data/T3.npy")
X_data_4,w_data_4=np.load("/home/liuchang/OT-Flow/posterior_data/T4.npy")
X_data_5,w_data_5=np.load("/home/liuchang/OT-Flow/posterior_data/T5.npy")
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


n0,b,pat=plt.hist(X_data_1,bins=100,color="white",density=True,weights=w_data_1)
mid=[]
height=[]
for i in range(b.shape[0]-1):
  mid.append((b[i]+b[i+1])/2)
  height.append(pat[i].get_height())
mid=np.array(mid)
height=np.array(height)



# plt.hist(X_data,weights=w_data,density=True,bins=50)
# plt.plot(mid,height)
plt.plot(X_data_1,w_data_1,label="true poterior T=0.5")
plt.plot(X_data_2,w_data_2,label="true poterior T=1")
plt.plot(X_data_3,w_data_3,label="true poterior T=2")
plt.plot(X_data_4,w_data_4,label="true poterior T=4")
plt.plot(X_data_5,w_data_5,label="true poterior T=5")
# plt.ylim(0,5)
plt.legend()
plt.grid()
plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/posterior.png", dpi=300)
plt.close()


# 查看正向
X,w=np.load("OT-Flow/1d_plot/figs/forward.npy")
plt.hist(X,bins=100,weights=w,density=True)
# plt.scatter(X,w,s=1)
plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/forward.png", dpi=300)
plt.close()


zfull=np.load("OT-Flow/1d_plot/figs/zfull.npy") ##zfull should be [2048,dimension,9]
# print(zfull.shape)
shuffle=np.linspace(20,9700,num=50,dtype=int)
z_traject=zfull[shuffle,0,:]
z_weight=zfull[shuffle,-1,:]
# print(z_traject.shape)

# for i in range(50):
#   plt.plot(z_traject[i,:],np.linspace(0,1,9))
# plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/trajectory.png", dpi=300)
# plt.close()

# for i in range(9):
#   plt.scatter(z_traject[:,i],z_weight[:,i],s=1,label="t={}".format (i))
# plt.legend()
# plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/weight.png", dpi=300)
# plt.close()




X,w=np.load("OT-Flow/1d_plot/figs/inverse.npy")
# plt.scatter(X,w,s=1)
plt.hist(X,bins=100,weights=w,density=True,label="generated samples")
plt.plot(X_data_5,w_data_5,"red",label="real posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$")
plt.ylabel("Density")
plt.ylim(0,7)
plt.legend()
plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/inverse.png", dpi=300)
plt.close()




# X,w=np.load("OT-Flow/1d_plot/figs/save.npy")

# p=np.argsort(X,axis=0)
# X_new=np.sort(X,axis=0)

# w_new=w[p].squeeze()
# print(X_new.shape)
# print(w_new.shape)



# # plt.scatter(X,w,s=1,label="generated samples")
# plt.hist(X, density=True, weights=w, bins=100)
# plt.plot(mid,height,"--")
# plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/savetest.png", dpi=300)
# plt.close()


