from cProfile import label
import seaborn 
import numpy as np
import matplotlib.pyplot as plt
import math

X_data_10,w_data_10=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/forward_10.npy")
X_data_5,w_data_5=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/forward_5.npy")
X_data_2,w_data_2=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/forward_2.npy")
X_data_10_inv,w_data_10_inv=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/inverse_10.npy")
X_data_5_inv,w_data_5_inv=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/inverse_5.npy")
X_data_2_inv,w_data_2_inv=np.load("C:/Users/shark/桌面/WFR-main/1d_plot/figs/inverse_2.npy")



plt.figure(figsize=(12,8))
plt.subplot(2, 3, 1)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_10[0:1000:,i],w_data_10[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()

plt.subplot(2, 3, 2)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_5[0:1000:,i],w_data_5[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()

plt.subplot(2, 3, 3)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_2[0:1000:,i],w_data_2[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()

plt.subplot(2, 3, 4)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_10_inv[0:1000:,i],w_data_10_inv[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()

plt.subplot(2, 3, 5)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_5_inv[0:1000:,i],w_data_5_inv[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()

plt.subplot(2, 3, 6)
for i in [0,2,4,6,8]:
    plt.scatter(X_data_2_inv[0:1000:,i],w_data_2_inv[0:1000,i],label="T=%.2f"%(i/8),s=3)
plt.xlim(-6,6)
plt.ylim(0,3.5)
plt.legend(loc="upper right",fontsize="10",markerscale=2.5)
plt.grid()
# plt.show()
plt.savefig('C:/Users/shark/桌面/weight_small_update2.pdf',dpi=600)

