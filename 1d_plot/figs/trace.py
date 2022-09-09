from turtle import color
import seaborn 
import numpy as np
import matplotlib.pyplot as plt
import math

Zfull=np.load("/home/liuchang/OT-Flow/1d_plot/figs/zfull.npy")
Zfull_inverse=np.load("/home/liuchang/OT-Flow/1d_plot/figs/zfull_inverse.npy")
print(Zfull.shape)
print(Zfull_inverse.shape)


fig = plt.figure(figsize=(20,20))

for i in range(50,80):
  plt.plot(Zfull[i,0,:],np.linspace(0,1,9),"-o",color="green")

plt.grid(linestyle="--")
plt.xlabel("$Position$",fontsize=40)
plt.ylabel("t",fontsize=40)
plt.ylim(0,1)
plt.title("Forward",fontsize=40)
plt.xticks(size=25)
plt.yticks(size=25)
plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/posterior.png", dpi=300)
plt.close()


fig = plt.figure(figsize=(20,20))
ax = plt.gca()

for i in range(50,80):
  plt.plot(Zfull_inverse[i,0,:],np.linspace(0,1,9),"-o",color="green")

plt.grid(linestyle="--")
plt.xlabel("$Position$",fontsize=40)
plt.ylabel("t",fontsize=40)
plt.ylim(0,1)
plt.title("Inverse",fontsize=40)
plt.xticks(size=25)
plt.yticks(size=25)
plt.savefig("/home/liuchang/OT-Flow/1d_plot/figs/inverse.png", dpi=300)
plt.close()