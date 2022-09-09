import seaborn 
import numpy as np
import matplotlib.pyplot as plt

X,w=np.load("bernoulli.npy")
X_1,w_1=np.load("bernoulli_1.npy")
p=np.argsort(X_1,axis=0)
X_new=np.sort(X_1,axis=0)

w_new=w_1[p].squeeze()
print(X_new.shape)
print(w_new.shape)
from scipy.interpolate import spline
# weight=(1e3*w_1).astype(np.int)
# record=[]
# for i in range(X_1.shape[0]):
#   for k in range(int(weight[i])):
#     record.append(X_1[i])
# record=np.array(record)
# print(record.shape)

plt.scatter(X,w,s=1,label="given data")
# plt.scatter(X_1,w_1,s=1,label="generated samples")
plt.plot(X_new,w_new,label="fitted posterior with generated samples",color="orange")
# plt.hist(record,bins=100,density=True)
plt.xlim(-0.5,2)
plt.xlabel("estimated initial position x")
plt.ylabel("weight")
plt.grid(linestyle = '--')
plt.legend()
plt.show()

print(w.mean())
