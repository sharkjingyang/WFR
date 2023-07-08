import seaborn 
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.datasets
import sys
import torch

batch_size=20000
n_Bins=50


X_normal=np.random.normal(size=(batch_size,2))

data="test_high_dim"
if data == "swissroll":
    data_set = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
    data_set = data_set.astype("float32")[:, [0, 2]]
    data_set /= 5

if data == "checkerboard":
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    x = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    data_set=x

if data == "8gaussians":
        rng = np.random.RandomState()
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        epsilon = 0.0
        for i in range(int(batch_size*(1-epsilon))):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        for i in range(batch_size-int(batch_size*(1-epsilon))):
            point = rng.randn(2)*1.414*2
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        data_set=dataset

if data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data_set=data

if data == "test_high_dim":
    dataset = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
    dataset = dataset.astype("float32")[:, [0, 2]]
    dataset /= 5
    data_set=np.random.normal(size=(batch_size,8))
    data_set[:,0]=dataset[:,0]
    data_set[:,1]=dataset[:,1]


z_data=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/z_final.npy")
w_data=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/w_final.npy")
z_data_inv=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/z_inverse.npy")
w_data_inv=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/w_inverse.npy")

plt.figure()

plt.subplot(1, 4, 1)
plt.hist2d(data_set[:,0],data_set[:,1],bins=n_Bins,range=[[-4, 4], [-4, 4]])
plt.title("Data")

plt.subplot(1, 4, 2)
plt.hist2d(X_normal[:,0],X_normal[:,1],bins=n_Bins,range=[[-4, 4], [-4, 4]])
plt.title("Normal")


plt.subplot(1, 4, 3)
plt.hist2d(z_data[:,0],z_data[:,1],bins=n_Bins,weights=w_data,range=[[-4, 4], [-4, 4]])
plt.title("z=f(x)")

plt.subplot(1, 4, 4)
plt.hist2d(z_data_inv[:,0],z_data_inv[:,1],bins=n_Bins,weights=w_data_inv,range=[[-4, 4], [-4, 4]])
plt.title("x=f^-1(z)")
plt.show()
