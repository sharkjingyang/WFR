import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
import math
import matplotlib.pyplot as plt

def calculate_weight_for_theta(theta):
    data = scipy.io.loadmat('Stein-Variational-Gradient-Descent-master/data/covertype.mat')
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1
    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d + 1
    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
    print("X_train shape")
    print(X_train.shape)
    weight=np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        if i %100==0:
            print(i)
        w=theta[i, :-1]
        n_test=len(y_test)
        prob = np.zeros([n_test, 1])
        coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(w, n_test, 1), X_test), axis=1))
        prob[:, 0] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        llh = np.exp(np.mean(np.log(prob)) )
        weight[i]=llh  ##last row is likelihood   i.e. weight
    weight=weight/weight.sum()*weight.shape[0]
    return weight


LOWX=-1
HIGHX=1
LOWY=-1
HIGHY=1
nBins=70
d1=22
d2=27

theta=np.load("data/data_WFR/noweight_theta.npy")
mu = (theta).mean(axis=0)
s = theta.std(axis=0)
thet_normalized=(theta-mu)/s
print(theta.shape)
Gaussian_samples=theta+np.random.randn(5000,theta.shape[1])*5
Gaussian_weight=calculate_weight_for_theta(Gaussian_samples)
normalized_Gaussian=(Gaussian_samples-mu)/s
fig, axs = plt.subplots( 1,3)
fig.set_size_inches(12, 6)
im1 , _, _, map1 = axs[0].hist2d(thet_normalized[:,d1], thet_normalized[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
axs[0].set_title('equal weight samples (SVGD)')
im2 , _, _, map2 = axs[1].hist2d(normalized_Gaussian[:,d1],normalized_Gaussian[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins,weights=Gaussian_weight)
axs[1].set_title('osc_samples with weight')
im3 , _, _, map3 = axs[2].hist2d(normalized_Gaussian[:,d1],normalized_Gaussian[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
axs[2].set_title('osc_samples without weight')
plt.show()

# #calculate weight for SVGD sample:
# weight=calculate_weight_for_theta(theta)
# np.save("data/theta_SVGD/weight.npy",weight)
# weight=np.load("data/theta_SVGD/weight.npy")
# print(weight.max())


theta_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:,:-1]
weight_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:,-1]
weight_osc_normalized=weight_osc/weight_osc.sum()*weight_osc.shape[0]
print(weight_osc_normalized.max())
theta_osc=(theta_osc-mu)/s


theta_WFR_gen=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/WFR_gen_theta_weight.npy")[:,:-1]
weight_WFR_gen=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/WFR_gen_theta_weight.npy")[:,-1]



fig, axs = plt.subplots( 1,4)
fig.set_size_inches(24, 6)
# hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
im1 , _, _, map1 = axs[0].hist2d(thet_normalized[:,d1], thet_normalized[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
axs[0].set_title('equal weight samples (SVGD)')
im2 , _, _, map2 = axs[1].hist2d(theta_osc[:,d1], theta_osc[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
axs[1].set_title('osc_samples without weight')
im3 , _, _, map3 = axs[2].hist2d(theta_osc[:,d1], theta_osc[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins, weights=weight_osc_normalized)
axs[2].set_title('osc_samples with weight')
im4 , _, _, map4 = axs[3].hist2d(theta_WFR_gen[:,d1], theta_WFR_gen[:,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins,weights=weight_WFR_gen)
axs[3].set_title('gen WFR samples')
plt.show()



