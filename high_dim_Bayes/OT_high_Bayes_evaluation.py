import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg') # for linux server with no tkinter
# matplotlib.use('Agg') # assume no tkinter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import sys
sys.path.append('C:/Users/shark/桌面/WFR-main')
from src.OTFlowProblem import *
import numpy as np
import os
import h5py
import datasets
from torch.nn.functional import pad
from matplotlib import colors # for evaluateLarge
from torchvision import models
import argparse
import os
import time
import datetime
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters
import torch
import numpy.matlib as nm
import scipy.io
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    elif name =="swin_dim40":
        return datasets.SWIN_DIM40()
    
    elif name =="SVGD_data":
        return datasets.SVGD_DATA()

    else:
        raise ValueError('Unknown dataset')

def evaluation( theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
    
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]

def evaluation_weighted( theta, X_test, y_test):
        theta_w = theta[:, :-2]
        weight = theta[:, -1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
    
        for t in range(M):
        
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta_w[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob_weightedsum= np.mean(prob*weight, axis=1)
        acc = np.mean(prob_weightedsum > 0.5)
        # llh = np.mean(np.log(prob)*weight)
        llh = np.mean(np.log(prob_weightedsum))
        return [acc, llh]

def calculate_weight_for_theta(theta,X_test,y_test):

    weight=np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        w=theta[i, :-1]
        n_test=len(y_test)
        prob = np.zeros([n_test, 1])
        coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(w, n_test, 1), X_test), axis=1))
        prob[:, 0] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        llh = np.exp(np.mean(np.log(prob)) )
        weight[i]=llh  ##last row is likelihood   i.e. weight
    weight_normalized=weight/weight.sum()*weight.shape[0]
    print("max weight  is %f" % weight_normalized.max())
    print("min weight  is %f" % weight_normalized.min())
    return weight


data = load_data('SVGD_data')
data.trn.x = torch.from_numpy(data.trn.x)
# print(data.trn.x.shape)
data.val.x = torch.from_numpy(data.val.x)
# print(data.val.x.shape)


test_batch_size=100
p_samples = cvt(data.val.x[0:test_batch_size,:]) ##SVGD samples 
nSamples = p_samples.shape[0]
y = cvt(torch.randn(nSamples,data.trn.x.shape[1])) # sampling from rho_1 / standard normal



net = Phi(nTh=2, m=256, d=data.trn.x.shape[1], alph=[1.0,100.0,15.0]) 
a=torch.load("experiments/cnf/large/SVGD_noisedata_well_trained.pth")


net.load_state_dict(a['state_dict'])
net.cuda()
print("---load finish----------")
net.eval()


def plot4(net, x, y, nt_val, sPath, sTitle="", doPaths=False):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]


    fx = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 70
    LOWX  = -4
    HIGHX = 4
    LOWY  = -4
    HIGHY = 4

    if d > 50: # assuming bsds
        # plot dimensions d1 vs d2 
        d1=0
        d2=1
        LOWX  = -0.15   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.15
        LOWY  = -0.15
        HIGHY = 0.15
    if d > 700: # assuming MNIST
        d1=0
        d2=1
        LOWX  = -10   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 10
        LOWY  = -10
        HIGHY = 10
    elif d==8: # assuming gas
        LOWX  = -0.4   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.4
        LOWY  = -0.4
        HIGHY = 0.4 
        d1=2
        d2=3
        nBins = 100
    elif d==56: 
        LOWX  = -3   
        HIGHX = 3
        LOWY  = -3
        HIGHY = 3
        d1=17
        d2=18
        nBins = 70

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    # hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
    im1 , _, _, map1 = axs[0, 0].hist2d(x.detach().cpu().numpy()[:, d1], x.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[0, 0].set_title('x from rho_0')
    im2 , _, _, map2 = axs[0, 1].hist2d(fx.detach().cpu().numpy()[:, d1], fx.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[0, 1].set_title('f(x)')
    im3 , _, _, map3 = axs[1, 0].hist2d(finvfx.detach().cpu().numpy()[: ,d1] ,finvfx.detach().cpu().numpy()[: ,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 0].set_title('finv( f(x) )')
    im4 , _, _, map4 = axs[1, 1].hist2d(genModel.detach().cpu().numpy()[:, d1], genModel.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 1].set_title('finv( y from rho1 )')

    fig.colorbar(map1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(map2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(map3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(map4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )


    # plot paths
    if doPaths:
        forwPath = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        backPath = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)

        # plot the forward and inverse trajectories of several points; white is forward, red is inverse
        nPts = 10
        pts = np.unique(np.random.randint(nSamples, size=nPts))
        for pt in pts:
            axs[0, 0].plot(forwPath[pt, 0, :].detach().cpu().numpy(), forwPath[pt, 1, :].detach().cpu().numpy(), color='white', linewidth=4)
            axs[0, 0].plot(backPath[pt, 0, :].detach().cpu().numpy(), backPath[pt, 1, :].detach().cpu().numpy(), color='red', linewidth=2)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # axs[i, j].get_yaxis().set_visible(False)
            # axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    return genModel[:,0:data.trn.x.shape[1]].detach().cpu().numpy()

sPath = os.path.join('C:/Users/shark/桌面/WFR-main/high_dim_Bayes/fig_train_after.png')

with torch.no_grad():
    theta_generate=plot4(net=net, x=p_samples, y=y, nt_val=10, sPath=sPath, sTitle="jijinini", doPaths=False)





data = scipy.io.loadmat('C:/Users/shark/桌面/OT-Flow-master/data/SVGD_data/covertype.mat')
X_input = data['covtype'][:, 1:] # N*54
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N, 1])]) #N*55
d = X_input.shape[1]
D = d + 1



##we need unormalize the generate data to use them for inference
# theta_data= np.load('C:/Users/shark/桌面/WFR-main/data/data_WFR/noweight_theta.npy') ##theta used to train
# mu = theta_data.mean(axis=0)
# s = theta_data.std(axis=0)
# theta_generate=theta_generate*s+mu


#-----------------------------------------------------------------------------------------------------------------
## use SVGD+noise as data to train
theta_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:100,:-1]
# weight_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:,-1]
# weight_osc_normalized=weight_osc/weight_osc.sum()*weight_osc.shape[0]
# print(weight_osc_normalized.max())

mu = (theta_osc).mean(axis=0)
s = theta_osc.std(axis=0)
theta_generate=theta_generate*s+mu

weight_osc=calculate_weight_for_theta(theta_osc,X_input,y_input)
weight_osc_normalized=weight_osc/weight_osc.sum()*weight_osc.shape[0]
theta_weight=np.hstack((theta_osc,weight_osc_normalized.reshape(-1,1)))

time_start=time.time()
weight_generate=calculate_weight_for_theta(theta_generate,X_input,y_input)
weight_generate_normalized=weight_generate/weight_generate.sum()*weight_generate.shape[0]
theta_gen_weight=np.hstack((theta_generate,weight_generate_normalized.reshape(-1,1)))

# print ('[accuracy, log-likelihood] of train samples, then calculate likelihood as weight' )
# print (evaluation_weighted(theta_weight, X_input, y_input)) #0.7400397926376736 for SVGDnoise small   #0.6724324235342 for SVGDnoise big
print ('[accuracy, log-likelihood] of new generated OT samples, then calculate likelihood as weight' )
print (evaluation_weighted(theta_gen_weight, X_input, y_input)) #0.7100042684144218 for SVGDnoise small   #0.662345745215342 for SVGDnoise big

time_end=time.time()
print('time cost',time_end-time_start,'s')



theta_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:100,:-1]
mu = (theta_osc).mean(axis=0)
s = theta_osc.std(axis=0)
theta_WFR_gen=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/WFR_gen_theta_weight.npy")[:100,:-1]
theta_WFR_gen=theta_WFR_gen*s+mu
weight_WFR_gen=np.load("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/WFR_gen_theta_weight.npy")[:100,-1]

time_start=time.time()
weight_WFR_gen_normalized=weight_WFR_gen/weight_WFR_gen.sum()*weight_WFR_gen.shape[0]
theta_WFR_gen_weight=np.hstack((theta_WFR_gen,weight_WFR_gen_normalized.reshape(-1,1)))
print ('[accuracy, log-likelihood] of new generated WFR theta samples ' )
print (evaluation_weighted(theta_WFR_gen_weight, X_input, y_input))  #0.7380845834509442 for SVGDnoise small  #0.669424190922 for SVGDnoise big
time_end=time.time()
print('time cost',time_end-time_start,'s')
#-------------------------------------------------------------------------------------------------------------------











# [accuracy, log-likelihood] of theta data samples
# [0.7561547782145636, -0.5169536088871499]
# [accuracy, log-likelihood] of new generated theta samples
# [0.7456661824540629, -0.5690497484497747]
# Calculating likelihood for new generated samples
# max weight  is 1.817906
# min weight  is 0.023908
# [accuracy, log-likelihood] of new generated theta samples with likelihood weighted
# [0.7548518791350265, -0.5502347363995422]
# [accuracy, log-likelihood] of new generated WFR theta samples 
# [0.7292121333122208, -0.5921642036455818]

### train set are SVGD samples +noise (calculate likelihood as weight for WFR)
## for CNF: we train a CNF to generate above samples, then calculate likelihood as weight, to predict
## for WFR: we train a UOT-gen to generate  samples with weight, then directly predict