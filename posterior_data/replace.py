import numpy as np
import math
import matplotlib.pyplot as plt

X,w=np.load("/home/liuchang/OT-Flow/posterior_data/T4_inverse.npy")
observe_data=np.load("/home/liuchang/OT-Flow/posterior_data/observe.npy")

dt=0.1
sigma=0.4
T=5

def Gaussian(x,mu,sigma):
  return 1/np.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)

def analytical_sol(x,t):
  return x/np.sqrt(x**2+(1-x**2)*np.exp(-2*t))


def cal_pro(x_prior,ob_data,dt,Tspan):
  dt_space=np.arange(start=Tspan[0]+dt,stop=Tspan[1]+dt/2,step=dt)
  true_sol=analytical_sol(x=x_prior,t=dt_space)
  p=Gaussian(ob_data[int(Tspan[0]/dt)+1:int(Tspan[1]/dt)+1],true_sol,sigma)
  return np.exp(np.sum(np.log(p)))

w_pen=np.zeros(X.shape)
for i in range(X.shape[0]):
  w_pen[i]=cal_pro(x_prior=X[i],ob_data=observe_data,dt=0.1,Tspan=[4,5])
w_new=(w*w_pen)/np.sum(w*w_pen)*X.shape[0]

np.save("/home/liuchang/OT-Flow/posterior_data/T5_replace.npy",(X,w_new))




