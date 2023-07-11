import numpy as np 
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st
# x= np.array([1e-1,9e-2,8e-2,7e-2,6e-2,5e-2,4e-2,3e-2,2e-2,1e-2 ,   9e-3,8e-3,7e-3,6e-3,5e-3,4.5e-3,4e-3,3e-3,2.3e-3,2e-3,1e-3,5e-4])
# y=  np.array([2.3375,2.3625,2.381,2.4086,2.427,2.4501,2.4682,2.4932,2.5302,2.567   ,2.5572,2.5553,2.5616,2.5682,2.5724,  2.5823,2.5764,2.5760,2.5692,2.5751,2.6012,2.6052])
x= np.array([1e-1,9e-2,8e-2,7e-2,5e-2,3e-2,1e-2 ,8e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3,5e-4])
y=  np.array([ 2.21307,2.23069 ,2.24180 ,2.25356,2.31278,2.34931,2.38141,2.37828,2.38479,2.39744,2.39522,2.41258,2.42824,2.44539,2.44825])


plt.plot(x,y)
plt.xscale("log")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\mathcal{J}_{SWFR}$")
# plt.legend(loc=4,prop = {'size':15})
# plt.show()
plt.savefig("1d_plot/J_1_gamma_.pdf")