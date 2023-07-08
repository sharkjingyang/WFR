import numpy as np 
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st
matplotlib.rcParams.update({'font.size': 12})

alpha_space=np.array([1,2,3,4,5,6,7,8,9,10])
# pd_WFR=np.array([ 1.24899,1.70790,1.93175,2.06203,2.14693,2.20659,2.25081,2.28492,2.31203,2.33411])
pd_WFR=np.array([ 1.28320,1.76947,2.01062,2.14881,2.24125,2.30542,2.35210,2.39070,2.41570,2.43553])
NN_WFR=np.array([[1.42168,1.85996,2.05889,2.12049,2.24734,2.26395,2.31326,2.37562,2.37874,2.39380],
                 [1.46796,1.84327,2.02567,2.14841,2.18723,2.30069,2.34534,2.34793,2.35661,2.38741],
                 [1.41121,1.83905,2.00171,2.14761,2.23024,2.27415,2.34825,2.39973,2.41759,2.45231],
                 [1.44298,1.82440,2.08899,2.15556,2.24687,2.29267,2.30580,2.33897,2.39039,2.41360],
                 [1.45767,1.86357,2.04551,2.18985,2.22489,2.34874,2.29064,2.37123,2.38301,2.39792]])

##gamma=0.05
# NN_WFR=np.array([[1.38939,1.78184,1.97542,2.09418,2.13442,2.18746,2.24437,2.24298,2.32382,2.30046],
#                  [1.39461,1.78618,2.01113,2.07367,2.16842,2.19280,2.25383,2.26576,],
#                  [1.37356,1.79800,1.98922,2.06187,2.12319,2.17085,2.33048,2.27083,],
#                  [1.40025,1.79260,1.94826,2.12166,2.16508,2.25858,2.21024,2.25501,],
#                  [1.38699,1.80611,1.98319,2.06289,2.18762,2.20164,2.20804,]])



# # alpha=50:2.511533 alpha=100:2.542784  OT:2.568021
# plt.plot(alpha_space,pd_WFR,label="primal_dual")
# # plt.plot(alpha_space,NN_WFR,label="UOT-gen")
# plt.xlim(0.1,11)
# plt.xlabel("alpha")
# plt.ylabel("WFR distance")
# plt.legend()
# plt.show()



#
# predicted expect and calculate confidence interval
data_points=10
predicted_expect = np.mean(NN_WFR, 0)
low_CI_bound, high_CI_bound = st.t.interval(0.95, data_points - 1,
                                            loc=np.mean(NN_WFR, 0),
                                            scale=st.sem(NN_WFR))

# plot confidence interval
x = np.linspace(1, data_points , num=data_points)
plt.plot(alpha_space,predicted_expect, linewidth=3., label='UOT-gen')
plt.plot(alpha_space,pd_WFR,label="primal-dual hybrid algorithm")
# plt.plot(Mu, color='r', label='grand truth')
plt.fill_between(alpha_space, low_CI_bound, high_CI_bound, alpha=0.5,)
plt.xlim(0.1,11)
plt.ylim(0.2,3)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\mathcal{J}_{SWFR}$")
plt.legend(loc=4,prop = {'size':15})
plt.show()
