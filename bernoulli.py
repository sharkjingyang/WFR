from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import math

dt=1
sigma=0.4
T=50

def analytical_sol(x,t):
  return x/np.sqrt(x**2+(1-x**2)*np.exp(-2*t))

##True solution
T_space=np.linspace(0,T,num=1000)
true_sol=analytical_sol(x=0.2,t=T_space)

#observed data
dt_space=np.arange(start=0,stop=T+dt,step=dt)

observe_data=analytical_sol(x=0.2,t=dt_space)+np.random.normal(loc=0,scale=sigma,size=dt_space.shape)


# plt.plot(T_space,true_sol,"--",label="analytic solution")
# plt.scatter(dt_space[1:],observe_data[1:],s=4,color="red",label="observation")
# plt.ylim(-2,3)
# plt.xlabel("t")
# plt.ylabel("v(t)")
# plt.grid(linestyle = '--')
# plt.legend()
# plt.show()


def Gaussian(x,mu,sigma):
  return 1/np.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)



#calculate probability
def cal_pro(x_prior,ob_data,dt,Tspan):
  dt_space=np.arange(start=Tspan[0]+dt,stop=Tspan[1]+dt/2,step=dt)
  true_sol=analytical_sol(x=x_prior,t=dt_space)
  # print(int(Tspan[0]/dt))
  # print(int(Tspan[1]/dt))

  p=Gaussian(ob_data[int(Tspan[0]/dt)+1:int(Tspan[1]/dt)+1],true_sol,sigma)
  # print(p)
  # print(np.exp(np.sum(np.log(p))))
  return np.exp(np.sum(np.log(p)))


X_prior=np.random.normal(loc=0.5,scale=1,size=(10000,1))
w=np.zeros(X_prior.shape)
for i in range(X_prior.shape[0]):
  w[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=1,Tspan=[0,50])
w=w/np.sum(w)*X_prior.shape[0]





np.save("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy",(X_prior,w))
p=np.argsort(X_prior,axis=0)
X_prior=np.sort(X_prior,axis=0)
w=w[p].squeeze().reshape(-1,1)
w=w*Gaussian(X_prior,0.5,1)
# w=w/np.sum(w)*X_prior.shape[0]
plt.plot(X_prior,w,"red",label="real posterior")
plt.grid(linestyle="--")
plt.xlabel("$x_0$",fontsize=20)
plt.ylabel("Density",fontsize=20)
plt.xlim(-1,3)
plt.ylim(0,3)
# plt.title("T=0.5",fontsize=30)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(fontsize=12)
plt.savefig("/home/liuchang/OT-Flow/posterior_data/inverse.png", dpi=300)
plt.close()


# w_1=np.zeros(X_prior.shape)
# w_2=np.zeros(X_prior.shape)
# w_3=np.zeros(X_prior.shape)
# w_4=np.zeros(X_prior.shape)
# w_5=np.zeros(X_prior.shape)
# w_12=np.zeros(X_prior.shape)
# w_23=np.zeros(X_prior.shape)
# w_34=np.zeros(X_prior.shape)
# w_45=np.zeros(X_prior.shape)

# for i in range(X_prior.shape[0]):
#   w_1[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=0.1,Tspan=[0,0.5])
#   w_2[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=0.1,Tspan=[0,1])
#   w_3[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=0.1,Tspan=[0,2])
#   w_4[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=0.1,Tspan=[0,4])
#   w_5[i]=cal_pro(x_prior=X_prior[i],ob_data=observe_data,dt=0.1,Tspan=[0,5])


# # w=w/np.sum(w)*X_prior.shape[0]

# w_1=w_1/np.sum(w_1)*X_prior.shape[0]
# w_2=w_2/np.sum(w_2)*X_prior.shape[0]
# w_3=w_3/np.sum(w_3)*X_prior.shape[0]
# w_4=w_4/np.sum(w_4)*X_prior.shape[0]
# w_5=w_5/np.sum(w_5)*X_prior.shape[0]


# np.save("/home/liuchang/OT-Flow/posterior_data/observe.npy",observe_data)
# np.save("/home/liuchang/OT-Flow/posterior_data/T1.npy",(X_prior,w_1))
# np.save("/home/liuchang/OT-Flow/posterior_data/T2.npy",(X_prior,w_2))
# np.save("/home/liuchang/OT-Flow/posterior_data/T3.npy",(X_prior,w_3))
# np.save("/home/liuchang/OT-Flow/posterior_data/T4.npy",(X_prior,w_4))
# np.save("/home/liuchang/OT-Flow/posterior_data/T5.npy",(X_prior,w_5))

