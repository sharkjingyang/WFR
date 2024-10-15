import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg
import time

mu=np.load("alpha=100.npy")
Nx=40
Nt=30
alpha=1
tao=1
X=[0,1]
T=[0,1]
x_space=np.linspace(X[0],X[1],Nx)
t_space=np.linspace(T[0],T[1],Nt+1)
X, Y = np.meshgrid(x_space, t_space)
dx=1/(Nx-1)
dt=1/Nt


def Operator_L_u_alpha(mu):
    L=np.zeros((Nx,Nx))
    for i in range(1,Nx-1):
        L[i,i]=0.5/dx**2*(mu[i-1]+mu[i])+0.5/dx**2*(mu[i]+mu[i+1])
    L[0,0]=0.5/dx**2*(mu[0]+mu[1])
    L[Nx-1,Nx-1]=0.5/dx**2*(mu[Nx-2]+mu[Nx-1])
    for i in range(Nx-1):
        L[i,i+1]=-0.5/dx**2*(mu[i]+mu[i+1])
        L[i+1,i]=-0.5/dx**2*(mu[i]+mu[i+1])
    L=L+np.eye(Nx)*alpha
    return L

def E_mu(mu):  ## WFR distance
    L2_mu_n=0
    for n in range(Nt):
        mu_n=mu[n,:]
        L=Operator_L_u_alpha(mu_n)
        L_inv=np.linalg.inv(L)
        s=(mu[n+1,:]-mu[n,:])
        L2_mu_n=L2_mu_n+np.matmul(np.matmul(s.T,L_inv),s)
    L2_mu_n=L2_mu_n*dx/dt
    return L2_mu_n
print("UW_2=%f "%E_mu(mu))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, mu, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('mu')
ax.set_title('3D Plot')
plt.show()

plt.plot(x_space,mu[15,:])
plt.show()