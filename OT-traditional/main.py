import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg
import time
# def mu(x): #mix Gaussian
#     a=1/np.sqrt(2*np.pi)*np.exp(-(x+3)**2/2)
#     b=1/np.sqrt(2*np.pi)*np.exp(-(x-3)**2/2)
#     return 1/3*a+2/3*b

# def nu(x): #Gaussian
#     return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def Gaussian(x,mu,sigma): #Gaussian
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

def rho_0(x):
    return Gaussian(x,mu=1/5,sigma=math.sqrt(1e-3))

def rho_1(x):
    return Gaussian(x,mu=4/5,sigma=math.sqrt(1e-3))*1.4

Nx=40
Nt=30
alpha=1
tao=0.1
X=[0,1]
T=[0,1]
x_space=np.linspace(X[0],X[1],Nx)
t_space=np.linspace(T[0],T[1],Nt+1)
mu=np.random.uniform(low=0,high=1,size=(Nt+1,Nx))
# mu=np.ones((Nt+1,Nx))/Nx
mu[0,:]=rho_0(x_space)
mu[-1,:]=rho_1(x_space)
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

def dE_dmu_n(mu,n):
    L_mu_n_inv=np.linalg.inv(Operator_L_u_alpha(mu[n,:]))
    L_mu_nminus1_inv=np.linalg.inv(Operator_L_u_alpha(mu[n-1,:]))

    A=-2*np.matmul(L_mu_n_inv,mu[n+1,:]-mu[n,:])
    B=2*np.matmul(L_mu_nminus1_inv,mu[n,:]-mu[n-1,:])
    c=np.matmul(L_mu_n_inv,mu[n+1,:]-mu[n,:])
    # A=-2*np.linalg.solve(Operator_L_u_alpha(mu[n,:]),mu[n+1,:]-mu[n,:])
    # B=2*np.linalg.solve(Operator_L_u_alpha(mu[n-1,:]),mu[n,:]-mu[n-1,:])
    # c=A*-0.5
    
    C=np.zeros(Nx)
    for i in range(Nx):
        C[i]=-L_ei(c,i)
    return (A+B+C)*dx/dt

def L_ei(mu,i):  ##calculate <u,L_ei*u>
    if i==0:
        return 0.5*(mu[i+1]-mu[i])**2/dx**2
    if i==Nx-1:
        return 0.5*(mu[i]-mu[i-1])**2/dx**2
    if 1<=i<=Nx-2:
        return 0.5*(mu[i]-mu[i-1])**2/dx**2+0.5*(mu[i+1]-mu[i])**2/dx**2


X, Y = np.meshgrid(x_space, t_space)
lam=0
k_max=100000
for k in range(1,k_max):
    print(k)
    lam=0.5*(1+math.sqrt(1+4*lam**2))
    lam_nex=0.5*(1+math.sqrt(1+4*lam**2))
    gamma=(1-lam)/lam_nex
    mu_half=mu.copy()
    for t in range(1,Nt):
        mu_half[t,:]=mu[t,:]-tao*dE_dmu_n(mu,t)
    
    mu_half[mu_half<0]=0

    mu=(1-gamma)*mu_half+gamma*mu

    if (k+1)%100==0:
        
        # fig = plt.figure()
        plt.ion()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, mu, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('mu')
        ax.set_title('3D Plot')
        plt.show()
        plt.pause(0.2)
        plt.clf()

np.save("mu.npy",mu)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, mu, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('mu')
# ax.set_title('3D Plot')
# plt.show()




