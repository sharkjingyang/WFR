# add source term without applying birth-death process

# explicitly compute every weight of points, RK4 or FE

# act as OTFlowProblem.py
import numpy as np
from math import *
import torch
from torch.nn.functional import pad
from src.Phi import *
from src.KSD import *

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
cvt = lambda x: x.to(device, non_blocking=True)
def vec(x):
    """vectorize torch tensor x"""
    return x.view(-1, 1)

def density(x,centers,weights=[0.5, 0.5]):

    return sum(1/sqrt(2*pi)*exp(-(x-i[0])**2/2) * i[1] for i in zip(centers, weights))

def OTFlowProblem_1d(x, Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=1,jjj=0, device=device):

    # X_data,w_data=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy")
    # w_data=torch.from_numpy(w_data).squeeze().to(device).float()
    # X_data=np.random.normal(size=(10000,1))-3+6*(np.random.uniform(size=(10000,1))>(1/3))

    w_data=torch.ones(x.shape[0]).to(device).squeeze().float()

    # weight_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:x.shape[0],-1]
    # w_data=weight_osc/weight_osc.sum()*weight_osc.shape[0]
    # w_data=torch.from_numpy(w_data).squeeze().to(device).float()

   
    h = (tspan[1] - tspan[0]) / nt
 
    z = pad(x, (0, 5), value=0)
    
    
    z=torch.cat((z,w_data.reshape(-1,1)),dim=1)

    tk = tspan[0]

    v0 = getv0(z, Phi, tspan[0])
    if stepper == 'rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, tk, tk + h, v0, alphaa=alphaa)
            tk += h

    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    new_weight = torch.exp(z[:, -4] + z[:, -2])*w_data
    # print("check weight------")
    # print(torch.sum(z[:,-1]))
    # print(torch.sum(z[:, -4] + z[:, -2]))
    # print(torch.mean(new_weight))
    # print("check end---")
    new_weight =new_weight/torch.sum(new_weight)*new_weight.shape[0]
    

    
   
    costL = torch.sum(z[:, -3])  # dv, 0.5*rho*v^2+0.5*alpha*rho*g^2

    d = z.shape[1] - 6
    l = z[:, d]  # log-det
    if jjj == 0:
        # costC = 0.5 * d * log(2 * pi) - torch.mean(l) + 0.5 * torch.mean(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True))+torch.mean(z[:,-2])
        costC = 0.5 * d * log(2 * pi) - torch.dot(l,w_data/w_data.shape[0]) + 0.5 * torch.dot(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True).squeeze(),w_data/w_data.shape[0])+torch.dot(z[:,-2],w_data/w_data.shape[0])
                # +z[0,-4]

    elif jjj==1:
        costC =  -torch.dot(z[:,-4],w_data/w_data.shape[0])
    elif jjj==2:
        costC = 0.5 * d * log(2 * pi) - torch.mean(l) + 0.5 * torch.mean(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True)) + torch.mean(z[:, -2])+z[0,-4]

    costV = torch.mean(z[:, -5])
    cs = [costL, costC, costV]

    return costC, costL, costV, cs, new_weight, z[0,-4]

def OTFlowProblem_ex(x, Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=10,jjj=0, device=device):
    """
    BD means applying birth death process

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.

    :param x:       input data tensor nex-by-d
    :param rho_x:   density at x (\rho_0), suppose we know it
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1] - tspan[0]) / nt

    z = pad(x, (0, 5, 0, 0), value=0)
    z = pad(z, (0, 1, 0, 0), value=1)

    tk = tspan[0]

    v0 = getv0(z, Phi, tspan[0])
    if stepper == 'rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, tk, tk + h, v0, alphaa=alphaa)
            tk += h
    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    new_weight = torch.exp(z[:, -4] + z[:, -2]).unsqueeze(-1)

    # check:
    # print(z[0,-4],-torch.log(torch.mean(torch.exp(z[:,-2]))))   # 二者相等

    # new KL
    d = z.shape[1] - 6
    l = z[:, d]  # log-det
    if jjj == 0:
        costC = 0.5 * d * log(2 * pi) - torch.mean(l) + 0.5 * torch.mean(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True))+torch.mean(z[:,-2])
    else:
        costC = -z[0,-4]

    costL = torch.sum(z[:, -3]) 
    costV = torch.sum(z[:, -5])
    cs = [costL, costC, costV]
    # costC forward KL ( exclude \hat{Phi})
    # costL  0.5*rho*v^2+0.5*alpha*rho*g^2
    # costV regularization on velocity field
    # cs = [costL, costC, costV]
    # new_weight unormalized weight 
    # z[0,-4] 1/aplha*\hat{Phi}
    return costC, costL, costV, cs, new_weight, z[0, -4]   



def OTFlowProblem_ex_1d(x, rho_x,Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], is_1d=True, alphaa=10,jjj=0,device=device):
    """
    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.

    :param x:       input data tensor nex-by-d
    :param rho_x:   density at x (\rho_0), suppose we know it
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1] - tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    if is_1d:
        z = pad(x.unsqueeze(-1), (0, 5), value=0)
        z = pad(z, (0, 1, 0, 0), value=1)
    else:
        z = pad(x, (0, 5, 0, 0), value=0)
        z = pad(z, (0, 1, 0, 0), value=1)
    tk = tspan[0]

    v0 = getv0(z, Phi, tspan[0])
    if stepper == 'rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h, v0, alphaa=alphaa)
            tk += h

    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h


    new_weight = torch.exp(z[:, -4] + z[:, -2]).unsqueeze(-1)

    new_weight_2 = z[:,-1]/torch.sum(z[:,-1])

    # 二者相同
    # print(new_weight.squeeze()[:10])
    # print(z[:,-1][:10])

    costL = torch.sum(z[:, -3])  # dv, 0.5*rho*v^2+0.5*alpha*rho*g^2
    costV = torch.sum(z[:, -5])  # dp

    # new KL
    d = 1
    l = z[:, d]  # log-det
    if jjj == 0:
        # costC = 0.5 * d * log(2 * pi) - torch.mean(l) + 0.5 * torch.mean(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True))+torch.mean(z[:,-2])+z[0,-4]
        costC = 0.5*d*log(2*pi)+0.5*torch.dot(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True).squeeze(), new_weight_2)-torch.dot(l,new_weight_2)+torch.dot(z[:,-4]+z[:,-2], new_weight_2) + torch.dot(torch.log(rho_x),new_weight_2)

    else:
        costC = z[0,-4]

    cs = [costL, costC, costV]

    return costC, costL, costV, cs, new_weight, z[:, :2]


def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1] - 6
    l = z[:, d]  # log-det

    c = 0.5*d*log(2*pi)-torch.dot(l, z[:, -1]/torch.sum(z[:,-1]))+0.5*torch.dot(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True).squeeze(), z[:, -1]/torch.sum(z[:,-1]))

    return c

cvt = lambda x: x.to(device, non_blocking=True)
def stepRK4(odefun, z, Phi, t0, t1, v0, alphaa=1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+3, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+3, features at time t1
    """

    h = t1 - t0  # step size
    z0 = z


    K = h * odefun(z0, t0, Phi, v0, alphaa=alphaa)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), Phi, v0, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), Phi, v0, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + K, t0 + h, Phi, v0, alphaa=alphaa)
    z += (1.0 / 6.0) * K

    # d = z.shape[1]-5
    # z1 = pad(z[:, :d], (0, 1, 0, 0), value=t0+h)
    # Vx = Phi(z1)
    # Vx = Vx-torch.dot(Vx.squeeze(), z[:, -1])

    # return z, Vx

    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z


def integrate_ex_Bayes(x, net, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], intermediates=False, alphaa=10):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """
    # 注意alphaa要匹配

    h = (tspan[1] - tspan[0]) / nt


    z = pad(x, (0, 5, 0, 0), value=0)

    # X_data,w_data=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy")
    # w_data=torch.from_numpy(w_data).squeeze().to(device).float()
    X_data=np.random.normal(size=(10000,1))-3+6*(np.random.uniform(size=(10000,1))>(1/3))
    w_data=torch.ones(X_data.shape).to(device).squeeze().float()

    # z = pad(z, (0, 1, 0, 0), value=1)
    z=torch.cat((z,w_data.reshape(-1,1)),dim=1)

    tk = tspan[0]
    d = net.d
    phi0 = net(z[:, :d+1])

    if intermediates:  # save the intermediate values as well
        zFull = torch.zeros(*z.shape, nt + 1, device=x.device,
                            dtype=x.dtype)  # make tensor of size z.shape[0], z.shape[1], nt
        phifull = torch.zeros(z.shape[0], 1, nt+1, device=x.device,dtype=x.dtype)
        # phifull = torch.zeros(z.shape[0], 1, nt + 1, device=x.device, dtype=x.dtype)
        zFull[:, :, 0] = z

        phifull[:,:,0]=phi0

        v0 = getv0(z,net,t=tspan[0])
        if stepper == 'rk4':
            for k in range(nt):
                # zFull[:,:,k+1], V= stepRK4(odefun,zFull[:, :, k], net, alph, tk, tk + h,v0,alphaa=alphaa)
                zFull[:, :, k + 1] = stepRK4(odefun, zFull[:, :, k], net, tk, tk + h, v0, alphaa=alphaa)
                phifull[:,:,k+1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:, :, k + 1] = stepRK1(odefun, zFull[:, :, k], net, alph, tk, tk + h,alphaa=alphaa)
                phifull[:, :, k + 1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        return zFull, phifull

    else:
        v0 = getv0(z, net, t=tspan[0])
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun, z, net,  tk, tk + h, v0, alphaa=alphaa)
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z, net, alph, tk, tk + h,alphaa=alphaa)
                tk += h

        return z,0

    # return in case of error
    return -1



def odefun(x, t, net, v0, alphaa=1):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 6

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    # z2 = pad(x[:, :d], (0, 1, 0, 0), value=1-t)

    # X_data,w_data=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy")
    # w_data=torch.from_numpy(w_data).squeeze().to(device).float()
    # X_data=np.random.normal(size=(10000,1))-3+6*(np.random.uniform(size=(10000,1))>(1/3))
    X_data=x

    w_data=torch.ones(X_data.shape[0]).to(device).squeeze().float() # 初始weight

    # weight_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:x.shape[0],-1]
    # w_data=weight_osc/weight_osc.sum()*weight_osc.shape[0]
    # w_data=torch.from_numpy(w_data).squeeze().to(device).float()

    
    unnorm=torch.exp(x[:,-4]+x[:,-2])*w_data
    new_weight=unnorm/torch.sum(unnorm)*x.shape[0]

    Vx0 = net(z)
    
    ds0 = torch.dot(Vx0.squeeze(), new_weight/x.shape[0])
    Vx = Vx0-ds0

    # print("check phi-phi_bar")
    # print(torch.min(new_weight))
    # print(torch.mean(Vx))

    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    dl = -trH.unsqueeze(1)
    dp = torch.mul(new_weight.unsqueeze(-1)/x.shape[0], torch.sum(torch.pow(dx*new_weight.unsqueeze(-1)-v0*w_data.reshape(-1,1),2),1, keepdims=True))
    
    ds = (1 / alphaa * ds0) * torch.ones_like(dl)
    dv = torch.mul(0.5 * (torch.sum(torch.pow(dx, 2), 1, keepdims=True)+1/alphaa*(Vx**2)), new_weight.unsqueeze(-1)/x.shape[0])
    ds2= -1/alphaa*Vx0
    dw = -1/alphaa*torch.mul(Vx, new_weight.unsqueeze(-1))  # unormalized
 

    # v00 = v0/new_weight.unsqueeze(-1)
    return torch.cat((dx, dl, dp, ds, dv, ds2, dw), 1)

def getv0(x,net,t):

    nex, d_extra = x.shape
    d = d_extra - 6
    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

    gradPhi = net.trHess(z, justGrad=True)

    return -gradPhi[:, 0:d]

# def getv1(x,net):
#
#     nex, d_extra = x.shape
#     d = d_extra - 6
#     z = pad(x[:, :d], (0, 1, 0, 0), value=1)  # concatenate with the time t
#
#     gradPhi, trH = net.trHess(z)
#
#     return -gradPhi[:, 0:d]

def odefun_copy(x, t, net, v0, alphaa=1):



    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 6

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    # z2 = pad(x[:, :d], (0, 1, 0, 0), value=1-t)

    new_weight = torch.exp(x[:,-4]+x[:,-2]) /(torch.sum(torch.exp(x[:,-4]+x[:,-2])))*x.shape[0]

    # new_weight[new_weight<10**(-5)]=0
    # z = z[new_weight>10**(-5)]
    Vx0 = net(z)
    # Vx02 = net(z2)
    ds0 = torch.dot(Vx0.squeeze(), new_weight/x.shape[0])
    Vx = Vx0-ds0

    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    dl = -trH.unsqueeze(1)
    dv = torch.mul(0.5 * (torch.sum(torch.pow(dx, 2), 1, keepdims=True)+1/alphaa*(Vx**2)), new_weight.unsqueeze(-1)/x.shape[0])


    dw = -1/alphaa*torch.mul(Vx, new_weight.unsqueeze(-1))


    ds2=-1/alphaa*Vx0

    v00 = v0/new_weight.unsqueeze(-1)

    dp = torch.mul(new_weight.unsqueeze(-1)/x.shape[0], torch.sum(torch.pow(dx*new_weight.unsqueeze(-1)-v0,2),1, keepdims=True))

    ds = (1 / alphaa * ds0) * torch.ones_like(dw)

    return torch.cat((dx, dl, dp, ds, dv, ds2, dw), 1)

def integrate_ex_copy(x, net, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], intermediates=False, alphaa=10):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """
    # 注意alphaa要匹配

    h = (tspan[1] - tspan[0]) / nt

    z = pad(x, (0, 5, 0, 0), value=0)
    z = pad(z, (0, 1, 0, 0), value=1)

    tk = tspan[0]
    d = net.d
    phi0 = net(z[:, :d+1])

    if intermediates:  # save the intermediate values as well
        zFull = torch.zeros(*z.shape, nt + 1, device=x.device,
                            dtype=x.dtype)  # make tensor of size z.shape[0], z.shape[1], nt
        phifull = torch.zeros(z.shape[0], 1, nt+1, device=x.device,dtype=x.dtype)
        # phifull = torch.zeros(z.shape[0], 1, nt + 1, device=x.device, dtype=x.dtype)
        zFull[:, :, 0] = z

        phifull[:,:,0]=phi0

        v0 = getv0(z,net,t=tspan[0])
        if stepper == 'rk4':
            for k in range(nt):
                # zFull[:,:,k+1], V= stepRK4(odefun,zFull[:, :, k], net, alph, tk, tk + h,v0,alphaa=alphaa)
                zFull[:, :, k + 1] = stepRK4(odefun_copy, zFull[:, :, k], net, tk, tk + h, v0, alphaa=alphaa)
                phifull[:,:,k+1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:, :, k + 1] = stepRK1(odefun_copy, zFull[:, :, k], net, alph, tk, tk + h,alphaa=alphaa)
                phifull[:, :, k + 1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        return zFull, phifull

    else:
        v0 = getv0(z, net, t=tspan[0])
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun_copy, z, net,  tk, tk + h, v0, alphaa=alphaa)
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun_copy,z, net, alph, tk, tk + h,alphaa=alphaa)
                tk += h

        return z,0

    # return in case of error
    return -1
