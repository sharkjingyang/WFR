# add source term without applying birth-death process

# explicitly compute every weight of points, RK4 or FE

# act as OTFlowProblem.py
import numpy as np
from math import *
import torch
from torch.nn.functional import pad
from src.Phi import *
from src.KSD import *

device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')
cvt = lambda x: x.to(device, non_blocking=True)
def vec(x):
    """vectorize torch tensor x"""
    return x.view(-1, 1)

def density(x,centers,weights=[0.5, 0.5]):

    return sum(1/sqrt(2*pi)*exp(-(x-i[0])**2/2) * i[1] for i in zip(centers, weights))

def OTFlowProblem_ex(x, rho_x, Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0], is_1d=False, alphaa=1,device=device):
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

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    if is_1d:
        z = pad(x.unsqueeze(-1), (0, 4), value=0)
        z = pad(z, (0, 1, 0, 0), value=1/x.shape[0])
    else:
        z = pad(x, (0, 4, 0, 0), value=0)
        z = pad(z, (0, 1, 0, 0), value=1 / x.shape[0])
    tk = tspan[0]

    TT = torch.zeros(z.shape[0],1).to(device)
    # costV = torch.zeros(1).to(device)

    v0 = getv0(z, Phi)
    if stepper == 'rk4':
        for k in range(nt):
            z, V= stepRK4(odefun, z, Phi, alph, tk, tk + h, v0, alphaa=alphaa)
            tk += h
            TT = TT + V

    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # ASSUME all examples are equally weighted
    # e_term = torch.pow(e, -h*TT/alphaa)
    # upset_term = 1/x.shape[0]*torch.dot(torch.log(rho_x).squeeze(), e_term.squeeze())
    # print("sum w_i:{}".format(torch.sum(z[:,-1])))

    costL = torch.sum(z[:, -2])  # dv, 0.5*rho*v^2+0.5*alpha*rho*g^2
    # upset_term = torch.dot(torch.log(rho_x).squeeze(), z[:, -1])
    # costC = C(z)-1/alphaa*h*torch.dot(TT.squeeze(), z[:, -1])+upset_term   # KL divergence without ## log(rho0)##


    costC = KSD(z[:, 0:2], z[:, -1].unsqueeze(1), h=1)
    # costC=KSD_IMQ(z[:, 0:2], z[:, -1].unsqueeze(1), c=1, l=1, beta=-0.5)
    # costC = FSSD(z[:, 0:2], z[:, -1].unsqueeze(1), J = 4096, h=5)
    # costC = FSSD_IMQ(z[:, 0:2], z[:, -1].unsqueeze(1), J = 4096,c=1,l=1,beta=-0.5)
    # costC = 0.5*torch.dot(torch.sum(torch.pow(z[:, 0:2], 2), 1, keepdims=True).squeeze(), z[:, -1])
    # f0 = 0
    # aa = cvt(torch.zeros(z[:,:2].shape))
    # aa[:,0] = z[:,0]/torch.linalg.norm(z[:, 0])
    # aa[:,1] = z[:,1]/torch.linalg.norm(z[:, 1])
    # n = z.shape[0]
    # for i in range(2):
    #     # |x_i-x_j|^2
    #     f_temp = (aa[:, i].unsqueeze(1).T - aa[:, i].unsqueeze(1))**2
    #     f0 = f0 + f_temp
    # f1 = cvt(torch.eye(n))
    # f0 = f0+f1
    # costC2 = (torch.sum(1/(torch.sqrt(f0)))-n)/n/(n-1)

    # ka = torch.sum(torch.pow(z[:, 0:2], 2), dim=1, keepdim=True)
    # costC2 = KSD_kappa(ka, z[:, -1].unsqueeze(1), h=1)   # 卡方
    # costC2 = (torch.dot(z[:,0]/torch.linalg.norm(z[:,0]), z[:,1]/torch.linalg.norm(z[:,1])))**2  # mutual information

    # costR = torch.sum(z[:, -3])   # HJB
    costV = torch.sum(z[:, -4])
    # A = torch.cat(((z[:,0]/torch.linalg.norm(z[:,0])).unsqueeze(1), (z[:,1]/torch.linalg.norm(z[:,1])).unsqueeze(1)), dim=1)

    # aa, aaa = torch.sum(A[:,0]**2), torch.sum(A[:,1]**2)
    cs = [costL, costC, costV]

    # return dot(cs, alph)  , cs
    return sum(i[0] * i[1] for i in zip(cs, alph)), cs, z[:, -1], z[:, :1]

cvt = lambda x: x.to(device, non_blocking=True)
def stepRK4(odefun, z, Phi, alph, t0, t1, v0, alphaa=1):
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

    d = z.shape[1]-5
    z1 = pad(z[:, :d], (0, 1, 0, 0), value=t0+h)
    Vx = Phi(z1)
    Vx = Vx-torch.dot(Vx.squeeze(), z[:, -1])

    # diffusion
    # z[:,:2]=z[:,:2]+cvt(0.01 * torch.randn(z[:,:2].shape))
    return z, Vx


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


def integrate_ex(x, net, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0, 1.0], intermediates=False, alphaa=50):
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

    h = (tspan[1] - tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 4, 0, 0), value=tspan[0])
    z = pad(z, (0, 1, 0, 0), value=1/z.shape[0])

    tk = tspan[0]
    d = net.d
    phi0 = net(z[:, :d+1])

    if intermediates:  # save the intermediate values as well
        zFull = torch.zeros(*z.shape, nt + 1, device=x.device,
                            dtype=x.dtype)  # make tensor of size z.shape[0], z.shape[1], nt
        phifull = torch.zeros(z.shape[0], 1, nt+1, device=x.device,dtype=x.dtype)
        zFull[:, :, 0] = z

        phifull[:,:,0]=phi0

        v0 = getv0(z,net)
        if stepper == 'rk4':
            for k in range(nt):
                zFull[:,:,k+1], V= stepRK4(odefun,zFull[:, :, k], net, alph, tk, tk + h,v0,alphaa=alphaa)
                phifull[:,:,k+1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:, :, k + 1] = stepRK1(odefun, zFull[:, :, k], net, alph, tk, tk + h,alphaa=alphaa)
                phifull[:, :, k + 1] = net(pad(z[:, :d], (0, 1, 0, 0), value=tk))
                tk += h

        return zFull, phifull

    else:
        v0 = getv0(z, net)
        if stepper == 'rk4':
            for k in range(nt):
                z, V= stepRK4(odefun, z, net, alph, tk, tk + h,v0,alphaa=alphaa)
                tk += h

        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z, net, alph, tk, tk + h,alphaa=alphaa)
                tk += h

        return z,0

    # return in case of error
    return -1


def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1] - 5
    l = z[:, d]  # log-det

    c = 0.5*d*log(2*pi)-torch.dot(l, z[:, -1])+0.5*torch.dot(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True).squeeze(), z[:, -1])

    return c


def odefun(x, t, net, v0, alphaa=0.1):
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
    d = d_extra - 5

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

    # compute V(z(x_i,t))-\bar V(z(x,t))
    Vx0 = net(z)
    Vx = Vx0-torch.dot(Vx0.squeeze(), x[:, -1])

    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    # dl = -(1.0 / alph[0]) * trH.unsqueeze(1)+Vx         # V-\bar V 积分为0
    dl = -trH.unsqueeze(1)
    dv = torch.mul(0.5 * (torch.sum(torch.pow(dx, 2), 1, keepdims=True)+1/alphaa*(Vx**2)), x[:, -1].unsqueeze(-1))
    # origin paper: dr = torch.abs(-gradPhi[:, -1].unsqueeze(1) + alph[0] * dv)
    # new dr
    dw = -1/alphaa*torch.mul(Vx, x[:, -1].unsqueeze(-1))
    dr = torch.mul(torch.abs(-gradPhi[:, -1].unsqueeze(1) + 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)+0.5/alphaa*torch.sum(torch.pow(Vx0[:, 0:d], 2) , 1 ,keepdims=True)),x[:, -1].unsqueeze(-1))


    v1 = v0/(nex*x[:,-1].unsqueeze(-1))
    dp = torch.mul(x[:, -1].unsqueeze(-1), torch.sum(torch.pow(dx-v1,2),1, keepdims=True))

    return torch.cat((dx, dl, dp, dr, dv, dw), 1)

def getv0(x,net):

    nex, d_extra = x.shape
    d = d_extra - 5
    z = pad(x[:, :d], (0, 1, 0, 0), value=0)  # concatenate with the time t

    gradPhi, trH = net.trHess(z)

    return -gradPhi[:, 0:d]