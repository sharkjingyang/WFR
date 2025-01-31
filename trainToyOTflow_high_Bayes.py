# trainToyOTflow.py
# training driver for the two-dimensional toy problems
import sys
import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters
from src.plotter import plot1d
import datasets


from source_without_BD_Bayes import *
from src.mmd_new import *

import config

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
    
    elif name =="SVGD_data":
        return datasets.SVGD_DATA()

    else:
        raise ValueError('Unknown dataset')
cf = config.getconfig()

# if cf.gpu: # if gpu on platform
#     def_viz_freq = 1000
#     def_batch    = 100
#     def_niter    = 500
# else:  # if no gpu on platform, assume debugging on a local cpu
#     def_viz_freq = 100
#     def_batch    = 2048
#     def_niter    = 1000

def_viz_freq = 100
def_batch = 1000
def_niter = 1000

parser = argparse.ArgumentParser('UOT-Flow')
parser.add_argument(
    '--data', choices=['1d','swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','WFR_high_dim',],
    type=str, default='WFR_high_dim'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps") # origin:8
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation") # origin:8
parser.add_argument('--alph'  , type=str, default='1.0,100.0, 5.0')  # origin 1, 100, 5
parser.add_argument('--m'     , type=int, default=256)  #32 for us
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=0.04)
parser.add_argument("--drop_freq"   , type=int  , default=50000, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=10000)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=20000)
parser.add_argument('--alphaa', type=float, default=10)   #source term coefficient

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]


# get precision type
if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = torch.device( 'cpu')


def stepRK4_new(odefun_new, z, Phi, t0, t1, alphaa=args.alphaa):


    h = t1 - t0  # step size
    z0 = z

    K = h * odefun_new(z0, t0, Phi,alphaa=alphaa)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun_new(z0 + 0.5 * K, t0 + (h / 2), Phi, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun_new(z0 + 0.5 * K, t0 + (h / 2), Phi, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun_new(z0 + K, t0 + h, Phi, alphaa=alphaa)
    z += (1.0 / 6.0) * K

    return z

def odefun_new(x, t, net,  alphaa=args.alphaa):
    
    d = x.shape[1]-1
    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    Vx0 = net(z)
    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    ds2= -1/alphaa*Vx0

    return torch.cat((dx, ds2), 1)

def compute_loss_1(net, x,x2, nt):

    # costC forward KL ( exclude \hat{Phi})
    # costL  0.5*rho*v^2+0.5*alpha*rho*g^2
    # costV regularization on velocity field
    # cs = [costL, costC, costV]
    # new_weight unormalized weight 
    # z[0,-4] 1/aplha*\hat{Phi}
    costC1, costL1, costV1, cs, weights, aaaa1 = OTFlowProblem_1d(x, net, [0,1], nt=nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=args.alphaa,jjj=0, device=device)
    # costC2, costL2, costV2, cs2, weights2, aaaa2 = OTFlowProblem_1d(x2, net, [1,0], nt=nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=args.alphaa,jjj=1, device=device)

    tspan = [1, 0]
    h = (tspan[1] - tspan[0]) / nt

    z2 = pad(x2, (0, 1, 0, 0), value=0)

    tk = tspan[0]
    for k in range(nt):
        z2 = stepRK4_new(odefun_new, z2, net, tk, tk + h, alphaa=args.alphaa)
        tk += h

    costC2 = torch.log(torch.mean(torch.exp(z2[:, -1])))
    Jc = (costC1+costC2) * 100 + costL1 * 1 +costV1*1 +torch.abs(aaaa1-costC2)*1
    cs[1] = cs[1] + costC2
    return Jc, cs, weights, 0  ##total cost


def f(x):
    return x**2

if __name__ == '__main__':


    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    ## setup data [nSamples, d]
    ## use one batch as the entire data set
    print(args.data)
    if args.data=='WFR_high_dim':
        theta_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:1000,:-1]
        weight_osc=np.load("data/data_WFR/theta_SVGD_osc_weight.npy")[:1000,-1]
        weight_osc_normalized=weight_osc/weight_osc.sum()*weight_osc.shape[0]
        mu = (theta_osc).mean(axis=0)
        s = theta_osc.std(axis=0)
        # print(weight_osc_normalized.max())
        theta_osc=(theta_osc-mu)/s
        d=theta_osc.shape[1]
        x0=torch.from_numpy(theta_osc).to(device).squeeze().float().reshape(-1,d)
        x2 = torch.randn(size=(theta_osc.shape[0],d)).to(device)

    if args.data=='Bernoulli':
        X_data,w_data=np.load("/home/liuchang/OT-Flow/single_plot_Baysian/bernoulli.npy")
        w_data=torch.from_numpy(w_data).to(device).squeeze().float()
        x0=torch.from_numpy(X_data).to(device).squeeze().float()
        x2 = torch.randn(size=(x0[0],d)).to(device)

    
    # else:
    #     # data = load_data(args.data)
    #     X_data = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
    #     x0=torch.from_numpy(X_data).to(device).squeeze().float()
    #     d=x0.shape[1]
    #     x2 = torch.randn(size=(X_data.shape[0],d)).to(device)



    ## neural network for the potential function Phi
    d      = x0.shape[1]
    alph   = args.alph
    nt     = args.nt
    nt_val = args.nt_val
    nTh    = args.nTh
    m      = args.m
    net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net = net.to(prec).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay ) # lr=0.04 good

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    logger.info("nt={:}   nt_val={:}".format(nt, nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    end = time.time()
    best_loss = float('inf')
    bestParams = None

   

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s} {:9s}     {:9s}  {:9s}  {:9s}  {:9s} {:9s} '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'V(velocity)', 'valLoss', 'valL', 'valC', 'valR',
            'valV'
        )
    )

    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    net.train()
    MMD_list = []
    ITR = []
    Time = []
    MSE=[]
    DEVI=[]
    t = 0

   

    costL_list=[]
    for itr in range(1, args.niters + 1):
        # train
        optim.zero_grad()

        loss, costs, weights,devi = compute_loss_1(net, x0, x2,nt=args.nt)
        loss.backward()
        optim.step()
        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}   {:9.3f}  {:9.3f}  {:9.3f}  {:9.3f} '.format(
                itr, time_meter.val , loss, costs[0], costs[1], costs[2]
            )
        )
        # costC forward KL ( exclude \hat{Phi})
        # costL  0.5*rho*v^2+0.5*alpha*rho*g^2
        # costV regularization on velocity field
        # cs = [costL, costC, costV]


        t = t+time_meter.val
        costL_list.append(float(costs[0]))


        # save best set of parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_costs = costs
            utils.makedirs(args.save)
            best_params = net.state_dict()
            torch.save({
                    'args': args,
                    'state_dict': best_params,
                }, os.path.join(args.save,
                                start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data, int(alph[1]),
                                                                                        int(alph[2]), m)))
            net.train()


        # validate
        if itr % args.val_freq == 0 :
            with torch.no_grad():
                net.eval()

                x0val, rho_x0val = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,
                                                          require_density=False)
                x0val = cvt(torch.from_numpy(x0val))
                x3 = torch.randn(args.batch_size).to(device)


                test_loss, test_costs, test_weights, test_devi = compute_loss_1(net, x0val,x3, nt=nt_val)


                nSamples = args.batch_size
                y1 = cvt(torch.randn(nSamples, d))
                genModel, _ = integrate_ex_Bayes(y1[:, 0:d], net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph,
                                           alphaa=args.alphaa)

                # mmd1 = MMD_Weighted(x0val.unsqueeze(-1), genModel[:, 0].unsqueeze(-1), genModel[:, -1].unsqueeze(-1))

                total_weights=torch.sum(genModel[:,-1])
                # print('total_weights:', total_weights)
                # mse = (torch.dot(genModel[:,0]**2, genModel[:,-1])/total_weights-10)**2


                # add to print message
                log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} '.format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2], 0
                )

                ITR.append(itr)
                Time.append(t)

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss   = test_loss.item()
                    best_costs = test_costs
                    utils.makedirs(args.save)
                    best_params = net.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict': best_params,
                    }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data, int(alph[1]), int(alph[2]),m)))
                    net.train()

        logger.info(log_message)

        # create plots
        if itr % args.viz_freq == 0:
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                nSamples = 1000

              
                if args.data=='WFR_high_dim':
                    p_samples=torch.from_numpy(theta_osc[:nSamples,:]).to(device).squeeze().float().reshape(-1,d)
                else:
                    p_samples= cvt(torch.Tensor(toy_data.inf_train_gen(args.data, batch_size=nSamples, require_density=False) ))


                y = cvt(torch.randn(nSamples, d)) # sampling from the standard normal (rho_1)

                sPath_1 = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                sPath_2 = os.path.join(args.save, 'figs', start_time + '_{:04d}_time.png'.format(itr))
                sPath_3 = os.path.join(args.save, 'figs', start_time + '_{:04d}_weight_1.png'.format(itr))
                sPath_4 = os.path.join(args.save, 'figs', start_time + '_{:04d}_weight_2.png'.format(itr))
                sPath_5 = os.path.join(args.save, 'figs', start_time + '_{:04d}_unweighted.png'.format(itr))

                plot1d(net, p_samples, y, nt_val, sPath_1, sPath_2, sPath_3, sPath_4, sPath_5,doPaths=True,
                          sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                                 ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1],
                                                                        alph[2], nt, m, nTh), alphaa=args.alphaa)

        

                net.load_state_dict(curr_state)
                net.train()

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p['lr'] /= args.lr_drop
            print("lr: ", p['lr'])

        # resample data
        if itr % args.sample_freq == 0:
            logger.info("resampling")
            x0= toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
            x0 = cvt(torch.from_numpy(x0))
            x2 = torch.randn(size=(x0.shape[0],d)).to(device)

        end = time.time()

    print('costL_list:', costL_list)
    # print(ITR)

    # print('Inverse_Flow_MMD',MMD_list)
    # print('Inverse_Flow_MSE',MSE)
    print(Time)
    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))