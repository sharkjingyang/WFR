# trainToyOTflow.py
# training driver for the two-dimensional toy problems
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

# from src.OTFlowProblem import *
# from src.birth_death_process import *
from source_without_BD import *
# from src.mmd import *
from src.mmd_new import *

import config

cf = config.getconfig()

if cf.gpu: # if gpu on platform
    def_viz_freq = 500
    def_batch    = 4096
    def_niter    = 2000
else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 100
    def_batch    = 2048
    def_niter    = 1000

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['1d','swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='1d'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps") # origin:8
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation") # origin:8
parser.add_argument('--alph'  , type=str, default='1.0,100.0, 5.0')  # origin 1, 100, 5
parser.add_argument('--m'     , type=int, default=32)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=0.1)
parser.add_argument("--drop_freq"   , type=int  , default=100, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=25)
parser.add_argument('--alphaa', type=float, default=10)

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


def compute_loss(net, x, rho_x, nt):
    print(nt)
    Jc, cs, weights, position = OTFlowProblem_ex(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True, alphaa=args.alphaa)
    return Jc, cs, weights, position


if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # neural network for the potential function Phi
    d      = 1
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

    # setup data [nSamples, d]
    # use one batch as the entire data set
    print(args.data)
    x0, rho_x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
    x0 = cvt(torch.from_numpy(x0))
    rho_x0 = cvt(torch.from_numpy(rho_x0))

    x0val, rho_x0val = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size, require_density=False)
    x0val = cvt(torch.from_numpy(x0val))
    rho_x0val = cvt(torch.from_numpy(rho_x0val))

    # x0val2, rho_x0val2 = toy_data.inf_train_gen(args.data, batch_size=, require_density=False)
    # x0val2 = cvt(torch.from_numpy(x0val2))
    # rho_x0val2 = cvt(torch.from_numpy(rho_x0val2))

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
    t = 0
    for itr in range(1, args.niters + 1):
        # train

        optim.zero_grad()
    
        loss, costs, weights,position  = compute_loss(net, x0, rho_x0, nt=args.nt)

        loss.backward()
        optim.step()
        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                itr, time_meter.val , loss, costs[0], costs[1], costs[2]
            )
        )
        t = t+time_meter.val
        # validate
        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                net.eval()
                test_loss, test_costs, test_weights, test_position = compute_loss(net, x0val, rho_x0val, nt=nt_val)

                nSamples = 100
                y1 = cvt(torch.randn(nSamples, d))
                genModel, _ = integrate_ex(y1[:, 0:d], net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph,
                                           alphaa=args.alphaa)

                mmd1 = MMD_Weighted(x0val.unsqueeze(-1), genModel[:, 0].unsqueeze(-1), genModel[:, -1].unsqueeze(-1))


                # add to print message
                log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} '.format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2], mmd1
                )

                MMD_list.append(float(mmd1))
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

                nSamples = 2000
                p_samples, _ = cvt(torch.Tensor(toy_data.inf_train_gen(args.data, batch_size=nSamples, require_density=False) ))
                y = cvt(torch.randn(nSamples, d)) # sampling from the standard normal (rho_1)

                sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                sPath2 = os.path.join(args.save, 'figs(2)', start_time + '_{:04d}.png'.format(itr))
                plot1d(net, p_samples, y, nt_val, sPath,sPath2, doPaths=True,
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
            x0, rho_x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
            x0 = cvt(torch.from_numpy(x0))
            rho_x0 = cvt(torch.from_numpy(rho_x0))


        end = time.time()

    print(ITR)
    print(MMD_list)
    print(Time)
    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))