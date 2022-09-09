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
from src.plotter import *

# from src.OTFlowProblem import *
# from src.birth_death_process import *
from source_without_BD import *

import config

cf = config.getconfig()

if cf.gpu: # if gpu on platform
    def_viz_freq = 100
    def_batch    = 5000
    def_niter    = 10000
else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 100
    def_batch    = 2048
    def_niter    = 1000

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['1d','swissroll','2gaussians','8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='rings'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps") # origin:8
parser.add_argument("--nt_val", type=int, default=12, help="number of time steps for validation") # origin:8
parser.add_argument('--alph'  , type=str, default='1.0,10.0,1.0,1.0') # origin 1, 100, 5
parser.add_argument('--m'     , type=int, default=32)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=0.05)
parser.add_argument("--drop_freq"   , type=int  , default=1000, help="how often to decrease learning rate") # origin 100
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='double', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=50)
parser.add_argument('--gpu'     , type=int, default=3)
parser.add_argument('--sample_freq', type=int, default=25)
parser.add_argument('--logrho0_freq', type=int, default=1000)
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
    # Jc , cs = OTFlowProblem_BD(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True,alphaa=args.alphaa)
    Jc, cs, weights, position = OTFlowProblem_ex(x, rho_x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph, is_1d=False, alphaa=args.alphaa,device=device)
    # Jc, cs = OTFlowProblem(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True)
    return Jc, cs, weights, position


if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # neural network for the potential function Phi
    d      = 2
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
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
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
    x0, rho_x0, d_net = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,require_density=True,device=device)
    x0 = cvt(torch.from_numpy(x0))
    rho_x0 = cvt(torch.from_numpy(rho_x0))

    x0val, rho_x0val, d_net2 = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size,require_density=True,device=device)
    # x0val = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size, require_density=True)
    x0val = cvt(torch.from_numpy(x0val))
    rho_x0val = cvt(torch.from_numpy(rho_x0val))

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s} {:9s}     {:9s}  {:9s}  {:9s}  {:9s} {:9s} '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'V(velocity)', 'valLoss', 'valL', 'valC', 'valR',
            'valV'
        )
    )

    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    net.train()
    for itr in range(1, args.niters + 1):
        # train
        optim.zero_grad()
        loss, costs, weights, position = compute_loss(net, x0, rho_x0, nt=args.nt)
        loss.backward()
        optim.step()


        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} '.format(
                itr, time_meter.val , loss, costs[0], costs[1], costs[2], costs[3]
            )
        )

        # validate
        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                net.eval()
                test_loss, test_costs, test_weights, test_position = compute_loss(net, x0val, rho_x0val, nt=nt_val)

                # add to print message
                log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} '.format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2], test_costs[3]
                )

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss   = test_loss.item()
                    best_costs = test_costs
                    utils.makedirs(args.save)
                    best_params = net.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict': best_params,
                    }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m)))
                    net.train()

        logger.info(log_message)

        if itr % args.logrho0_freq == 0:
            x0, rho_x0, d_net = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=True, device=device)
            x0 = cvt(torch.from_numpy(x0))
            rho_x0 = cvt(torch.from_numpy(rho_x0))

        # create plots
        if itr % args.viz_freq == 0:
            nSamples = 20000
            p_samples = toy_data.inf_train_gen(args.data, batch_size=nSamples, require_density=False,device=device)
            p_samples = cvt(torch.from_numpy(p_samples))
            y = cvt(torch.randn(nSamples, d))
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                # plot1d_BD(net, p_samples, y, nt_val, sPath, doPaths=False, sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                #             ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1], alph[2], nt, m, nTh),alphaa=args.alphaa)
                plot2d(net, p_samples, y, nt_val, sPath, doPaths=True,
                          sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                                 ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1],
                                                                        alph[2], nt, m, nTh))
                net.load_state_dict(curr_state)
                net.train()

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p['lr'] /= args.lr_drop
            print("lr: ", p['lr'])

        # resample data
        if itr % args.sample_freq == 0:
            # resample data [nSamples, d+1]
            logger.info("resampling")
            x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,require_density=False,device=device)  # load data batch
            x0 = cvt(torch.from_numpy(x0))  # convert to torch, type and gpu
            y0 = d_net(x0)
            y0 = y0.detach().cpu().numpy()
            x1 = x0.detach().cpu().numpy()
            w = np.sum(np.power(x1, 2), 1, keepdims=True)
            rho_x0 = np.exp(-w/2+y0)/((2*pi)**(d/2))
            rho_x0 = cvt(torch.from_numpy(rho_x0))

        end = time.time()

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))