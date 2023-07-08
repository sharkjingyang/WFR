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
from src.plotter import plot4, plot1d, plot1d_BD

# from src.OTFlowProblem import *
# from src.birth_death_process import *
from source_without_BD_WGAN import *
from WGAN_models import *
import config
import matplotlib
cf = config.getconfig()

if cf.gpu: # if gpu on platform
    def_viz_freq = 100
    def_batch    = 4096
    def_niter    = 1500
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
parser.add_argument('--alph'  , type=str, default='1.0,0.0,10') # origin 1, 100, 5
parser.add_argument('--m'     , type=int, default=32)
parser.add_argument('--nTh'   , type=int, default=2)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--lrD', type=float, default=0.005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda0'   , type=float, default=0.1)
parser.add_argument('--lambda1'   , type=float, default=10)
parser.add_argument('--lambda2'   , type=float, default=10)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lrG'          , type=float, default=0.05) # origin without GAN: 0.1
parser.add_argument("--drop_freq"   , type=int  , default=100, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim1'       , type=str  , default='adam', choices=['adam'])
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


def compute_loss(net, x, rho_x, nt, t0=0, t1=1,alphaa=args.alphaa):
    # Jc , cs = OTFlowProblem_BD(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True,alphaa=args.alphaa)
    Jc, cs, weights, position = OTFlowProblem_ex(x, rho_x, net, [t0, t1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True, alphaa=alphaa)
    # Jc, cs = OTFlowProblem(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True)
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
    netD = W1_D(d=d, ndf=args.ndf)
    netD = netD.to(prec).to(device)
    netDD = W1_D(d=d, ndf=args.ndf)
    netDD = netDD.to(prec).to(device)

    optimG = torch.optim.Adam(net.parameters(), lr=args.lrG, weight_decay=args.weight_decay) # lr=0.04 good
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lrD, betas=[0.0, 0.9])
    optimDD = torch.optim.Adam(netDD.parameters(), lr=args.lrD, betas=[0.0, 0.9])

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optimG)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    end = time.time()
    best_loss = float('inf')
    bestParams = None


    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}     {:9s}  {:9s}   {:9s} {:9s} {:9s}{:9s}'.format(
            'iter', ' time','lossG', 'L (L_2)', 'R (HJB)', 'V(velocity)','W1', 'valLoss', 'valL', 'valR', 'valV', 'valW1'
        )
    )

    logger.info(log_msg)

    time_meter = utils.AverageMeter()


    for itr in range(1, args.niters + 1):
        # if itr < 25 or itr % args.sample_freq == 0:
        #     Diters = 1
        # else:
        #     Diters = 1  # default: 5
        Diters = 1


        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in netDD.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in net.parameters():
            p.requires_grad = False

        i=0

        while i < Diters:
            i = i+1

            x0, rho_x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
            x0 = cvt(torch.from_numpy(x0))
            rho_x0 = cvt(torch.from_numpy(rho_x0))
            y = cvt(torch.randn(x0.shape[0], d))

            with torch.no_grad():
                loss, costs, weights, position = compute_loss(net, x0, rho_x0, nt=args.nt, t0=0, t1=1, alphaa=args.alphaa)
                loss_1, costs_1, weights_1, position_1 = compute_loss(net, y.squeeze(), rho_x0, nt=args.nt, t0=1, t1=0, alphaa=10000)

            fx = netD(position).squeeze()
            fy = netD(y).squeeze()
            gp = compute_gradient_penalty(netD, real_samples=y, fake_samples=position, device=device)
            lossD = torch.dot(fx, weights) - torch.mean(fy)+gp*args.lambda0

            optimD.zero_grad()
            lossD.backward(retain_graph=True)
            optimD.step()


            fx2 = netDD(position_1).squeeze()
            fy2 = netDD(x0.unsqueeze(-1)).squeeze()
            gp2 = compute_gradient_penalty(netDD, real_samples=x0.unsqueeze(-1), fake_samples=position_1, device=device)
            lossDD = torch.dot(fx2, weights_1)-torch.mean(fy2)+gp2*args.lambda0
            optimDD.zero_grad()
            lossDD.backward(retain_graph=True)
            optimDD.step()


        for p in netD.parameters():
            p.requires_grad = False
        for p in netDD.parameters():
            p.requires_grad = False
        for p in net.parameters():
            p.requires_grad = True


        x0, rho_x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=False)
        x0 = cvt(torch.from_numpy(x0))
        rho_x0 = cvt(torch.from_numpy(rho_x0))
        y = cvt(torch.randn(x0.shape[0], d))
        loss, costs, weights, position = compute_loss(net, x0, rho_x0, nt=args.nt, t0=0, t1=1, alphaa=args.alphaa)
        loss_1, costs_1, weights_1, position_1 = compute_loss(net, y.squeeze(), rho_x0, nt=args.nt, t0=1, t1=0, alphaa=10000)

        fx1 = netD(position).squeeze()
        fx3 = netDD(position_1).squeeze()
        loss1 = -torch.dot(fx1, weights)
        loss2 = -torch.dot(fx3, weights_1)
        lossG = loss + torch.abs(loss_1) + loss1 * args.lambda1 + loss2 * args.lambda1


        optimG.zero_grad()
        lossG.backward()
        optimG.step()

        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e}'.format(
                itr, time_meter.val , lossG, costs[0], costs[1], costs[2], loss2
            )
        )


        if itr % args.val_freq == 0 or itr == args.niters:
            x0val, rho_x0val = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size, require_density=False)
            x0val = cvt(torch.from_numpy(x0val))
            rho_x0val = cvt(torch.from_numpy(rho_x0val))
            y = cvt(torch.randn(x0.shape[0], d))
            with torch.no_grad():
                net.eval()
                netD.eval()
                test_loss1, test_costs, test_weights, test_position = compute_loss(net, x0val, rho_x0val, nt=nt_val, alphaa=args.alphaa)
                test_loss2, test_costs2, test_weights2, test_position2 = compute_loss(net, y.squeeze(), rho_x0val, nt=nt_val,
                                                                                  alphaa=10000)
                fx1 = netD(test_position).squeeze()
                fx2 = netDD(test_position2).squeeze()
                test_loss_1 = -torch.dot(fx1, test_weights)
                test_loss_2 = -torch.dot(fx2, test_weights2)
                test_loss = test_loss1+torch.abs(test_loss2)+test_loss_1*args.lambda1+test_loss_2*args.lambda1
                # add to print message
                log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} '.format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2], test_loss_2
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

        logger.info(log_message)  # print iteration

        # create plots
        if itr % args.viz_freq == 0:
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                nSamples = 200000
                p_samples, _ = cvt(torch.Tensor(toy_data.inf_train_gen(args.data, batch_size=nSamples,require_density=False) ))
                y = cvt(torch.randn(nSamples,d)) # sampling from the standard normal (rho_1)

                sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                # plot1d_BD(net, p_samples, y, nt_val, sPath, doPaths=False, sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                #             ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1], alph[2], nt, m, nTh),alphaa=args.alphaa)
                plot1d(net, p_samples, y, nt_val, sPath, doPaths=True,
                          sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                                 ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1],
                                                                        alph[2], nt, m, nTh), alphaa=args.alphaa)
                net.load_state_dict(curr_state)
                net.train()

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optimG.param_groups:
                p['lr'] /= args.lr_drop

        # resample data
        if itr % args.sample_freq == 0:
            # resample data [nSamples, d+1]
            logger.info("resampling")
            x0, rho_x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,require_density=False)  # load data batch
            x0 = cvt(torch.from_numpy(x0))  # convert to torch, type and gpu
            rho_x0 = cvt(torch.from_numpy(rho_x0))

        end = time.time()

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))