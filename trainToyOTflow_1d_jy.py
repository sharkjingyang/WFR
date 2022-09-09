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
    def_viz_freq = 300
    def_batch    = 4096
    def_niter    = 1500
else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 100
    def_batch    = 2048
    def_niter    = 1000

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['1d','swissroll','2gaussians','8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='1d'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps") # origin:8
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation") # origin:8
parser.add_argument('--alph'  , type=str, default='1.0,100.0,5.0') # origin 1, 100, 5
parser.add_argument('--m'     , type=int, default=32)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=0.05)
parser.add_argument("--drop_freq"   , type=int  , default=200, help="how often to decrease learning rate") # origin 100
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='double', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='1d_plot')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
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

def compute_loss(net, x,x2, nt):


    costC1,costL1, costV1, cs, weights, position ,Phi_penalty_forward= OTFlowProblem_ex(x, net, [0, 1], nt=nt, stepper="rk4",
                                                                    alph=net.alph, is_1d=False, alphaa=args.alphaa,jjj=0,device=device)


    costC2, costL2, costV2, cs2, weights2, position2 ,Phi_penalty_inverse= OTFlowProblem_ex(x2, net, [1, 0], nt=nt, stepper="rk4",
                                                                        alph=net.alph, is_1d=False, alphaa=args.alphaa,
                                                                        jjj=1, device=device)

    #
    # position4 = position2.cpu().detach().numpy()
    # plt.plot(position4[:, 0], position4[:, 1])
    # plt.show()
    # d=2
    # genModel, _ = integrate_ex(x2[:, 0:d], net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph)
    # costC2 = genModel[0, -4] / x.shape[0]

    # a = torch.sum(genModel[:,-1])
    # b = genModel[:,-1]
    # costC2 = torch.mean(torch.log(genModel[:,-1]))
    # Jc, cs = OTFlowProblem(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph, is_1d=True)

    # costL1 = torch.sum(genModel[:,-3])
    # costV1 = torch.sum(genModel[:,-5])
    # print(costC2+costC1, costL1, costV1)
    # print('weights_min:{},weights_max:{}, weights2:{}, weights2:{}'.format(torch.min(weights),torch.max(weights),torch.min(weights2),torch.max(weights2)))


    penalty_coff=1
    penalty_Phi=penalty_coff*torch.abs(Phi_penalty_forward-Phi_penalty_inverse)
    # print("penalty_Phi is %f" % penalty_Phi)
    Jc = (costC1)*100+costL1*5+costV1*1+penalty_Phi
    # Jc = (costC1+costC2)*100+costL1*5+costV1*1
    # cs[1]=cs[1]+costC2
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

    # x0, rho_x0, d_net = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,require_density=True,device=device) ## rho_x0(tensor w.r.t x0) and rho_0 network
    # x0 = cvt(torch.from_numpy(x0))
    # rho_x0 = cvt(torch.from_numpy(rho_x0))
    # x2 = torch.randn((4096, 1)).to(device)
    # x3 = torch.randn((4096, 1)).to(device)
    # x0val, rho_x0val, d_net2 = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size,require_density=True,device=device)  
    # x0val = cvt(torch.from_numpy(x0val))
    # rho_x0val = cvt(torch.from_numpy(rho_x0val))

    x0=np.random.normal(size=(4096,1))-2+4*(np.random.uniform(size=(4096,1))>0.5)
    x0 = cvt(torch.from_numpy(x0))
    x2 = torch.randn((4096, 1)).to(device)
    x3 = torch.randn((4096, 1)).to(device)
    x0val=np.random.normal(size=(4096,1))-2+4*(np.random.uniform(size=(4096,1))>0.5)
    x0val = cvt(torch.from_numpy(x0val))


    



    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s} {:9s}     {:9s}  {:9s}  {:9s}  {:9s} {:9s} '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'V(velocity)', 'valLoss', 'valL', 'valC', 'valR',
            'valV'
        )
    )
    logger.info(log_msg)
    time_meter = utils.AverageMeter()
    net.train()

    record_loss_totol=[]
    record_loss_L2=[]

    for itr in range(1, args.niters + 1):
        # train
        optim.zero_grad()
        loss, costs, weights, position = compute_loss(net, x0, x2, nt=args.nt)

        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optim.step()

        # position3 = position.cpu().detach().numpy()
        # plt.plot(position3[:, 0], position3[:, 1])
        # plt.show()

        # if itr==1 or itr%100==0:
        #     a = torch.linspace(-4, 4, 1000)
        #     uu = []
        #     for i in range(1000):
        #         uu.append(torch.cat((a[i] * torch.ones_like(a).unsqueeze(-1), a.unsqueeze(-1)), dim=-1))
        #     u = torch.cat(uu).to(device)
        #     t = 0
        #     for i in range(9):
        #         sPath2 = os.path.join(args.save, 'figs2', start_time + '_{}.png'.format(t))
        #         if not os.path.exists(os.path.dirname(sPath2)):
        #             os.makedirs(os.path.dirname(sPath2))
        #         z = pad(u, (0, 1, 0, 0), value=t)
        #         q = net(z).reshape(1000, 1000)
        #         figure = plt.figure()
        #         axes = Axes3D(figure)
        #         XX = np.linspace(-4, 4, 1000)
        #         TT = np.linspace(-4,4,1000)
        #         XX, TT = np.meshgrid(XX, TT)
        #         plt.xlabel("X")
        #         plt.ylabel("T")
        #         plt.title("itr:{}, t:{}".format(0, t))
        #         q = q.cpu().detach().numpy()
        #         axes.plot_surface(XX, TT, q, cmap='rainbow')
        #         plt.savefig(sPath2, dpi=300)
        #         t = t + 0.125

        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}    {:.4f}   {:.4f}    {:.4f}      {:.4f}  '.format(  #{:9.3e}  {:9.3e}  {:9.3e}{:9.3e}
                itr, time_meter.val , loss, costs[0], costs[1], costs[2]
            )
        ) 

        # validate
        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                net.eval()
                test_loss, test_costs, test_weights, test_position = compute_loss(net, x0val, x3,  nt=nt_val)


                ##record val loss------------
                record_loss_totol.append(test_loss.cpu().numpy())
                record_loss_L2.append(test_costs[0].cpu().numpy())

            

                #--------------------

                # add to print message
                log_message += '   {:.4f}    {:.4f}    {:.4f}      {:.4f}  '.format(  #{:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}
                    test_loss, test_costs[0], test_costs[1], test_costs[2]
                )

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    best_costs = test_costs
                    utils.makedirs(args.save)
                    best_params = net.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict': best_params,
                    }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m)))
                    net.train()

        logger.info(log_message)

        # if itr % args.logrho0_freq == 0:
        #     x0, rho_x0, d_net = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, require_density=True, device=device)
        #     x0 = cvt(torch.from_numpy(x0))
        #     rho_x0 = cvt(torch.from_numpy(rho_x0))

        # create plots
        if itr % args.viz_freq == 0:
            nSamples = 20000
            # p_samples = toy_data.inf_train_gen(args.data, batch_size=nSamples, require_density=False,device=device)
            p_samples=np.random.normal(size=(nSamples,1))-2+4*(np.random.uniform(size=(nSamples,1))>0.5)
            p_samples = cvt(torch.from_numpy(p_samples))
            y = cvt(torch.randn(nSamples, d))
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                sPath2 = os.path.join(args.save, 'figs(2)', start_time + '_{:04d}.png'.format(itr))

                # plot1d_BD(net, p_samples, y, nt_val, sPath, doPaths=False, sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                #             ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1], alph[2], nt, m, nTh),alphaa=args.alphaa)
                # plot1d(net, p_samples, y, nt_val, sPath, doPaths=True,
                #           sTitle='{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  '
                #                  ' nt {:d}   m {:d}  nTh {:d}  '.format(args.data, best_loss, best_costs[1], alph[1],
                #                                                         alph[2], nt, m, nTh))
                
                plot1d(net, p_samples, y, nt_val, sPath,sPath2, doPaths=False,
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
            # resample data [nSamples, d+1]
            logger.info("resampling")
            x0=np.random.normal(size=(4096,1))-2+4*(np.random.uniform(size=(4096,1))>0.5)
            x0 = cvt(torch.from_numpy(x0))



            # x0 = toy_data.inf_train_gen(args.data, batch_size=args.batch_size,require_density=False,device=device)  # load data batch
            # x0 = cvt(torch.from_numpy(x0))  # convert to torch, type and gpu
            # y0 = d_net(x0)
            # y0 = y0.detach().cpu().numpy()
            # x1 = x0.detach().cpu().numpy()
            # w = np.sum(np.power(x1, 2), 1, keepdims=True)
            # rho_x0 = np.exp(-w/2+y0)/((2*pi)**(d/2))
            # rho_x0 = cvt(torch.from_numpy(rho_x0))
            x2 = torch.randn((4096, 1)).to(device)

        end = time.time()

    # np.save("experiments/cnf/toy/figs_penalty_8gaussians/total_loss.npy",np.array(record_loss_totol))
    # np.save("experiments/cnf/toy/figs_penalty_8gaussians/L2_loss.npy",np.array(record_loss_L2))

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))