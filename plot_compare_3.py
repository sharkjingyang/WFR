# trainToyOTflow.py
# training driver for the two-dimensional toy problems
import argparse
import os
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters
from src.plotter import plot1d
from src.OTFlowProblem import *
from source_without_BD import *
from src.mmd_new import *

import config

cf = config.getconfig()

def_viz_freq = 500
def_batch = 100000
def_niter = 1000

parser = argparse.ArgumentParser('UOT-Flow')
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
parser.add_argument('--val_freq', type=int, default=5000)
parser.add_argument('--gpu'     , type=int, default=1)
parser.add_argument('--sample_freq', type=int, default=5000)
parser.add_argument('--alphaa', type=float, default=5)

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


def compute_loss_1(net, x,x2, nt):

    costC1, costL1, costV1, cs, weights, aaaa1 = OTFlowProblem_1d(x, net, [0,1], nt=nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=args.alphaa,jjj=0, device=device)
    costC2, costL2, costV2, cs2, weights2, aaaa2 = OTFlowProblem_1d(x2, net, [1,0], nt=nt, stepper="rk4", alph=[1.0, 1.0, 1.0], alphaa=args.alphaa,jjj=1, device=device)

    Jc = (costC1+costC2) * 100 + costL1 * 1 + costV1 * 1+torch.abs(aaaa1+aaaa2)
    print(aaaa1,aaaa2)
    cs[1] = cs[1] + cs2[1]
    return Jc, cs, weights, 0


def f(x):
    return x**2

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
    net_2 = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net_2 = net_2.to(prec).to(device)
    # net_5 = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    # net_5 = net_5.to(prec).to(device)
    net_10 = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net_10 = net_10.to(prec).to(device)
    net_ot = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net_ot = net_ot.to(prec).to(device)

    # optim1 = torch.optim.Adam(net_inverse.parameters(), lr=args.lr, weight_decay=args.weight_decay ) # lr=0.04 good
    # optim2 = torch.optim.Adam(net_ot.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(net_2)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    logger.info("nt={:}   nt_val={:}".format(nt, nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net_2)))
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

    # net_inverse.train()
    # net_ot.train()
    MMD_list = []
    ITR = []
    Time = []
    MSE=[]
    DEVI=[]
    t = 0

    path0 = os.path.join(args.save, 'figs4', 'ot_flow.pth')
    net0 = torch.load(path0)
    net_ot.load_state_dict(net0['state_dict'])

    path2 = os.path.join(args.save, 'figs4','alpha_2.pth')
    net2 = torch.load(path2)
    net_2.load_state_dict(net2['state_dict'])

    # path5 = os.path.join(args.save, 'figs4', 'alpha_5.pth')
    # net5 = torch.load(path5)
    # net_5.load_state_dict(net5['state_dict'])

    path10 = os.path.join(args.save, 'figs4', 'alpha_10.pth')
    net10 = torch.load(path10)
    net_10.load_state_dict(net10['state_dict'])



    with torch.no_grad():
        # net_ot.eval()
        # net_inverse.eval()

        nSamples = 100000
        x, _ = cvt(torch.Tensor(toy_data.inf_train_gen(args.data, batch_size=nSamples, require_density=False) ))
        y = cvt(torch.randn(nSamples, d))

        w = [1 / sqrt(2 * pi) * exp(-x ** 2 / 2) for x in np.linspace(-10, 10, 1000)]
        w_0 = [density(x, centers=[-3, 3, 3], weights=[1 / 3, 1 / 3, 1 / 3]) for x in np.linspace(-10, 10, 1000)]

        sPath = os.path.join(args.save, 'figs7', 'backward_compare_{}.png'.format(nSamples))
        fig, axs = plt.subplots(5, 5)
        fig.set_size_inches(25, 25)
        nBins=33
        doPaths = 1
        if doPaths:
            # forwPath2, phi2 = integrate_ex(x.unsqueeze(-1)[:, 0:d], net_2, [0.0, 1.0], nt_val, stepper="rk4", alph=net_2.alph,
            #                              intermediates=True, alphaa=2)
            # forwPath2 = forwPath2.detach().cpu().numpy()
            # phi2 = phi2.detach().cpu().numpy()

            # forwPath5, phi5 = integrate_ex(x.unsqueeze(-1)[:, 0:d], net_5, [0.0, 1.0], nt_val, stepper="rk4",
            #                                alph=net_5.alph,
            #                                intermediates=True, alphaa=5)
            # forwPath5 = forwPath5.detach().cpu().numpy()
            # phi5 = phi5.detach().cpu().numpy()

            # forwPath10, phi10 = integrate_ex(x.unsqueeze(-1)[:, 0:d], net_10, [0.0, 1.0], nt_val, stepper="rk4",
            #                                alph=net_10.alph,
            #                                intermediates=True, alphaa=10)
            # forwPath10 = forwPath10.detach().cpu().numpy()
            # phi10 = phi10.detach().cpu().numpy()

            backPath2, phi_inv2 = integrate_ex(y[:, 0:d], net_2, [1.0, 0.0], nt_val, stepper="rk4", alph=net_2.alph,
                                               intermediates=True, alphaa=2)
            backPath2 = backPath2.detach().cpu().numpy()
            phi_inv2 = phi_inv2.detach().cpu().numpy()
            #
            # backPath5, phi_inv5 = integrate_ex(y[:, 0:d], net_5, [1.0, 0.0], nt_val, stepper="rk4", alph=net_2.alph,
            #                                    intermediates=True, alphaa=5)
            # backPath5 = backPath5.detach().cpu().numpy()
            # phi_inv5 = phi_inv5.detach().cpu().numpy()
            #
            backPath10, phi_inv10 = integrate_ex(y[:, 0:d], net_10, [1.0, 0.0], nt_val, stepper="rk4", alph=net_2.alph,
                                               intermediates=True, alphaa=10)
            backPath10 = backPath10.detach().cpu().numpy()
            phi_inv10= phi_inv10.detach().cpu().numpy()

            # forwPath_ot = integrate(x.unsqueeze(-1), net_ot, [0.0, 1.0], nt_val, stepper="rk4", alph=net_ot.alph,
            #                      intermediates=True)
            # forwPath_ot = forwPath_ot.detach().cpu().numpy()

            backPath_ot = integrate(y[:, 0:d], net_ot, [1.0, 0.0], nt_val, stepper="rk4", alph=net_ot.alph, intermediates=True)
            backPath_ot = backPath_ot.detach().cpu().numpy()



            font_dict = {'weight': 'semibold', 'family':'Times New Roman','size': 20}
            # weight choices: light, normal, medium, semibold, bold, heavy, black
            font_dict = {'weight': 'semibold', 'size': 25}
            for i in range(5):
                print(i)
                n0,b0,patches0=axs[0, i].hist(backPath_ot[:, 0, 2*i], density=True, color='white', bins=nBins)
                axs[0, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                axs[0, i].set_title('t={}'.format(1-0.25 * i), fontdict=font_dict)
                axs[0, i].set_xlim(-5, 5)
                axs[0, i].set_ylim(0, 0.43)
                axs[0, i].set_yticks([0,0.1,0.2,0.3,0.4])
                axs[0, i].set_yticklabels([0,0.1,0.2,0.3,0.4], fontsize=16)
                axs[0, i].set_xticks(ticks=[])
                mid_points = []
                heights = []
                for j in range(len(patches0)):
                    mid_points.append((b0[j] + b0[j + 1]) / 2)
                    heights.append(patches0[j].get_height())
                axs[0, i].plot(mid_points, heights, 'b', linewidth=5)

                n1,b1,patches1=axs[1, i].hist(backPath10[:, 0, 2 * i], density=True,color='white',
                               bins=nBins)
                axs[1, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                axs[1, i].set_xlim(-5, 5)
                axs[1, i].set_ylim(0, 0.43)
                axs[1, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                axs[1, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                axs[1, i].set_xticks(ticks=[])
                mid_points=[]
                heights=[]
                for j in range(len(patches1)):
                    mid_points.append((b1[j]+b1[j+1])/2)
                    heights.append(patches1[j].get_height())
                axs[1, i].plot(mid_points, heights,'b', linewidth=5)

                n2, b2, patches2=axs[2, i].hist(backPath10[:, 0, 2 * i], density=True,color='white', weights=backPath10[:, -1, 2*i],
                               bins=nBins)
                axs[2, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                axs[2, i].set_xlim(-5, 5)
                axs[2, i].set_ylim(0, 0.43)
                axs[2, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                axs[2, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                axs[2, i].set_xticks(ticks=[])
                mid_points = []
                heights = []
                for j in range(len(patches2)):
                    mid_points.append((b2[j] + b2[j + 1]) / 2)
                    heights.append(patches2[j].get_height())
                axs[2, i].plot(mid_points, heights,'b', linewidth=5)





                # axs[3, i].hist(forwPath5[:, 0, 2 * i], density=True,
                #                bins=nBins)
                # axs[3, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                # axs[3, i].set_xlim(-5, 5)
                # axs[3, i].set_ylim(0, 0.43)
                # axs[3, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                # axs[3, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                # axs[3, i].set_xticks(ticks=[])
                #
                # axs[4, i].hist(forwPath5[:, 0, 2 * i], density=True, weights=forwPath5[:, -1, 2 * i],
                #                bins=nBins)
                # axs[4, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                # axs[4, i].set_xlim(-5, 5)
                # axs[4, i].set_ylim(0, 0.43)
                # axs[4, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                # axs[4, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                # axs[4, i].set_xticks(ticks=[])


                n3, b3, patches3=axs[3, i].hist(backPath2[:, 0, 2 * i], density=True,color='white',
                               bins=nBins)
                axs[3, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                axs[3, i].set_xlim(-5, 5)
                axs[3, i].set_ylim(0, 0.43)
                axs[3, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                axs[3, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                axs[3, i].set_xticks(ticks=[])
                mid_points = []
                heights = []
                for j in range(len(patches3)):
                    mid_points.append((b3[j] + b3[j + 1]) / 2)
                    heights.append(patches3[j].get_height())
                axs[3, i].plot(mid_points, heights,'b', linewidth=5)

                n4, b4, patches4 =axs[4, i].hist(backPath2[:, 0, 2 * i], density=True,color='white',weights=backPath2[:, -1, 2 * i],
                               bins=nBins)
                axs[4, i].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
                axs[4, i].set_xlim(-5, 5)
                axs[4, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                axs[4, i].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
                axs[4, i].set_xticks([-4, -2, 0, 2, 4])
                axs[4, i].set_xticklabels([-4, -2, 0, 2, 4], fontsize=16)
                axs[4, i].set_ylim(0, 0.43)
                mid_points = []
                heights = []
                for j in range(len(patches4)):
                    mid_points.append((b4[j] + b4[j + 1]) / 2)
                    heights.append(patches4[j].get_height())
                axs[4, i].plot(mid_points, heights,'b', linewidth=5)


                if i>=1:
                    axs[0, i].set_yticks(ticks=[])
                    axs[1, i].set_yticks(ticks=[])
                    axs[2, i].set_yticks(ticks=[])
                    axs[3, i].set_yticks(ticks=[])
                    axs[4, i].set_yticks(ticks=[])
                    # axs[5, i].set_yticks(ticks=[])
                    # axs[6, i].set_yticks(ticks=[])

            axs[0, 0].set_ylabel('OT-Flow',fontdict=font_dict)
            axs[1, 0].set_ylabel('UOT '+ chr(945)+'=10(unweighted)',fontdict=font_dict)
            axs[2, 0].set_ylabel('UOT '+ chr(945)+'=10',fontdict=font_dict)
            # axs[3, 0].set_ylabel('UOT '+ chr(945)+'=5(unweighted)', fontdict=font_dict)
            # axs[4, 0].set_ylabel('UOT '+ chr(945)+'=5', fontdict=font_dict)
            axs[3, 0].set_ylabel('UOT '+ chr(945)+'=2(unweighted)', fontdict=font_dict)
            axs[4, 0].set_ylabel('UOT '+ chr(945)+'=2', fontdict=font_dict)

            if not os.path.exists(os.path.dirname(sPath)):
                os.makedirs(os.path.dirname(sPath))
            plt.savefig(sPath, dpi=300)
            plt.close()



            # # plot weight
            # sPath_w = os.path.join(args.save, 'figs7', 'weight_1.png'.format(nSamples))
            # fig, axs = plt.subplots(2, 3)
            # fig.set_size_inches(21, 14)
            #
            # for j in range(3):
            #     if j==0:
            #         forwPath=forwPath10
            #         axs[0, j].set_title(chr(945) + ' = 10', fontdict=font_dict)
            #     elif j==1:
            #         forwPath = forwPath5
            #         axs[0, j].set_title(chr(945) + ' = 5', fontdict=font_dict)
            #     else:
            #         forwPath = forwPath2
            #         axs[0, j].set_title(chr(945) + ' = 2', fontdict=font_dict)
            #
            #     for i in range(nt_val + 1):
            #         tt1 = np.sort(forwPath[:, 0, i].squeeze())[::1]
            #         idx = np.argsort(forwPath[:, 0, i].squeeze())[::1]
            #         tt2 = forwPath[:, -1, i].squeeze()
            #         tt2 = tt2[idx]
            #         if i % 2 == 0:
            #             axs[0, j].scatter(tt1, tt2, s=0.1, label='t={}'.format(i / nt_val))
            #         else:
            #             axs[0, j].scatter(tt1, tt2, s=0.1)
            #
            #
            #     x_numpy = x.detach().cpu().numpy().squeeze()
            #     for i in range(nt_val + 1):
            #         tt1 = np.sort(x_numpy)[::1]
            #         idx = np.argsort(x_numpy)[::1]
            #         tt2 = forwPath[:, -1, i].squeeze()
            #         tt2 = tt2[idx]
            #         if i % 2 == 0:
            #             axs[1, j].scatter(tt1, tt2, s=0.5, label='t={}'.format(i / nt_val))
            #         else:
            #             axs[1, j].scatter(tt1, tt2, s=0.5)
            #
            #     axs[0, j].legend(loc='upper left',fontsize=16)
            #     axs[1, j].legend(loc='upper left',fontsize=16)
            #
            #     axs[0, j].set_xlim(-6.5, 6.5)
            #     axs[0, j].set_xticks([-6, -4, -2, 0,2, 4, 6])
            #     axs[0, j].set_xticklabels([-6, -4, -2, 0,2, 4, 6], fontsize=16)
            #
            #     # axs[1, j].set_xlim(-6.5, 6.5)
            #
            #     axs[0, j].set_ylim(0, 3.7)
            #     axs[0, j].set_yticks([0, 0.5, 1.0,1.5, 2.0,2.5, 3.0,3.5])
            #     axs[0, j].set_yticklabels([0, 0.5, 1.0,1.5, 2.0,2.5, 3.0,3.5], fontsize=16)
            #
            #     axs[1, j].set_xlim(-6.5, 6.5)
            #     axs[1, j].set_xticks([-6, -4, -2, 0,2, 4, 6])
            #     axs[1, j].set_xticklabels([-6, -4, -2, 0,2, 4, 6], fontsize=16)
            #
            #     axs[1, j].set_ylim(0, 3.7)
            #     axs[1, j].set_yticks([0, 0.5, 1.0,1.5, 2.0,2.5, 3.0,3.5])
            #     axs[1, j].set_yticklabels([0, 0.5, 1.0,1.5, 2.0,2.5, 3.0,3.5], fontsize=16)
            #
            #
            # if not os.path.exists(os.path.dirname(sPath_w)):
            #     os.makedirs(os.path.dirname(sPath_w))
            # plt.savefig(sPath_w, dpi=300)
            # plt.close()

            # fig, axs = plt.subplots(1, 1)
            # fig.set_size_inches(12, 12)
            # sPath_path_1 = os.path.join(args.save, 'figs7', 'forward_path.png'.format(nSamples))
            # nPts = 50
            # pts = np.unique(np.random.randint(nSamples, size=nPts))
            #
            # for pt in pts:
            #     axs.plot(backPath2[pt, 0, :], np.linspace(0,1,nt_val+1), color='green', linewidth=1, label='trajectory')
            #
            # axs.set_xlim(-6, 6)
            # axs.set_ylim(-0.05, 1.05)
            # axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # axs.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            # axs.set_xticks([-6, -4, -2, 0, 2, 4, 6])
            # axs.set_xticklabels([-6, -4, -2, 0, 2, 4, 6], fontsize=16)
            # axs.set_ylabel('t',fontsize=20)
            # axs.set_xlabel('position',fontsize=20)


            # if not os.path.exists(os.path.dirname(sPath_path_1)):
            #     os.makedirs(os.path.dirname(sPath_path_1))
            # plt.savefig(sPath_path_1, dpi=300)
            # plt.close()



