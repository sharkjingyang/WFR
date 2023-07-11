# plotter.py
import matplotlib
import torch

try:
    matplotlib.use('TkAgg') # origin : 'TkAgg'
except:
    matplotlib.use('Agg') # for linux server with no tkinter
matplotlib.use('Agg') # assume no tkinter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
from src.OTFlowProblem import *
# from src.birth_death_process import *
# from source_without_BD_WGAN_2d import *
from source_without_BD import *
import numpy as np
import os
import h5py
import datasets
from torch.nn.functional import pad
from matplotlib import colors # for evaluateLarge
from lib.toy_data import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['image.cmap'] = 'inferno'

def plot4(net, x, y, nt_val, sPath, sTitle="", doPaths=False):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]


    fx = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 70
    LOWX  = -4
    HIGHX = 4
    LOWY  = -4
    HIGHY = 4

    if d > 50: # assuming bsds
        # plot dimensions d1 vs d2 
        d1=0
        d2=1
        LOWX  = -0.15   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.15
        LOWY  = -0.15
        HIGHY = 0.15
    if d > 700: # assuming MNIST
        d1=0
        d2=1
        LOWX  = -10   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 10
        LOWY  = -10
        HIGHY = 10
    elif d==8: # assuming gas
        LOWX  = -0.4   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.4
        LOWY  = -0.4
        HIGHY = 0.4 
        d1=2
        d2=3
        nBins = 100
    elif d==56: # SVGD_data
        LOWX  = -3
        HIGHX = 3
        LOWY  = -3
        HIGHY = 3
        d1=18
        d2=17
        nBins = 70
    else:
        d1=0
        d2=1

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)
    fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    # hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
    im1 , _, _, map1 = axs[0, 0].hist2d(x.detach().cpu().numpy()[:, d1], x.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[0, 0].set_title('x from rho_0')
    im2 , _, _, map2 = axs[0, 1].hist2d(fx.detach().cpu().numpy()[:, d1], fx.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[0, 1].set_title('f(x)')
    im3 , _, _, map3 = axs[1, 0].hist2d(finvfx.detach().cpu().numpy()[: ,d1] ,finvfx.detach().cpu().numpy()[: ,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 0].set_title('finv( f(x) )')
    im4 , _, _, map4 = axs[1, 1].hist2d(genModel.detach().cpu().numpy()[:, d1], genModel.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 1].set_title('finv( y from rho1 )')


    fig.colorbar(map1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(map2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(map3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(map4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )


    # plot paths
    if doPaths:
        forwPath = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        backPath = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)

        # plot the forward and inverse trajectories of several points; white is forward, red is inverse
        nPts = 30
        pts = np.unique(np.random.randint(nSamples, size=nPts))
        for pt in pts:
            axs[0, 0].plot(forwPath[pt, 0, :].detach().cpu().numpy(), forwPath[pt, 1, :].detach().cpu().numpy(), color='white', linewidth=4)
            axs[0, 0].plot(backPath[pt, 0, :].detach().cpu().numpy(), backPath[pt, 1, :].detach().cpu().numpy(), color='red', linewidth=2)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # axs[i, j].get_yaxis().set_visible(False)
            # axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
        
    new_path="C:/Users/shark/桌面/WFR-main/high_dim_Bayes/fig_train_immediate.png"
    # plt.savefig(sPath, dpi=300)
    plt.savefig(new_path, dpi=300)
    plt.close()



def plot2d(net, x, y, nt_val, sPath, sTitle="", doPaths=False):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]


    fx,_ = integrate_ex(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    finvfx,_ = integrate_ex(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel,_ = integrate_ex(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 33


    LOWX  = -4
    HIGHX = 4
    LOWY  = -4
    HIGHY = 4

    if d > 50: # assuming bsds
        # plot dimensions d1 vs d2
        d1=0
        d2=1
        LOWX  = -0.15   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.15
        LOWY  = -0.15
        HIGHY = 0.15
    if d > 700: # assuming MNIST
        d1=0
        d2=1
        LOWX  = -10   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 10
        LOWY  = -10
        HIGHY = 10
    elif d==8: # assuming gas
        LOWX  = -2   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX =  2
        LOWY  = -2
        HIGHY =  2
        d1=2
        d2=3
        nBins = 100
    else:
        d1=0
        d2=1

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    # fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    fx1 = fx.detach().cpu().numpy()
    genModel1 = genModel.detach().cpu().numpy()
    x_numpy = x.detach().cpu().numpy().squeeze()
    # all_points1 = np.zeros((1, 2))
    # all_points2 = np.zeros((1, 2))

    # fx1[:, -1] = (np.exp(fx1[:, -4] + fx1[:, -2]))/np.sum(np.exp(fx1[:, -4] + fx1[:, -2]))*x.shape[0]
    # genModel1[:,-1]=np.exp(genModel1[:,-4]+genModel1[:,-2])/np.sum(np.exp(genModel1[:,-4]+genModel1[:,-2]))*x.shape[0]

    # print('fx1', fx1[:,-1], min(fx1[:,-1]),max(fx1[:,-1]), sum(fx1[:,-1]))
    # print('genModel1', genModel1[:,-1], min(genModel1[:,-1]),max(genModel1[:,-1]),sum(genModel1[:,-1]))

    # w1 = w2 = 20
    # for i in range(fx1.shape[0]):
    #     all_points1 = np.concatenate((all_points1, np.ones((int(fx1[:, -1][i]*w1), 2))*fx1[:, :2][i]), axis=0)
    #     all_points2 = np.concatenate((all_points2, np.ones((int(genModel1[:, -1][i] * w2), 2)) * genModel1[:, :2][i]), axis=0)
    # all_points1 = all_points1[1:]
    # all_points2 = all_points2[1:]
    # hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
    im1 , _, _, map1 = axs[0, 0].hist2d(x.detach().cpu().numpy()[:, d1], x.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[0, 0].set_title('x from rho_0')
    # im2 , _, _, map2 = axs[0, 1].hist2d(all_points1[:, d1], all_points1[:, d2], range=[[-4, 4], [-4, 4]], bins = nBins)
    im2, _, _, map2 = axs[0, 1].hist2d(fx1[:, d1], fx1[:, d2],weights= fx1[:,-1], range=[[-4, 4], [-4, 4]], bins=nBins)
    axs[0, 1].set_title('f(x)')
    im3 , _, _, map3 = axs[1, 0].hist2d(finvfx[:, :2].detach().cpu().numpy()[:, d1], finvfx[:, :2].detach().cpu().numpy()[: ,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 0].set_title('finv( f(x) )')
    im4 , _, _, map4 = axs[1, 1].hist2d(genModel1[:, d1], genModel1[:, d2], weights=genModel1[:,-1],range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 1].set_title('finv( y from rho1 )')

    # fig.colorbar(map1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    # fig.colorbar(map2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    # fig.colorbar(map3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    # fig.colorbar(map4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )


    # plot paths
    if doPaths:
        forwPath,_ = integrate_ex(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        backPath,_ = integrate_ex(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)

        # plot the forward and inverse trajectories of several points; white is forward, red is inverse
        nPts = 10
        pts = np.unique(np.random.randint(nSamples, size=nPts))
        for pt in pts:
            axs[0, 0].plot(forwPath[pt, 0, :].detach().cpu().numpy(), forwPath[pt, 1, :].detach().cpu().numpy(), color='white', linewidth=4)
            axs[0, 0].plot(backPath[pt, 0, :].detach().cpu().numpy(), backPath[pt, 1, :].detach().cpu().numpy(), color='red', linewidth=2)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # axs[i, j].get_yaxis().set_visible(False)
            # axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # a = torch.linspace(-4, 4, 1000)
    # uu = []
    # for i in range(1000):
    #     uu.append(torch.cat((a[i] * torch.ones_like(b).unsqueeze(-1), b.unsqueeze(-1)), dim=-1))
    # u = torch.cat(uu).to(device)
    # t = 0
    # for i in range(9):
    #     z = pad(u, (0, 1, 0, 0), value=t)
    #     q = Phi(z).reshape(1000,1000)
    #     figure = plt.figure()
    #     axes = Axes3D(figure)
    #     XX = a
    #     TT = a
    #     XX, TT = np.meshgrid(XX, TT)
    #     plt.xlabel("X")
    #     plt.ylabel("T")
    #     plt.title("itr:{}, t:{}".format(0, t))
    #     q = q.cpu().detach().numpy()
    #     axes.plot_surface(XX, TT, q, cmap='rainbow')
    #     plt.show()
    #     t = t+0.125

def plot1d(net, x, y, nt_val, sPath, sPath2, sPath3, sPath4, sPath5, sTitle="", doPaths=False, alphaa=1):
    """
    only used to plot toy 1-dimension
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    # if d != 1:
    #     print("Error dimension")
    #     return -1
    nSamples = x.shape[0]

    # X_data=np.random.normal(size=(10000,1))-3+6*(np.random.uniform(size=(10000,1))>(1/3))
    # w_data=torch.ones(X_data.shape).to(device).squeeze().float()

    # fx,phi= integrate_ex(torch.from_numpy(X_data).to(device).squeeze().float().reshape(-1,1), net, [0.0, 1], nt_val, stepper="rk4", alph=net.alph, alphaa=alphaa,intermediates=True)
    fx,phi= integrate_ex(x, net, [0.0, 1], nt_val, stepper="rk4", alph=net.alph, alphaa=alphaa,intermediates=True)
    genModel,phi_bar= integrate_ex_copy(y[:, 0:d], net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph, alphaa=alphaa,intermediates=True)

    genModel1 = genModel.detach().cpu().numpy()
    X_p=genModel1[:,0:d,-1]
    w_p=genModel1[:,-1,-1]
    np.save("single_plot_Baysian/z_inverse_1d.npy",X_p)
    np.save("single_plot_Baysian/w_inverse_1d.npy",w_p)

    
    if d>50:
        ##high Bayes experiment
        fx1=fx.detach().cpu().numpy()
        genModel1 = genModel.detach().cpu().numpy()
        X_p=fx1[:,0:d,-1]
        w_p=fx1[:,-1,-1]
        # np.save("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/z_final.npy",X_p)
        # np.save("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/w_final.npy",w_p)
        X_p=genModel1[:,0:d,-1]
        w_p=genModel1[:,-1,-1]
        # np.save("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/z_inverse.npy",X_p)
        # np.save("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/w_inverse.npy",w_p)
        
        theta_gen_weight=np.hstack((X_p,w_p.reshape(-1,1)))
        np.save("C:/Users/shark/桌面/WFR-main/high_dim_Bayes/WFR_gen_theta_weight.npy",theta_gen_weight)





    if d==2:
        fx1=fx.detach().cpu().numpy()
        genModel1 = genModel.detach().cpu().numpy()
        X_p=fx1[:,0:d,-1]
        w_p=fx1[:,-1,-1]
        Gaussian_p=y.detach().cpu().numpy()
        data_p=x.detach().cpu().numpy()
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_final.npy",X_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_final.npy",w_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/Gaussian_samples.npy",Gaussian_p)
        X_p=genModel1[:,0:d,-1]
        w_p=genModel1[:,-1,-1]
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_inverse.npy",X_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_inverse.npy",w_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/data_samples.npy",data_p)



    #plot 1d
    if d==1:
        # nBins=100
        # fx1=fx.detach().cpu().numpy()
        # genModel1 = genModel.detach().cpu().numpy()
    
        # X_p=fx1[:,0:d,:]
        # w_p=fx1[:,-1,:]
        
        # np.save("C:/Users/shark/桌面/WFR-main/1d_plot/figs/forward_5.npy",(X_p,w_p))
        # np.save("C:/Users/shark/桌面/WFR-main/1d_plot/figs/zfull.npy",fx.cpu().numpy())


        # X_plot=genModel1[:,0]
        # w_plot=genModel1[:,-1]
        # np.save("C:/Users/shark/桌面/WFR-main/1d_plot/figs/inverse_5.npy",(X_plot,w_plot))
        # np.save("C:/Users/shark/桌面/WFR-main/1d_plot/figs/zfull_inverse.npy",genModel.cpu().numpy())


        fx1=fx.detach().cpu().numpy()
        genModel1 = genModel.detach().cpu().numpy()
        X_p=fx1[:,0:d,-1]
        w_p=fx1[:,-1,-1]
        Gaussian_p=y.detach().cpu().numpy()
        data_p=x.detach().cpu().numpy()
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_final_1d.npy",X_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_final_1d.npy",w_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/Gaussian_samples_1d.npy",Gaussian_p)
        X_p=genModel1[:,0:d,-1]
        w_p=genModel1[:,-1,-1]
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/z_inverse_1d.npy",X_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/w_inverse_1d.npy",w_p)
        np.save("C:/Users/shark/桌面/WFR-main/MMD_data_output/output/data_samples_1d.npy",data_p)





    # p=np.argsort(X_plot,axis=0)
    # X_new=np.sort(X_plot,axis=0)
    # w_new=w_1[p].squeeze()

    # n0,b,pat=plt.hist(X_new,bins=100,color="white",density=True,weights=w_new)
    # mid=[]
    # height=[]
    # for i in range(b.shape[0]-1):
    #     mid.append((b[i]+b[i+1])/2)
    #     height.append(pat[i].get_height())
    # mid=np.array(mid)
    # height=np.array(height)
    


#     invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]
#     nBins = 100
#     fig, axs = plt.subplots(2, 2)
#     fig.set_size_inches(15, 10)
#     # fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

#     x = x.squeeze()
#     w = [1 / sqrt(2 * pi) * exp(-x ** 2 / 2) for x in np.linspace(-10, 10, 1000)]
#     w_0 = [density(x,centers=[-3,3,3], weights=[1/3,1/3,1/3]) for x in np.linspace(-10,10,1000)]

#     axs[0, 0].hist(list(x.detach().cpu().numpy()), density=True, bins=nBins)
#     axs[0, 0].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
#     axs[0, 0].set_title('x from rho_0')

#     # n, bins, patches=axs[0,1].hist(list(fx.detach().cpu().numpy()[:,0]),density=True, bins=nBins)


#     axs[1, 0].hist(list(finvfx.detach().cpu().numpy()[:,0]), density=True, bins=nBins)
#     axs[1, 0].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
#     axs[1, 0].set_title('finv( f(x) )')

#     fx1 = fx.detach().cpu().numpy()
#     genModel1 = genModel.detach().cpu().numpy()
#     x_numpy = x.detach().cpu().numpy().squeeze()
#     # all_points1 = []
#     # # all_points2 = []
#     # all_points3 = []
#     # print(fx1[:, -1][:100])
#     # print(genModel[:,-1][:100])

#     # for i in range(fx1.shape[0]):
#     #     all_points1.extend(list(np.ones(int(fx1[:, -1][i]*50))*fx1[:, 0][i]))
#     #     # all_points2.extend(list(np.ones(int(fx1[:, -1][i]*100))*x_numpy[i]))
#     #     all_points3.extend(list(np.ones(int(genModel1[:, -1][i] * 50)) * genModel1[:, 0][i]))



#     # axs[1, 1].hist(all_points3, density=True, bins=nBins)
#     axs[1, 1].hist(genModel1[:,0], density=True, weights=genModel1[:,-1], bins=nBins)
#     axs[1, 1].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
#     axs[1, 1].set_title('finv( y from rho1 )')

#     # axs[1, 2].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
#     # axs[1, 2].hist(all_points1, density=True, bins=nBins)
#     # axs[1, 2].set_title('w(x)f(x)')
#     # axs[1, 2].set_xlim(-5, 5)
#     #
#     # axs[0, 1].hist(list(fx.detach().cpu().numpy()[:, 0]), density=True, bins=nBins)
#     # axs[0, 1].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
#     # axs[0, 1].set_title('f(x)')
#     # axs[0, 1].set_xlim(-5, 5)

#     axs[0, 1].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
#     axs[0, 1].hist(fx1[:,0], density=True, weights=fx1[:, -1],bins=nBins)

#     axs[0, 1].set_title('w(x)f(x)')
#     axs[0, 1].set_xlim(-5, 5)


# # //

#     # axs[1, 1].hist(list(genModel.detach().cpu().numpy()[:, 0]), density=True, bins=nBins)
#     # axs[1, 1].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
#     # axs[1, 1].set_title('finv( y from rho1 )')

#     if not os.path.exists(os.path.dirname(sPath)):
#         os.makedirs(os.path.dirname(sPath))
#     plt.savefig(sPath, dpi=300)
#     plt.close()


#     fig, axs = plt.subplots(3, 3)
#     fig.set_size_inches(15, 15)
#     # fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

#     if doPaths:
#         forwPath, phi = integrate_ex(x.unsqueeze(-1)[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True, alphaa=alphaa)
#         # backPath, phi_inv = integrate_ex(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True, alphaa=alphaa)

#         forwPath1=forwPath.detach().cpu().numpy()
#         phi1=phi.detach().cpu().numpy()
#         for i in range(9):
#             a = int(i/3)
#             b = i-a*3
#             axs[a, b].hist(forwPath1[:,0,i], density=True, weights=forwPath1[:,-1,i], bins=nBins)
#             axs[a, b].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
#             axs[a, b].set_title('t={}'.format(0.125*i))
#             axs[a, b].set_xlim(-5, 5)

#         if not os.path.exists(os.path.dirname(sPath2)):
#             os.makedirs(os.path.dirname(sPath2))
#         plt.savefig(sPath2, dpi=300)
#         plt.close()

#         fig, axs = plt.subplots(3, 3)
#         fig.set_size_inches(15, 15)
#         for i in range(9):
#             a = int(i/3)
#             b = i-a*3
#             axs[a, b].hist(forwPath1[:,0,i], density=True, bins=nBins)
#             axs[a, b].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
#             axs[a, b].set_title('t={}'.format(0.125*i))
#             axs[a, b].set_xlim(-5, 5)

#         if not os.path.exists(os.path.dirname(sPath5)):
#             os.makedirs(os.path.dirname(sPath5))
#         plt.savefig(sPath5, dpi=300)
#         plt.close()


#         # plot weight_1
#         for i in range(nt_val + 1):
#             tt1 = np.sort(forwPath1[:, 0, i].squeeze())[::1]
#             idx = np.argsort(forwPath1[:, 0, i].squeeze())[::1]
#             tt2 = forwPath1[:, -1, i].squeeze()
#             tt2 = tt2[idx]
#             if i%2 == 0:
#                 plt.scatter(tt1, tt2,s=0.1,label='t={}'.format(i/nt_val))
#                 # plt.plot(tt1, tt2, label='t={}'.format(i / nt_val), linewidth=5)
#             else:
#                 plt.scatter(tt1, tt2,s=0.1)
#                 # plt.plot(tt1, tt2, linewidth=5)

#         plt.ylabel('weight')
#         plt.xlabel('position')
#         plt.legend()
#         if not os.path.exists(os.path.dirname(sPath3)):
#             os.makedirs(os.path.dirname(sPath3))
#         plt.savefig(sPath3, dpi=300)
#         plt.close()

#         # plot weight_2
#         for i in range(nt_val + 1):
#             tt1 = np.sort(x_numpy)[::1]
#             idx = np.argsort(x_numpy)[::1]
#             tt2 = forwPath1[:, -1, i].squeeze()
#             tt2 = tt2[idx]
#             if i%2==0:
#                 plt.scatter(tt1,tt2,s=0.5,label='t={}'.format(i/nt_val))
#                 # plt.plot(tt1,tt2,label='t={}'.format(i/nt_val),linewidth=5)
#             else:
#                 plt.scatter(tt1,tt2,s=0.5)
#                 # plt.plot(tt1, tt2, linewidth=5)
#         plt.ylabel('weight')
#         plt.xlabel('position')
#         plt.legend()
#         if not os.path.exists(os.path.dirname(sPath4)):
#             os.makedirs(os.path.dirname(sPath4))
#         plt.savefig(sPath4, dpi=300)
#         plt.close()

#         # fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))



#         # plot the forward and inverse trajectories of several points; white is forward, red is inverse
#     #     nPts = 30
#     #     pts = np.unique(np.random.randint(nSamples, size=nPts))
#     #     for pt in pts:
#     #         axs[0, 3].plot(forwPath[pt, 0, :].detach().cpu().numpy(), np.linspace(0,1,nt_val+1), color='green', linewidth=1)
#     #         # axs[1, 3].plot(backPath[pt, 0, :].detach().cpu().numpy(), np.linspace(0,1,nt_val+1), color='red', linewidth=1)
#     #
#     #
#     #     for i in range(nt_val+1):
#     #         tt1 = np.sort(forwPath[:, 0, i].detach().cpu().numpy().squeeze())[::1]
#     #         idx = np.argsort(forwPath[:, 0, i].detach().cpu().numpy().squeeze())[::1]
#     #         tt2 = forwPath[:, -1, i].detach().cpu().numpy().squeeze()
#     #         tt2 = tt2[idx]
#     #         if i%2 == 0:
#     #             axs[0, 4].plot(tt1, tt2,label='t={}'.format(i/nt_val))
#     #         else:
#     #             axs[0, 4].plot(tt1, tt2)
#     #
#     #         # , label='t={}'.format(i/nt_val)
#     #
#     #         tt3 = phi[:, :, i].squeeze().cpu().numpy()
#     #         tt3 = tt3[idx]
#     #         axs[0, 2].plot(tt1, tt3) #, label='t={}'.format(i/nt_val)
#     #
#     #         tt1 = np.sort(x_numpy)[::1]
#     #         idx = np.argsort(x_numpy)[::1]
#     #         tt3 = phi[:, :, i].squeeze().cpu().numpy()
#     #         tt3 = tt3[idx]
#     #         axs[1, 4].plot(tt1, tt3)
#     #         if i%2==0:
#     #             axs[1, 4].plot(tt1, tt3, label='t={}'.format(i/nt_val))
#     #         else:
#     #             axs[1, 4].plot(tt1,tt3)
#     #
#     #
#     #         # plot from another angle
#     #         tt1 = np.sort(x_numpy)[::1]
#     #         idx = np.argsort(x_numpy)[::1]
#     #         tt2 = forwPath[:, -1, i].detach().cpu().numpy().squeeze()
#     #         tt2 = tt2[idx]
#     #         if i%2==0:
#     #             axs[1, 3].plot(tt1,tt2*22222,label='t={}'.format(i/nt_val))
#     #         else:
#     #             axs[1, 3].plot(tt1,tt2*22222)
#     #
#     # axs[0, 4].legend()
#     # axs[0, 4].set_title('weights at rho_T')
#     # axs[1, 4].legend()
#     # axs[1, 4].set_title('\Phi at rho_0')
#     #
#     # axs[1, 3].set_title('weights at rho_0 (scaled)')
#     # axs[0, 2].set_title('\Phi')





def plot1d_OT(net, x, y, nt_val, sPath1, sPath2, sTitle="", doPaths=False):
    """
    only used to plot toy 1-dimension
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """
    d = net.d
    if d != 1:
        print("Error dimension")
        return -1
    nSamples = x.shape[0]

    x = x.unsqueeze(-1)
    fx = integrate(x, net, [0.0, 1], nt_val, stepper="rk4", alph=net.alph)

    finvfx = integrate(fx[:, 0].unsqueeze(-1), net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel = integrate(y, net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 100

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 10)
    fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    x = x.squeeze()
    w = [1 / sqrt(2 * pi) * exp(-x ** 2 / 2) for x in np.linspace(-10, 10, 1000)]
    w_0 = [density(x,centers=[-3,3,3], weights=[1/3,1/3,1/3]) for x in np.linspace(-10,10,1000)]

    axs[0, 0].hist(list(x.detach().cpu().numpy()), density=True, bins=nBins)
    axs[0, 0].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
    axs[0, 0].set_title('x from rho_0')

    axs[0, 1].hist(list(fx.detach().cpu().numpy()[:, 0]), density=True, bins=nBins)
    axs[0, 1].plot(np.linspace(-10, 10, 1000), w, 'r',linestyle='--',linewidth=5)
    axs[0, 1].set_title('f(x)')
    axs[0, 1].set_xlim(-5, 5)


    axs[1, 0].hist(list(finvfx.detach().cpu().numpy()[:,0]), density=True, bins=nBins)
    axs[1, 0].plot(np.linspace(-10, 10, 1000), w_0, 'r', linestyle='--', linewidth=5)
    axs[1, 0].set_title('finv( f(x) )')

    axs[1, 1].hist(list(genModel.detach().cpu().numpy()[:, 0]), density=True, bins=nBins)
    axs[1, 1].plot(np.linspace(-10, 10, 1000), w_0, 'r',linestyle='--',linewidth=5)
    axs[1, 1].set_title('finv( y from rho1 )')

    # if not os.path.exists(os.path.dirname(sPath1)):
    #     os.makedirs(os.path.dirname(sPath1))
    plt.savefig(sPath1, dpi=300)
    plt.close()

    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(15, 15)
    if doPaths:
        forwPath = integrate(x.unsqueeze(-1), net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        # backPath, phi_inv = integrate_ex(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True, alphaa=alphaa)

        forwPath1 = forwPath.detach().cpu().numpy()
        for i in range(9):
            a = int(i / 3)
            b = i - a * 3
            axs[a, b].hist(forwPath1[:, 0, i], density=True, bins=nBins)
            axs[a, b].plot(np.linspace(-10, 10, 1000), w, 'r', linestyle='--', linewidth=5)
            axs[a, b].set_title('t={}'.format(0.125 * i))
            axs[a, b].set_xlim(-5, 5)

        # if not os.path.exists(os.path.dirname(sPath2)):
        #     os.makedirs(os.path.dirname(sPath2))
        plt.savefig(sPath2, dpi=300)
        plt.close()

def plotAutoEnc(x, xRecreate, sPath):

    # assume square image
    s = int(math.sqrt(x.shape[1]))


    nex = 8

    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].reshape(s, s).detach().cpu().numpy())


    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plotAutoEnc3D(x, xRecreate, sPath):

    nex = 8

    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())


    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()



def plotImageGen(x, xRecreate, sPath):

    # assume square image
    s = int(math.sqrt(x.shape[1]))

    nex = 80
    nCols = nex//5


    fig, axs = plt.subplots(7, nCols)
    fig.set_size_inches(16, 7)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nCols):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        # axs[1, i].imshow(x[ nex//3 + i , : ].reshape(s,s).detach().cpu().numpy())
        # axs[2, i].imshow(x[ 2*nex//3 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[nCols + i,:].reshape(s,s).detach().cpu().numpy())
        
        axs[4, i].imshow(xRecreate[2*nCols + i,:].reshape(s,s).detach().cpu().numpy())
        axs[5, i].imshow(xRecreate[3*nCols + i , : ].reshape(s, s).detach().cpu().numpy())
        axs[6, i].imshow(xRecreate[4*nCols + i , : ].reshape(s, s).detach().cpu().numpy())

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plot4mnist(x, sPath, sTitle=""):
    """
    x - tensor (>4, 28,28)
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle)

    im1 = axs[0, 0].imshow(x[0,:,:].detach().cpu().numpy())
    im2 = axs[0, 1].imshow(x[1,:,:].detach().cpu().numpy())
    im3 = axs[1, 0].imshow(x[2,:,:].detach().cpu().numpy())
    im4 = axs[1, 1].imshow(x[3,:,:].detach().cpu().numpy())

    fig.colorbar(im1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(im2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(im3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(im4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()





