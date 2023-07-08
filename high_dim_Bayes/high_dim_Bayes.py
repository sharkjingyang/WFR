import torch
import numpy as np
import numpy.matlib as nm
import scipy.io
from sklearn.model_selection import train_test_split

def calculate_weight_for_theta(theta,X_test,y_test):
    
    weight=np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        if (i+1) %10==0:
            print(i)
        w=theta[i, :-1]
        n_test=len(y_test)
        prob = np.zeros([n_test, 1])
        coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(w, n_test, 1), X_test), axis=1))
        prob[:, 0] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        llh = np.exp(np.mean(np.log(prob)) )
        weight[i]=llh  ##last row is likelihood   i.e. weight
    weight_normalized=weight/weight.sum()*weight.shape[0]
    print("max weight  is %f" % weight_normalized.max())
    print("min weight  is %f" % weight_normalized.min())
    return weight

def evaluation( theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
    
        for t in range(M):
            if t%100==0:
                  print("itr %d"%t)
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]

def evaluation_weighted( theta, X_test, y_test):
        theta_w = theta[:, :-2]
        weight = theta[:, -1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
    
        for t in range(M):
            if t%100==0:
                  print("itr %d"%t)
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta_w[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob_weightedsum= np.mean(prob*weight, axis=1)
        acc = np.mean(prob_weightedsum > 0.5)
        # llh = np.mean(np.log(prob)*weight)
        llh = np.mean(np.log(prob_weightedsum))
        return [acc, llh]

data = scipy.io.loadmat('high_dim_Bayes/covertype.mat')
X_input = data['covtype'][:, 1:] # N*54
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N, 1])]) #N*55
d = X_input.shape[1]
D = d + 1
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)


theta_SVGD= np.load("data/theta_SVGD/noweight_theta.npy")[0:5000,:]  ##原始theta
# theta_weight=calculate_weight_for_theta(theta_SVGD,X_test,y_test) 


theta_SVGD_osc= theta_SVGD+np.random.randn(theta_SVGD.shape[0],theta_SVGD.shape[1])*5 ##加了noise的theta
weight_osc=calculate_weight_for_theta(theta_SVGD_osc,X_test,y_test) 
weight_osc_normalized=weight_osc/weight_osc.sum()*weight_osc.shape[0]

# theta_SVGD_Gaussian= np.random.randn(theta_SVGD.shape[0],theta_SVGD.shape[1])*10 ##加了noise的theta
# weight_Gaussian=calculate_weight_for_theta(theta_SVGD_Gaussian,X_test,y_test) 
# print(weight_Gaussian)


# print(weight)
# lh_max_index=np.argmax(weight)
# theta_max=theta_SVGD_osc[lh_max_index,:].reshape(1,-1)    ##likelihood最大的theta
# theta_SVGD_osc_mean=np.mean(theta_SVGD_osc*weight.reshape(-1,1),axis=0).reshape(1,-1)  ##加权平均后的theta


# theta_SVGD_weight=np.hstack((theta_SVGD,theta_weight.reshape(-1,1)))
theta_SVGD_osc_weight=np.hstack((theta_SVGD_osc,weight_osc.reshape(-1,1)))
# theta_SVGD_Gaussian_weight=np.hstack((theta_SVGD_Gaussian,weight_Gaussian.reshape(-1,1)))

np.save("data/data_WFR/noweight_theta.npy",theta_SVGD)
np.save("data/data_WFR/theta_SVGD_osc_weight.npy",theta_SVGD_osc_weight)

theta_SVGD_osc_weight_nomalized=np.hstack((theta_SVGD_osc,weight_osc_normalized.reshape(-1,1)))

print ('[accuracy, log-likelihood]')
print("-----------原始 SVGD samples------------------")
print (evaluation(theta_SVGD, X_input, y_input))
#[0.7556573702436439, -0.5144189375257269]


# print("-----------SVGD +noise samples (without weight)------------------")
# print (evaluation(theta_SVGD_osc, X_input, y_input))

# print("-----------SVGD +noise samples (likelihood最大的theta)------------------")
# print (evaluation(theta_max, X_input, y_input))

# print("-----------SVGD +noise samples (加权平均)------------------")
# print (evaluation(theta_SVGD_osc_mean, X_input, y_input))


print("-----------SVGD +noise samples (with weight)------------------")
print (evaluation_weighted(theta_SVGD_osc_weight_nomalized, X_input, y_input))








