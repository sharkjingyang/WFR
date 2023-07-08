import numpy as np
import scipy.io
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from svgd import SVGD
import math

def gamma_pdf(x, a, b):
    """
    计算 gamma 分布的概率密度函数
    参数:
    x (float): 随机变量的取值
    k (float): 形状参数
    theta (float): 尺度参数
    
    返回值:
    float: x 对应的概率密度函数值
    """
    if x <= 0 or a <= 0 or b<= 0:
        return 0
    else:
        coeff =  math.pow(1/b, a) /math.gamma(a)
        power_term = math.pow(x, a - 1)
        exponential_term = math.exp(-x *b)
        return coeff * power_term * exponential_term


def multivariate_normal_pdf(x, mean, s):
    d = len(x)
    cov=np.diag(s**2)
    coeff = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponent)

class BayesianLR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0
        
        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0
    
        
    def dlnprob(self, theta):
        
        if self.batchsize > 0:
            batch = [ i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize) ]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])
            
        Xs = self.X[ridx, :]  # 256*54
        Ys = self.Y[ridx]
        
        w = theta[:, :-1]  # logistic weights   100*54 sample number*feature
        
        alpha = np.exp(theta[:, -1])  # the last column is logalpha  
        d = w.shape[1]  #d=feature
        
        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1)) #||w||^2 * alpha/2
        
        coff = np.matmul(Xs, w.T)
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))   # 1/(1+exp(-x.T * w))  256*100   100个sample分别对256个样本做出的预测概率
    
        
        dw_data = np.matmul(((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d) , w)
       
        dw = dw_data * 1.0 * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale
        
        # dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # the last term is the jacobian term
        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha   # the last term is the jacobian term
        
        return np.hstack([dw, np.vstack(dalpha)])  # % first order derivative 
    
    def evaluation(self, theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
    
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]

if __name__ == '__main__':
    data = scipy.io.loadmat('Stein-Variational-Gradient-Descent-master/data/covertype.mat')

    
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1
    
    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d + 1
    
    # split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
    print("X_train shape")
    print(X_train.shape)


    a0, b0 = 1, 0.01 #hyper-parameters
    model = BayesianLR(X_train, y_train, 256, a0, b0) # batchsize = 256
    
    # initialization
    M = 100  # number of particles
    theta0 = np.zeros([M, D]);
    alpha0 = np.random.gamma(a0, b0, M); 

    mode="generate_data_OT" # "original SVGD"
    if mode=="original SVGD":
        for i in range(M):
            theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])
        theta = SVGD().update(x0=theta0, lnprob=model.dlnprob, bandwidth=-1, n_iter=6000, stepsize=0.05, alpha=0.9, debug=True)
        print ('[accuracy, log-likelihood]')
        print (model.evaluation(theta, X_test, y_test))
        print("--------------end-------------")
    
    if mode=="generate_data_OT":
        for gen_itr in range(100): #generate 10000 samples by repeat 100 times SVGD
            print("----------------gen_itr=%d-----------"%gen_itr)
            theta0 = np.zeros([M, D])
            alpha0 = np.random.gamma(a0, b0, M)
            for i in range(M):
                theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])
            theta = SVGD().update(x0=theta0, lnprob=model.dlnprob, bandwidth=-1, n_iter=6000, stepsize=0.05, alpha=0.9, debug=True)
            # print ('[accuracy, log-likelihood]')
            # print (model.evaluation(theta, X_test, y_test))
            print("--------------end-------------")

            if gen_itr==0:
                theta_record=theta.copy()
            else:
                theta_record=np.vstack([theta_record,theta])
                print(theta_record.shape)
        np.save("data/theta_SVGD/noweight_theta.npy",theta_record)
        
    
    if mode=="generate_data_WFR":
        theta_sample=np.load("data/theta_SVGD/noweight_theta.npy") # to obtain mu and s
        mu=theta_sample.mean(axis=0)
        s=theta_sample.std(axis=0)
        prior_flag="Gaussian"
        for gen_itr in range(100):
            print("----------------gen_itr=%d-----------"%gen_itr)
            theta0 = np.zeros([M, D+1]);
            alpha0 = np.random.gamma(a0, b0, M); 
            for i in range(M):
                if prior_flag=="gamma":
                    theta0[i, :-1] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])
                    w=theta0[i, :-2]
                    save_path="data/theta_SVGD/theta_SVGD_gamma.npy"
                    # prior_theta=(alpha0[i]/2/np.pi)**(d/2)*np.exp(-alpha0[i]/2*np.sum(w**2))*gamma_pdf(alpha0[i], a0, b0)

                if prior_flag=="Gaussian":
                    theta0[i, :-1] = np.random.normal(0, 1, d+1)*s+mu
                    w_aug=theta0[i, :-1]
                    w=theta0[i, :-2]
                    save_path="data/WFR_SVGD/WFR_SVGD_Gaussian.npy"
                    # prior_theta=multivariate_normal_pdf(w_aug, mu, s)

                n_test=len(y_test)
                prob = np.zeros([n_test, 1])
                coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(w, n_test, 1), X_test), axis=1))
                prob[:, 0] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
                llh = np.exp(np.mean(np.log(prob)) )
                # print(llh)
                theta0[i,-1]=llh  ##last row is likelihood   i.e. weight

            if gen_itr==0:
                theta_record=theta0.copy()
            else:
                theta_record=np.vstack([theta_record,theta0])
                print(theta_record.shape)   
        #theta_record 应该比X_train多两维，-2是log alpha，-1是llh
        theta_record[:,-1]=theta_record[:,-1]/np.sum(theta_record[:,-1])*theta_record.shape[0]
        np.save(save_path,theta_record)
