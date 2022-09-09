import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
# import autograd
import numpy as np

#defining the NN
class Net_PDE_3(nn.Module):
    def __init__(self):
      super(Net_PDE_3, self).__init__()
      self.fc1 = nn.Linear(3, 40)
      self.fc2 = nn.Linear(40, 40)
      self.fc3 = nn.Linear(40, 1)

      # self.RAF = sin_relu2(40)

    # x represents our data
    def forward(self, x):
      # Pass data through fc1
      x = self.fc1(x)
      # x = self.RAF(x)#F.relu(x)**3
      x = F.relu(x)**3
      x = self.fc2(x)
      # x = self.RAF(x)#F.relu(x)**3
      x = F.relu(x)**3
      x = self.fc3(x)

      return x

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

#Calculating the gradient
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y).to(device)
    #grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, allow_unused=True)[0].to(device)
    return grad

#Calculating the Jacobian
def get_jacobian(x, nn, param_shape):
  net_out = nn(x).flatten().to(device)
  jac = torch.zeros([net_out.shape[0], param_shape]).to(device)
  for i in range(net_out.shape[0]):
    gradients = []
    for param in nn.parameters():
      gradients.append(gradient(net_out[i], param).flatten())
    gradients = torch.cat(gradients)
    gradients = torch.unsqueeze(gradients, 0)
    jac[i, :] = gradients
  return jac

# Flattening the parameters into a list
def param_flat(nn):
  param_list = []
  for param in nn.parameters():
    param_list.append(param.flatten())
  param_list = torch.unsqueeze(torch.cat(param_list), 1)
  return param_list


####Initial Condition####

# Create random input and output data
dtype = torch.float
device = torch.device("cpu")
N = 10000
x = 0.001 + 0.998*torch.rand(N, 3)  # torch.linspace(-0.9999, 0, N, device=device, dtype=dtype)
print(x.shape)

# x = torch.unsqueeze(x, 1)
# Initial_condition=@(x) 1/2*sum(x.^2,1)+0*(prod(x,1).*prod(1-x,1)).*sin(2*pi*sum(x,1));

x_test = torch.rand(int(N/10), 3)  # torch.linspace(-0.9999, 0, N, device=device, dtype=dtype)
print(x.shape)

# x = torch.unsqueeze(x, 1)
# Initial_condition=@(x) 1/2*sum(x.^2,1)+0*(prod(x,1).*prod(1-x,1)).*sin(2*pi*sum(x,1));

y_test = 0.5*torch.sum(x_test**2, dim=1) + 10*(torch.prod((1-x_test), 1))*torch.prod(x_test, 1) # *torch.sin(2*math.pi*torch.sum(x_test,1))
y_test = y_test.unsqueeze(-1)

error_Alg3 = np.array([])

my_nn = Net_PDE_3()
print(my_nn)
my_nn.to(device)
result = my_nn(x)

error_10OP = np.array([])
ALR = np.array([])
ratio = np.array([])

M = 1000
dim = 3
Bdry = np.zeros([1,3])
for i in range(dim):
 L_bdry=np.random.uniform(size=[M,3])
 L_bdry[:,i]=0
 R_bdry=np.random.uniform(size=[M,3])
 R_bdry[:,i]=1
 #Bdry=np.append(np.append(Bdry,L_bdry, axis=0), R_bdry, axis=0)
 Bdry=np.concatenate((Bdry,L_bdry), axis=0)
 Bdry=np.concatenate((Bdry,R_bdry), axis=0)

x_bound = torch.from_numpy(Bdry)
x_bound = x_bound.type_as(x)

x_in = torch.cat((x, x_bound), 0)
#y =  0.5*torch.sum(x_in**2,dim=1) + 1*(torch.prod(x_in,1)*torch.prod(1-x_in,1))*torch.sin(2*math.pi*torch.sum(x_in,1))
y =  0.5*torch.sum(x_in**2,dim=1) + 10*(torch.prod((1-x_in),1))*torch.prod(x_in,1)#*torch.sin(2*math.pi*torch.sum(x_in,1))
y = y.unsqueeze(-1)

batch_size = 200
learning_rate = 1e-3
num_epoch = 200
num_iter = int(num_epoch*N/batch_size)

# Regression for the initial condition
for it in range(num_iter):
  idx = np.random.choice(x_in.shape[0], batch_size,replace=False)
  net_out = my_nn(x_in[idx])
  R = (y[idx] - net_out)  # Defining R

  # Renew samples each iteration
  param_shape = list(torch.nn.utils.parameters_to_vector(my_nn.parameters()).shape)
  grad_R = get_jacobian(x_in[idx], my_nn, param_shape[0])  # Getting the gradient with respect to the parameters
  A = grad_R
  b = R
  alpha = torch.matmul(torch.pinverse(A), b)
  param_vec = param_flat(my_nn)  # Calculating the parameter vector
  param_vec = param_vec + learning_rate*alpha  # Updating the parameters
  param_vec = torch.squeeze(param_vec)
  torch.nn.utils.vector_to_parameters(param_vec, my_nn.parameters())  # Putting the updated parameters back in the graph
  if it%10 == 0:
    error = np.linalg.norm(my_nn(x_test).cpu().detach().numpy()-y_test.cpu().detach().numpy())/np.linalg.norm(y_test.cpu().detach().numpy())
    error_10OP = np.append(error_10OP, error)
    print("it:{}, error:{}".format(it,error))
    ALR = np.append(ALR, learning_rate)
    if it>500 and it%500==0:
      err_rat = np.mean(error_10OP[-40:-20])/np.mean(error_10OP[-20:])
      print('Ratio')
      print(err_rat)
      ratio = np.append(ratio, err_rat)
      if  err_rat < 1.01:
        learning_rate *= 0.9
    if it % 1000 == 0:
      fig, ax = plt.subplots()
      it = 10*np.arange(error_10OP.shape[0])
      ax.plot(it, (error_10OP),color='r', label='20D Output')

      ax.set_xlabel('Iterations')
      ax.set_title('Error - 30D Output')
      ax.legend(loc='best')
      ax.set_yscale('log')
      plt.savefig("/home/chenjiaheng/OT flow/PDE_pictures/init_err")
      plt.clf()


###Solve PDE###

def my_nullspace(At, rcond=None):

    ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    vht=vht.T
    #print(vht.shape)
    #vht = torch.transpose(vht,0,1)

    Mt, Nt = ut.shape[0], vht.shape[1]
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    #numt= torch.sum(st > tolt)
    #numt = numt.int()
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.cpu().conj()
    #nullspace = np.conj(torch.transpose(vht[numt:,:],0,1).detach().cpu().numpy())
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    #return torch.from_numpy(nullspace)
    return nullspace

x.requires_grad=True
x_test = torch.rand(int(N/10),3)

y_bound = 0.5*torch.sum(x_bound**2,dim=1)
y_bound = torch.unsqueeze(y_bound, 1)
y_test = 0.5*torch.sum(x_test**2,dim=1)
y_test = torch.unsqueeze(y_test, 1)
y = 0.5*torch.sum(x**2,dim=1)
y = torch.unsqueeze(y, 1)
#x_bound = torch.unsqueeze(x_bound, 1)
batch_size = 200
bound_batch_size = 10
learning_rate = 1e-7
num_epoch = 2000
error_Alg3_MultiD = np.array([])
error = np.linalg.norm(my_nn(x).detach().numpy()-y.detach().numpy())/np.linalg.norm(y.detach().numpy())
print(error)

from torch.autograd import grad



num_iter = int(num_epoch*N/batch_size)
for it in range(num_iter):
  idx = np.random.choice(x.shape[0], batch_size,replace=False)
  x_batch = x[idx]
  idx_b = np.random.choice(x_bound.shape[0], bound_batch_size,replace=False)
  x_b = x_bound[idx_b]
  u = my_nn(x_batch)
  u.backward(torch.ones(u.size()),retain_graph=True)
  u_x = grad(outputs=u[:,0:1], inputs=x_batch, grad_outputs=torch.ones_like(u[:,0:1]), create_graph=True)[0]
  u_xx = grad(outputs=u_x, inputs=x_batch, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
  for i in range(1,u.shape[1]):
    u_x_val = grad(outputs=u[:,i:1+1], inputs=x_batch, grad_outputs=torch.ones_like(u[:,i:1+1]), create_graph=True)[0]
    u_xx_val = grad(outputs=u_x_val, inputs=x_batch, grad_outputs=torch.ones_like(u_x_val), create_graph=True)[0]
    u_x = torch.cat((u_x, u_x_val), 0)
    u_xx = torch.cat((u_xx, u_xx_val), 0)
  u_xx = torch.unsqueeze(torch.sum(u_xx, 1), 1)
  B = get_jacobian(x_b,my_nn,param_shape[0]) #Getting the gradient with respect to the parameters
  D = my_nullspace(B)
  A = torch.matmul(get_jacobian(x_batch,my_nn,param_shape[0]),D) #Getting the gradient with respect to the parameters
  #A = torch.cat((grad_in, B), 0).float()



  R = (u_xx - 3*torch.ones_like(u)) #(u_xx - u**3)  #Defining R
  #u_b = torch.tensor(np.array([[y[0]], [y[-1]]]))
  #B = u_b - my_nn(x_b) #torch.zeros([2,1])  # u_b - my_nn(x_b)
  print(R.shape)
  #Renew samples each iteration

  #Generate a small A and visualize it, with  full column rank
  param_shape = list(torch.nn.utils.parameters_to_vector(my_nn.parameters()).shape)

  CT_A = torch.conj(torch.transpose(A,0,1)).float()

  alpha = torch.matmul(torch.matmul(CT_A, torch.pinverse(torch.matmul(A, CT_A))), R)  # A'*pinv(A*A')*b torch.matmul(torch.inverse(A), b)
  param_vec = param_flat(my_nn)   # Calculating the parameter vector
  print(param_vec.shape)

  param_vec = param_vec + learning_rate*torch.matmul(D,alpha) #Updating the parameters
  param_vec = torch.squeeze(param_vec)
  torch.nn.utils.vector_to_parameters(param_vec, my_nn.parameters()) #Putting the updated parameters back in the graph
  if it%10==0:
    print(it)
    error = np.linalg.norm(my_nn(x_test).detach().numpy()-y_test.detach().numpy())/np.linalg.norm(y_test.detach().numpy())
    error_Alg3_MultiD = np.append(error_Alg3_MultiD,error)
    print(error)
  if it%1000 == 0:
     fig, ax = plt.subplots()
     it = 10*np.arange(error_Alg3_MultiD.shape[0])
     ax.plot(it, (error_Alg3_MultiD),color='r', label='20D Output')

     ax.set_xlabel('Iterations')
     ax.set_title('Error - Multi Input')
     ax.legend(loc='best')
     ax.set_yscale('log')
     plt.savefig('./PDE_err')
     plt.clf()
