import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from plotting import newfig, savefig

## define & declare global variables
L_x = 0.25 # domain size (x) = 1/4 m
L_y = 1.0 # domain size (y) = 1 m
nx = 101 # 101 grid points in 0 <= x <= L_x
ny = 401 # 401 grid points in 0 <= y <= L_y
dx = L_x/(nx-1) # delta x
dy = L_y/(ny-1) # delta y
dt = 0.05 # delta t = 0.5 s
d_theta = 0.5 # delta theta = 0.5 K
theta_0 = 300 # theta 0 = 300 K
g = 9.80665 # gravitational acceleration = 9.80665 m/s^2
T_total = 50 # T(total) = 50 seconds
nt = T_total/dt
save_frequency = 10 # save solution every 10 iterations

# artificial diffusion/viscosity
K_x = pow(10,-3)
K_y = K_x
beta = 207*pow(10,-6)

# formatting values
width = 14 # width of fixed-length data fields
precision = 6 # precision of data fields
# error bound values
tolerance = 0.000000001
max_iterations = 2000
# calculate alpha
sigma = (1/(1+pow((dx/dy),2)))*(cos(2*np.pi/nx)+pow((dx/dy),2)*cos(2*np.pi/ny))
alpha = 2/(1 + sqrt(1 - pow(sigma,2)))

def Poisson(streamfunction, omega)
  """ Poisson solver. Updates streamfunction and computes residual using omega. """
  # initialize residual matrix
  residual = [[0.0 for __ in range(nx)] for __ in range(ny)]
  
  for j in range(2,ny):
    if j == ny: # if periodic
      streamfunction[ny+1] = streamfunction[2] # update ghost node at y = ny+1
    for i in range(1,nx+1):
      if i == nx:
        streamfunction[j][nx+1] = streamfunction[j][2] # update ghost node at x = nx+1
        
      # calculate residual for interior nodes
      residual[j][i] = (1/(dx^2))*(streamfunction[j][i-1] - 2*streamfunction[j][i] + streamfunction[j][i+1]) + \
                       (1/(dy^2))*(streamfunction[j-1][i] - 2*streamfunction[j][i] + streamfunction[j+1][i]) - \
                       omega[j][i]
      
      # calculate streamfunction
      streamfunction[j][i] = streamfunction[j][i] + alpha*residual[j][i]/(2*(1/(dx^2) + 1/(dy^2)))
    # streamfunction[j][0] = streamfunction[j][nx-1] # update ghost node at x = 0
    # streamfunction[j][1] = streamfunction[j][nx] # enforce (x = 1) == (x = nx)
  # streamfunction[0] = streamfunction[ny-1] # update ghost node at y = 0
  # streamfunction[1] = streamfunction[ny] # enforce (y = 1) == (y = ny)
      
  for index in range(0,nx+2):
    streamfunction[0][index] = 0.0
    streamfunction[1][index] = 0.0
    streamfunction[ny][index] = 0.0
    streamfunction[ny+1][index] = 0.0
      
  return residual

def Jacobian(f1, f2):
  """ computes the Jacobian of two matrices, f1 and f2, and returns a scalar. """
  return J
  
class PhysicsInformedNN:
  # initialize the class
  def __init__(self, x, y, t, streamfunction, omega, theta, layers):
    X = np.concatenate([x, y, t],1)
    self.lb = X.min(0) #?
    self.ub = X.max(0) #?
    
    self.X = X
    self.x = X[:,0:1]
    self.y = X[:,1:2]
    self.t = X[:,2:3]
    
    self.streamfunction = streamfunction
    self.omega = omega
    self.theta = theta
    
    self.layers = layers
    
    # initialize NN
    self.weights, self.biases = self.initialize_NN(layers)
    # initialize parameters
    #?
    self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
                                                   log_device_placement = True))
    self.x_tf = tf.placeholder(tf.float32, shape = [None, self.x.shape[1]])
    self.y_tf = tf.placeholder(tf.float32, shape = [None, self.y.shape[1]])
    self.t_tf = tf.placeholder(tf.float32, shape = [None, self.t.shape[1]])
    
    self.streamfunction_tf = tf.placeholder(tf.float32, shape = [None, self.streamfunction.shape[1]])
    self.omega_tf = tf.placeholder(tf.float32, shape = [None, self.omega.shape[1]])
    self.theta_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])
    
    self.streamfunction_pred, self.omega_pred, self.theta_pred = net_NS(self.x_tf, self.y_tf, self.t_tf)
    
    self.loss = tf.reduce_sum(tf.square(self.streamfunction_tf - self.streamfunction_pred)) + \
                tf.reduce_sum(tf.square(self.omega_tf - self.omega_pred)) + \
                tf.reduce_sum(tf.square(self.theta_tf - self.theta_pred)) + \
                tf.reduce_sum(tf.square(Poisson(self.streamfuction_pred, self.omega_pred)))
    
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                            method = 'L-BFGS-B',
                                                            options = {'maxiter' = 50000,
                                                                       'maxfun' = 50000,
                                                                       'maxcor' = 50,
                                                                       'maxls' = 50,
                                                                       'ftol' = 1.0 * np.finfo(float).eps})
    self.optimizer_Adam = tf.train.AdamOptimizer()
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    
    init = tf.global_variables_initializer()
    self.sess.run(init)
    
  def initialize_NN(self, layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for ll in range(0, num_layers-1):
      W = self.xavier_init(size = [layers[ll], layers[ll+1]])
      b = tf.Variable(tf.zeros([1, layers[ll+1]], dtype = tf.float32), dtype = tf.float32)
      weights.append(W)
      biases.append(b)
    return weights, biases
  
  def xavier_init(self, size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)
  
  def neural_net(self, X, weights, biases):
    num_layers = len(weights) + 1
    H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # standardize
    for ll in range(0, num_layers - 2):
      W = weights[ll]
      b = biases[ll]
      H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y
                                                            
    
if __name__ == "__main__":
  N_train = 5000
  layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2] # specify the layers that we want
  
  # load data
  # read in training data (images are located in arushisinha98/rayleigh-taylor-2.0/ABs-101x401)
  # ("[VARIABLE]_[XXX].csv", XXX = 10, 20, ... 1000, VARIABLE = Streamfunction, Omega, Theta)
  streamfunction_vals, omega_vals, theta_vals = [], [], []
  
  files = ["Streamfunction_" + str(num) + ".csv" for num in range(10,1000,10)]
  for file in files:
    with open(file) as file_name:
      img = np.loadtxt(file_name, delimiter = ",")
      streamfunction_vals.append(np.array(img))
  print(streamfunction_vals.shape)
  print(streamfunction_vals.flatten()[:,None])
  
  files = ["Omega_" + str(num) + ".csv" for num in range(10,1000,10)]
  for file in files:
    with open(file) as file_name:
      img = np.loadtxt(file_name, delimiter = ",")
      omega_vals.append(np.array(img))
  print(omega_vals.shape)
  print(omega_vals.flatten()[:,None])
  
  files = ["Theta_" + str(num) + ".csv" for num in range(10,1000,10)]
  for file in files:
    with open(file) as file_name:
      img = np.loadtxt(file_name, delimiter = ",")
      theta_vals.append(np.array(img))
  print(theta_vals.shape)
  print(theta_vals.flatten()[:,None])
  
  # training data -- random w/o replacement
  idx = np.random.choice(nt, replace = False)
  streamfunction_train = streamfunction_vals[idx,:]
  omega_train = omega_vals[idx,:]
  theta_train = theta_vals[idx,:]
  
  # training
  model = PhysicsInformedNN(x_train, y_train, t_train, streamfunction_train, omega_train, theta_train, layers)
  model.train(200000)
  
  # test data (entire x, y, t, domain)
  t_test = np.linspace(0,nt)
  x_test = np.linspace(0,nx)
  y_test = np.linsapce(0,ny)
  
  streamfunction_pred, omega_pred, theta_pred = model.predict(x_test, y_test, t_test)
  
  # split streamfunction_pred into nt images and plot to compare with streamfunction_train images
  
  
def save_fig(outfile, files, fps = 5, loop = 1):
  """ Function to save GIFs """
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp = outfile, format = 'GIF', append_images = imgs[1:], save_all = True,
               duration = int(1000/fps), loop = loop)
