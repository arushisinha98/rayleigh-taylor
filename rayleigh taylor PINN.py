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

def Poisson(streamfunction, omega, residual, R, V, S, epsilon)
  """ Poisson solver. Updates streamfunction, residual, R, V, S, epsilon """
  for j in range(2,n):
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
      streamfunction[j][0] = streamfunction[j][nx-1] # update ghost node at x = 0
      streamfunction[j][1] = streamfunction[j][nx] # enforce (x = 1) == (x = nx)
      streamfunction[0] = streamfunction[ny-1] # update ghost node at y = 0
      streamfunction[1] = streamfunction[ny] # enforce (y = 1) == (y = ny)
      
      for index in range(0,nx+2):
        streamfunction[0][index] = 0.0
        streamfunction[1][index] = 0.0
        streamfunction[ny][index] = 0.0
        streamfunction[ny+1][index] = 0.0
        
      R = max_norm(remove_ghost_nodes(residual))
      V = max_norm(remove_ghost_nodes(omega))
      S = one_norm(remove_ghost_nodes(streamfunction))
      
      epsilon = R / ((2*(1/(dx^2) + 1/(dy^2))*S) + V)
      
  return streamfunction, omega, residual, R, V, S, epsilon
  
if __name__ == "__main__":
  N_train = 5000
  layers = [, , , , ] # specify the layers that we want
  
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
  
  files = ["Omega_" + str(num) + ".csv" for num in range(10,1000,10)]
  for file in files:
    with open(file) as file_name:
      img = np.loadtxt(file_name, delimiter = ",")
      omega_vals.append(np.array(img))
  print(omega_vals.shape)
  
  files = ["Theta_" + str(num) + ".csv" for num in range(10,1000,10)]
  for file in files:
    with open(file) as file_name:
      img = np.loadtxt(file_name, delimiter = ",")
      theta_vals.append(np.array(img))
  print(theta_vals.shape)
  
  # training data -- random w/o replacement
  idx = np.random.choice(nt, replace = False)
  streamfunction_train = streamfunction_vals[idx,:]
  omega_train = omega_vals[idx,:]
  theta_train = theta_vals[idx,:]
  
  # training
  model = PhysicsInformedNN(streamfunction_train, omega_train, theta_train, t_train, layers)
  
  
def save_fig(outfile, files, fps = 5, loop = 1):
  """ Function to save GIFs """
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp = outfile, format = 'GIF', append_images = imgs[1:], save_all = True,
               duration = int(1000/fps), loop = loop)
