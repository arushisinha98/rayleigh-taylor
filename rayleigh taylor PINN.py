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
L_z = 1.0 # domain size (z) = 1 m
nx = 101 # 101 grid points in 0 <= x <= L_x
nz = 401 # 401 grid points in 0 <= z <= L_z
dx = L_x/(nx-1) # delta x
dz = L_z/(nz-1) # delta z
dt = 0.05 # delta t = 0.5 s
d_theta = 0.5 # delta theta = 0.5 K
theta_0 = 300 # theta 0 = 300 K
g = 9.80665 # gravitational acceleration = 9.80665 m/s^2
T_total = 50 # T(total) = 50 seconds
nt = T_total/dt
save_frequency = 10 # save solution every 10 iterations

# artificial diffusion/viscosity
K_x = pow(10,-3)
K_z = K_x
beta = 207*pow(10,-6)

# formatting values
width = 14 # width of fixed-length data fields
precision = 6 # precision of data fields
# error bound values
tolerance = 0.000000001
max_iterations = 2000
# calculate alpha
sigma = (1/(1+pow((dx/dz),2)))*(cos(2*np.pi/nx)+pow((dx/dz),2)*cos(2*np.pi/nz))
alpha = 2/(1 + sqrt(1 - pow(sigma,2)))

# read in training data (images are located in arushisinha98/rayleigh-taylor-2.0/ABs-101x401)
# start with streamfunction ("Streamfunction_[XXX].csv", XXX = 10, 20, ... 1000)
x_train = []
files = ["Streamfunction_" + str(num) + ".csv" for num in range(10,1000,10)]
for file in files:
  with open(file) as file_name:
    profile = np.loadtxt(file_name, delimiter = " ")
    x_train.append(np.array(profile))

def Poisson(streamfunction, omega, residual, R, V, S, epsilon)
  """ Poisson solver. Updates streamfunction, residual, R, V, S, epsilon """
  for j in range(2,nz):
    if j == nz: # if periodic
      streamfunction[nz+1] = streamfunction[2] # update ghost node at y = nz+1
    for i in range(1,nx+1):
      if i == nx:
        streamfunction[j][nx+1] = streamfunction[j][2] # update ghost node at x = nx+1
        
      # calculate residual for interior nodes
      residual[j][i] = (1/(dx^2))*(streamfunction[j][i-1] - 2*streamfunction[j][i] + streamfunction[j][i+1]) + (1/(dz^2))*(streamfunction[j-1][i] - 2*streamfunction[j][i] + streamfunction[j+1][i]) - omega[j][i]
      
      # calculate streamfunction
      streamfunction[j][i] = streamfunction[j][i] + alpha*residual[j][i]/(2*(1/(dx^2) + 1/(dz^2)))
      streamfunction[j][0] = streamfunction[j][nx-1] # update ghost node at x = 0
      streamfunction[j][1] = streamfunction[j][nx] # enforce (x = 1) == (x = nx)
      streamfunction[0] = streamfunction[nz-1] # update ghost node at y = 0
      streamfunction[1] = streamfunction[nz] # enforce (y = 1) == (y = nz)
      
      for index in range(0,nx+2):
        streamfunction[0][index] = 0.0
        streamfunction[1][index] = 0.0
        streamfunction[nz][index] = 0.0
        streamfunction[nz+1][index] = 0.0
        
      R = max_norm(remove_ghost_nodes(residual))
      V = max_norm(remove_ghost_nodes(omega))
      S = one_norm(remove_ghost_nodes(streamfunction))
      
      epsilon = R / ((2*(1/(dx^2) + 1/(dz^2))*S) + V)
      
  return streamfunction, omega, residual, R, V, S, epsilon
  
def save_fig(outfile, files, fps = 5, loop = 1):
  """ Function to save GIFs """
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp = outfile, format = 'GIF', append_images = imgs[1:], save_all = True,
               duration = int(1000/fps), loop = loop)
