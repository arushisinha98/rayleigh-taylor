import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

## define & declare global variables
L_x = 2000 # domain size (x) = 2000 m
L_z = 2000 # domain size (z) = 2000 m
nx = 401 # 401 grid points in 0 <= x <= L_x
nz = 401 # 401 grid points in 0 <= z <= L_z
dx = L_x/(nx-1) # delta x
dz = L_z/(nz-1) # delta z
r0 = 250 # radius = 250 m
dt = 0.5 # delta t = 0.5 s
d_theta = 0.5 # delta theta = 0.5 K
theta_0 = 300 # theta 0 = 300 K
g = 9.80665 # gravitational acceleration = 9.80665 m/s^2
T_total = 1500 # T(total) = 1500 seconds
nt = T_total/dt
save_frequency = 50 # save solution every 50 iterations

x0 = L_x/2
z0 = 260

# artificial diffusion/viscosity
U = sqrt(r*r0*g*d_theta/theta_0)
Re = 1500
K_x = (2*r0*U)/Re
K_z = K_x

# formatting values
width = 14 # width of fixed-length data fields
precision = 6 # precision of data fields
# error bound values
tolerance = 0.0000001
max_iterations = 2000
# calculate alpha
sigma = (1/(1 + ((dx/d)^2)))*(cos(np.pi/nx) + ((dx/dz)^2)*cos(np.pi/nz))
alpha = 2/(1 + sqrt(1 - sigma^2))

# initialize potential temperature field

def save_fig(outfile, files, fps = 5, loop = 1):
  """ Function to save GIFs """
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp = outfile, format = 'GIF', append_images = imgs[1:], save_all = True,
               duration = int(1000/fps), loop = loop)


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

def Jacobian(f1, f2):
  J1 = 0.0
  J2 = 0.0
  J3 = 0.0
  J
