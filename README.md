# rayleigh-taylor

An attempt to implement a physics-informed neural network (PINN) to simulate a Rayleigh-Taylor instability, the perturbed interface of two fluids with different densities. As there is no exact solution to this system of two non-linear partial differential equations (PDEs), the ground truth is provided by a space-time discretized simulation of this instability and the physics-informed cost function includes the residuals of the PDEs and a discretized Poisson solver.

<a href = "https://safe.menlosecurity.com/https://www.sciencedirect.com/science/article/pii/S0021999118307125"> Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations <a> (Raissi et al., 2019) was used as a reference to help formulate the problem set-up.
