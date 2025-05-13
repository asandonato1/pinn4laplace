test repository for a simple physics informed neural network (PINN) to solve laplace's equation with dirichlet boundary conditions.

for pinn's, a "custom" loss function has to be implemented where L = a*L_NN + b * L_physics + c * L_boundary. a meta-learning approach was taken to determine the values of those coeffs (dynamicLambdas arg in train function/class def)
