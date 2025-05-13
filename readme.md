test repository for a simple physics informed neural network (PINN) to solve laplace's equation with dirichlet boundary conditions.

for pinn's, a "custom" loss function has to be implemented where 

```math
\mathcal{L}_{\text{Total}} = \lambda_1 \cdot \mathcal L_{\text{NN}} + \lambda_2 \cdot \mathcal L_{\text{Physics}} + \lambda_3 \cdot \mathcal L_{\text{Boundary}}
```

. a meta-learning approach was taken to determine the values of those coeffs (dynamicLambdas arg in train function/class def)
