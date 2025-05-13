class LaplaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( # this is equivalent to writing the neural network with all the fullyConnected's and tanh activation functions in the forward method
            nn.Linear(2, 64),
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.lambdaPhysics = nn.Parameter(torch.tensor(1.0)) # initializes the lambdas as unitary, as we assume they should have the same weight
        self.lambdaBoundary = nn.Parameter(torch.tensor(1.0))
        self.lambdaNNerror = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.current_x = x.clone().detach().requires_grad_(True) # this is required to check the forward pass/loss computation order 
        x = self.current_x

        return self.net(x)

    def pinnLoss(self, outputs, targets, dynamicLambda = False): # here, we define the pinnLoss method.
        if self.current_x is None:
            raise ValueError("Forward pass must be called before loss computation")
        
        # simple MSE for NN loss
        mseLoss = nn.MSELoss()(outputs, targets)

        if not torch.is_grad_enabled():
            return mseLoss

        # calculating the laplacian to calculate the physics loss
        x = self.current_x
        u = outputs # potential 

        grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad[:, 0:1]
        u_y = grad[:, 1:2]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        laplacian = u_xx + u_yy
        physicsLoss = torch.mean(laplacian**2)

        # to calculate the boundary loss, we use masking to discriminate the "borders" of our space
        eps = 1e-8 # this tolerance stops some weird NaN's from appearing 
        boundary_mask = (
            (x[:, 0] <= eps) | (x[:, 0] >= 1.0 - eps) |
            (x[:, 1] <= eps) | (x[:, 1] >= 1.0 - eps)
        )

        if boundary_mask.any(): 
            boundaryLoss = nn.MSELoss()(outputs[boundary_mask], targets[boundary_mask])
        else:
            boundaryLoss = torch.tensor(0.0, device=outputs.device)
        
        return (
                1.0 * mseLoss +
                1.0 * physicsLoss +
                1.0 * boundaryLoss
            ) if dynamicLambda == False else (
            
            self.lambdaNNerror * mseLoss + 
            self.lambdaPhysics * physicsLoss + 
            self.lambdaBoundary * boundaryLoss
            
            ) # this implements the option of dynamicLambdas




def train(model, x_in, x_b, borderPotential, interiorPotential, epochs=5000, lrParams=1e-3, lrLambdas=1e-4, dynamicLambda = False): # the train function needed some heavy tweaks too
    xAll = torch.tensor(np.vstack((x_in, x_b)), dtype=torch.float32)   # all of the points
    yAll = torch.tensor(np.vstack((interiorPotential, borderPotential)), dtype=torch.float32)  # all of the potentials


    # here, we discriminate between neural-network parameters and lambda values for custom loss functions

    paramsNN = [p for n, p in model.named_parameters() if not n.startswith('lambda')]
    paramsLambda = [p for n, p in model.named_parameters() if n.startswith('lambda')]

    optimizer = torch.optim.Adam([
        {'params': paramsNN, 'lr': lrParams},
        {'params' : paramsLambda, 'lr' : lrLambdas}

    ]) if dynamicLambda != False else torch.optim.Adam([{'params': paramsNN, 'lr': lrParams}]) # we can choose to not meta-learn the lambdas by setting dynamicLambda = False (default)

    for epoch in range(epochs):
        optimizer.zero_grad()

        yPred = model(xAll)

        loss = model.pinnLoss(yPred, yAll, dynamicLambda)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} \t Loss: {loss.item():.4f} \t Lambda_NN: {model.lambdaNNerror.item():.4f} \t Lambda_Phys: {model.lambdaPhysics.item():.4f} \t Lambda_BC: {model.lambdaBoundary.item():.4f}")

    return model # returns trained model

def evaluate(model, res=1000): # simple function to evaluate the model using a simple grid from 0,1 with res points.
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    pts_tensor = torch.tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        u = model(pts_tensor).numpy().reshape((res, res))
    return xx, yy, u

