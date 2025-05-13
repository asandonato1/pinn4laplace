def fdmLaplace(nx, ny, lx=5, ly=5,tolerance=1e-5, maxIterations = 10000): # lx and ly are the "sizes" of our space. we're using dirichlet boundary conditions, too

    dx, dy = lx/(nx - 1), ly/(ny - 1)
    
    V = np.zeros((nx, ny))

    V[-1, :], V[0, :], V[:, 0], V[:, -1] = 0.0,0.0,0.0,1.0

    for iteration in range(maxIterations):

        oldV = V.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                V[i, j] = 0.25 * (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1]) # simple fdm for deltaX = deltaY

        if(np.max(np.abs(oldV - V)) < tolerance):
            print(f"Convergence stopped, breaking at {iteration}-th iteration.")
            break

    return V


def generate_points(nInterior=10000, n_boundary=400): # this functions is just used to generate points for the space. i changed the naming convention to make reading easier.
    # internal points
    x_in = np.random.rand(nInterior, 2)

    # checking boundary coordinates. notice that each x is a (x,y) type object

    x_b1 = np.column_stack((np.zeros(n_boundary), np.random.rand(n_boundary)))       # x = 0
    x_b2 = np.column_stack((np.ones(n_boundary), np.random.rand(n_boundary)))        # x = 1
    x_b3 = np.column_stack((np.random.rand(n_boundary), np.zeros(n_boundary)))       # y = 0
    x_b4 = np.column_stack((np.random.rand(n_boundary), np.ones(n_boundary)))        # y = 1

    x_b = np.vstack((x_b1, x_b2, x_b3, x_b4))

    # boundary values. it's easier to just take V = 0 at the boundaries.
    u_b = np.vstack((
        np.zeros((n_boundary, 1)),     # x = 0
        np.ones((n_boundary, 1)),      # x = 1
        np.zeros((n_boundary, 1)),     # y = 0
        np.zeros((n_boundary, 1))      # y = 1
    ))

    return x_in, x_b, u_b


