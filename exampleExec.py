import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklean.model_selction import train_test_split

from fdmGenPoints import fdmLaplace, generate_points
from pinnUtilities import LaplaceModel, train, evaluate

def main():
    
    lx, ly = 1, 1
    nx, ny = 100, 100
    potential = fdmLaplace(nx, ny, lx, ly)


    model = LaplaceModel()
    x_in, x_b, borderPotential = generate_points()  # generates the points 
    uInside = np.zeros((len(x_in), 1))  # potential inside the region
    trained_model = train(model, x_in, x_b, borderPotential, uInside, epochs = 10000, dynamicLambda = True)
    xx, yy, uu = evaluate(trained_model)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))

    im1 = ax1.contourf(xx, yy, uu, levels=50, cmap="viridis")
    ax1.set_title(" $∇^2φ= 0$ using a Physics-Informed Neural Network")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    fig.colorbar(im1, ax = ax1, label="φ(x,y)")

    im2 = ax2.contourf(np.linspace(0, 1, 100),
                 np.linspace(0, 1, 100),
                 potential,
                 levels=50,
                 cmap="viridis")

    ax2.set_title("Numerical simulation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax = ax2, label="φ(x,y)")

    plt.tight_layout()
    plt.show()
