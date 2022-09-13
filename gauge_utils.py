import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import torch
from pytope import Polytope

#%%%

# Gauge function
def gauge(F,G,V):
    # Batched gauge function (batch-second)
    # For each column v_i of V, computes the gauge of v_i with respect to the polytope {x: Fx <= g_i}, where g_i is a column of G.
    # Each polytope needs to contain the origin as an interior point, meaning that G must be > 0 elementwise. 
    assert (G>0).all()
    return torch.max(F@V/G,dim = 0).values

# Gauge map
def gauge_map(F,G,V):
    # Batched gauge map (batch-second)
    # For each column v_i of V, maps v_i from the infinity norm ball to the polytope {x: Fx <= g_i}, where g_i is a column of G.
    inf_norm_v = torch.linalg.norm(V,ord = np.inf,dim=0)
    assert (inf_norm_v < 1).all()
    return V * inf_norm_v / gauge(F,G,V)

if __name__ == "__main__":    
    n = 2
    m = 6
    p = 4
    F = torch.tensor([[1,1],[1,-1],[2,0]]).type(torch.FloatTensor)
    F = torch.cat((F,-F)) # Left hand side matrix
    g = torch.ones((m,1))
    G = g@torch.ones((1,p)) # Right hand side vector
    X = (torch.rand(n,p)-.5)*2 # Query points
    Y = gauge_map(F,G,X)
    
    plt.figure(figsize=(8,4))
    
    plt.subplot(121)
    F0 = torch.cat((torch.eye(n),-torch.eye(n)))
    g0 = torch.ones((n*2,1))
    G0 = g0 @ torch.ones((1,p))
    P0 = Polytope(F0.numpy(),g0.numpy())
    P0.plot(fill=False,edgecolor = (0,0,0),linewidth=2)
    Z = gauge(F0,G0,X)
    for i in range(p):
        (Z[i].numpy()*P0).plot(fill=False,edgecolor = (1-i/p,0,(i+1)/p))
    plt.plot(X[0,:],X[1,:],'o')
    plt.title('Input set')
    plt.plot(0,0,'*')
    
    plt.subplot(122)
    P1 = Polytope(F.numpy(),g.numpy())
    P1.plot(fill = False,edgecolor = (0,0,0),linewidth=2)
    plt.plot(Y[0,:],Y[1,:],'o')
    Z = gauge(F,G,Y)
    for i in range(p):
        (Z[i].numpy()*P1).plot(fill=False,edgecolor = (1-i/p,0,(i+1)/p))
    plt.title('Output set')
    plt.plot(0,0,'*')