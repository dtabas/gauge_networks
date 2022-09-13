import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from pytope import Polytope

#%%

# Method for computing an invariant set taken from:
# C. Liu and I. M. Jaimoukha, “The computation of full-complexity polytopic robust control invariant sets,” in Proceedings of the IEEE Conference on Decision and Control, pp. 6233–6238, 2015.

def generate_invariant_set(A,B,Bw,Vx,Vu,Vw,x_bar,u_bar,w_bar,m):

    n = np.shape(A)[0]
    
    Pr = (np.random.rand(m,n)-.5)
    Pr = Pr/np.linalg.norm(Pr,ord=2,axis = 1,keepdims=True)
    
    mu,nu = np.shape(Vu)
    mw,nw = np.shape(Vw)
    mx,nx = np.shape(Vx)
    
    br = np.ones((m,1))
    
    X_ = cp.Variable((n,n))
    K_ = cp.Variable((nu,n))
    Lam = cp.Variable((m,m),diag = True)
    R = cp.Variable((n,n),PSD=True)
    mui = cp.Variable(m,nonneg=True)
    
    D = []
    W = []
    constraints = []
    for i in range(m):
        D.append(cp.Variable((m,m),diag=True))
        W.append(cp.Variable((mw,mw),diag=True))
        ei = np.zeros((m,1))
        ei[i] = 1
        f10 = cp.bmat([[(2*ei.T@Lam@br - br.T@D[i]@br - w_bar.T@W[i]@w_bar)/2, ei.T@Lam@Pr, np.zeros((1,nw)), np.zeros((1,n))],
                                    [np.zeros((n,1)), (X_.T + X_)/2, Bw, A@X_ + B@K_],
                                    [np.zeros((nw,1)), np.zeros((nw,n)),(Vw.T@W[i]@Vw)/2, np.zeros((nw,n))],
                                    [np.zeros((n,1)),np.zeros((n,n)),np.zeros((n,nw)),(Pr.T@D[i]@Pr)/2]])
        constraints.append(f10 + f10.T >> 0)
        constraints.append(D[i] >> 0)
        constraints.append(W[i] >> 0)
        
        f16 = cp.bmat([[(2*ei.T@Lam@br - mui[i])/2, ei.T@Lam@Pr, np.zeros((1,n))],
                      [np.zeros((n,1)), (X_.T + X_)/2, R],
                      [np.zeros((n,1)),np.zeros((n,n)),(mui[i]*np.eye(n))/2]])
        constraints.append(f16 + f16.T >> 0)
        
    C = []
    for j in range(mx):
        C.append(cp.Variable((m,m),diag=True))
        ej = np.zeros((mx,1))
        ej[j] = 1
        f11 = cp.bmat([[(2*ej.T@x_bar - br.T@C[j]@br)/2, ej.T@Vx@X_],
                                    [np.zeros((n,1)),(Pr.T@C[j]@Pr)/2]])
        constraints.append(f11 + f11.T >> 0)
        constraints.append(C[j] >> 0)
        
    G = []
    for k in range(mu):
        G.append(cp.Variable((m,m),diag=True))
        ek = np.zeros((mu,1))
        ek[k] = 1
        f12 = cp.bmat([[(2*ek.T@u_bar - br.T@G[k]@br)/2, ek.T@Vu@K_],
                                    [np.zeros((n,1)),(Pr.T@G[k]@Pr)/2]])
        constraints.append(f12 + f12.T >> 0)
        constraints.append(G[k] >> 0)
    
    constraints.append(Lam >> 0)
    obj = cp.Maximize(cp.log_det(R)) # Maximize volume of inscribed ellipsoid
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)#solver=cp.MOSEK)
    #print([prob.status, prob.value])

    X = np.linalg.inv(X_.value)
    K = K_.value @ X
    P = Pr @ X
    b = br
    
    w_bar = np.reshape(w_bar,(-1,1))
    S_i = Polytope(A = np.block([[P],[-P]]),b = np.block([[b],[b]])) # invariant set
    W = Polytope(A = np.block([[Vw],[-Vw]]),b = np.block([[w_bar],[-w_bar]])) # disturbance set
    S_t = S_i - Bw*W # target set
    
    return S_t.A, S_t.b, S_i.A, S_i.b, K
    
if __name__ == '__main__':
    
    A = np.array([[1.,1],[0,1]])
    B = np.array([[0],[1.]])
    w_bar = np.eye(1)*.1
    Bw = np.array([[1.],[1]])
    Vx = np.eye(2)
    Vw = np.eye(1)
    Vu = np.eye(1)
    u_bar = np.eye(1)
    x_bar = np.array([[1.],[1.]])
    
    Pt,bt,Pr,br,K = generate_invariant_set(A,B,Bw,Vx,Vu,Vw,x_bar,u_bar,w_bar,10)
    
    S_i = Polytope(A = np.block([[Pr],[-Pr]]),b = np.block([[br],[br]]))
    S_i.plot(alpha = 0.5,color = (0,0,0),label='Invariant set')
    S_t = Polytope(A = np.block([[Pt],[-Pt]]),b = np.block([[bt],[bt]]))
    S_t.plot(alpha = 0.5,label='Target set')
    plt.autoscale(enable=True)
    
    lim = Polytope(lb = (-1,-1), ub = (1,1))
    lim.plot(fill=False, edgecolor = (1,0,0),label = 'Limits')
    plt.legend()
    
    