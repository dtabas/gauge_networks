import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
'''import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
__package__ = parent_dir'''
from gauge_utils import gauge_map

#%%%

class gauge_NN(nn.Module):
    
    def __init__(self,sys,hidden_dim):
        
        super(gauge_NN,self).__init__()
        self.tau = sys.tau
        self.sys = sys
        n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = self.sys.sys_params
        self.input_dim = n
        self.hidden_dim = hidden_dim
        self.output_dim = m*self.tau
        self.loss_traj = []
        
        self.model = nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.output_dim))
    
    def loss_fn(self,U,X0,W):
        Mu,Mw,M0,F,g,H0,Hw = self.sys.MPC_params
        Q,R,QQ,RR = self.sys.cost_matrices
        batch = U.size()[1]
        M = int(W.size()[1]/batch)
        X = Mu@U.repeat_interleave(M,dim=1) + Mw@W + M0@X0.repeat_interleave(M,dim=1)
        #return (torch.linalg.norm(X,ord='fro')**2 + torch.linalg.norm(U,ord='fro')**2)/batch
        return torch.trace(X.T@QQ@X)/(batch*M) + torch.trace(U.T@RR@U)/batch
    
    def forward(self,X0):
        Mu,Mw,M0,F,g,H0,Hw = self.sys.MPC_params
        Q,R,QQ,RR = self.sys.cost_matrices
        K_weight,K_bias = self.sys.IP_matrices
        batch = X0.size()[1] # batch size
        o = torch.ones((1,batch)) # row vector of ones
        U_ip = K_weight @ X0 + K_bias @ o # Interior point from linear feedback
        V = torch.tanh(self.model(X0.T)).T # Relative prediction from neural net
        G = g @ o + H0 @ X0 - F @ U_ip # RHS of shifted inequalities
        U = gauge_map(F,G,V) + U_ip # Apply gauge map and shift
        return U
    
class penalty_NN(nn.Module):
    
    def __init__(self,sys,hidden_dim):
        
        super(penalty_NN,self).__init__()
        self.tau = sys.tau
        self.sys = sys
        n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = self.sys.sys_params
        self.input_dim = n
        self.hidden_dim = hidden_dim
        self.output_dim = m*self.tau
        self.loss_traj = []
        
        self.model = nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.output_dim))
    
    def loss_fn(self,U,X0,W):
        Mu,Mw,M0,F,g,H0,Hw = self.sys.MPC_params
        Q,R,QQ,RR = self.sys.cost_matrices
        batch = U.size()[1]
        M = int(W.size()[1]/batch)
        o = torch.ones((1,batch*M)) # row vector of ones
        G = g @ o + H0 @ X0.repeat_interleave(M,dim=1) + Hw@W # RHS of inequalities
        X = Mu@U.repeat_interleave(M,dim=1) + Mw@W + M0@X0.repeat_interleave(M,dim=1)
        alpha = 100 # Constraint violation penalty coefficient
        cvs = F@U.repeat_interleave(M,dim=1) - G # Constraint violations or residuals, Fx-g (use relu to only penalize when >= 0)
        #return (torch.linalg.norm(X,ord='fro')**2 + torch.linalg.norm(U,ord='fro')**2 + alpha*torch.sum(torch.relu(cvs)))/batch
        return torch.trace(X.T@QQ@X)/(batch*M) + torch.trace(U.T@RR@U)/batch + alpha*torch.sum(torch.relu(cvs))/(batch*M)
    
    def forward(self,X0):
        n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = self.sys.sys_params
        batch = X0.size()[1]
        V = torch.tanh(self.model(X0.T)).T # Relative prediction from neural net
        U = V * u_bar.repeat(self.tau,batch) # Scale hypercube to u_bar
        return U
    
class projection_NN(nn.Module):
    
    def __init__(self,sys,hidden_dim):
        
        super(projection_NN,self).__init__()
        self.tau = sys.tau
        self.sys = sys
        n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = self.sys.sys_params
        Mu,Mw,M0,F,g,H0,Hw = self.sys.MPC_params
        self.input_dim = n
        self.hidden_dim = hidden_dim
        self.output_dim = m*self.tau
        self.loss_traj = []
        
        self.model = nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim,self.output_dim))
    
        v = cp.Parameter((m*self.tau))
        x0_cp = cp.Parameter((n))
        u = cp.Variable((m*self.tau))
        obj = cp.Minimize(cp.norm(u-v)**2)
        constraints = [F@u <= g.flatten() + H0@x0_cp]
        prob = cp.Problem(obj,constraints)
        self.projection = CvxpyLayer(prob, parameters = [v,x0_cp], variables = [u])
    
    def loss_fn(self,U,X0,W):
        Mu,Mw,M0,F,g,H0,Hw = self.sys.MPC_params
        Q,R,QQ,RR = self.sys.cost_matrices
        batch = U.size()[1]
        M = int(W.size()[1]/batch)
        X = Mu@U.repeat_interleave(M,dim=1) + Mw@W + M0@X0.repeat_interleave(M,dim=1)

        #return (torch.linalg.norm(X,ord='fro')**2 + torch.linalg.norm(U,ord='fro')**2)/batch
        return torch.trace(X.T@QQ@X)/(batch*M) + torch.trace(U.T@RR@U)/batch
    
    def forward(self,X0):
        n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = self.sys.sys_params
        batch = X0.size()[1]
        V = torch.tanh(self.model(X0.T)) * (u_bar.repeat(self.tau,batch)).T
        U, = self.projection(V,X0.T)
        return U.T