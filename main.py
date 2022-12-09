import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import interior_point
import torch
import torch.optim as optim
import time
from pytope import Polytope
import ray
'''import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
__package__ = parent_dir'''
from MPC_nets import gauge_NN, penalty_NN, projection_NN
from gauge_utils import gauge, gauge_map

#%%% Setup

# Online MPC controller:
def MPC(syst,x0):
    n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = syst.sys_params
    Mu,Mw,M0,F,g,H0,Hw = syst.MPC_params
    X0 = np.reshape(x0,(n,1))
    U = cp.Variable((m*tau,1))
    X = Mu@U + M0@X0
    constraints = [F@U <= g + H0@X0]
    objective = cp.Minimize(cp.norm(X)**2 + cp.norm(U)**2)
    prob = cp.Problem(objective,constraints)
    prob.solve(solver=cp.SCS)
    return U.value,prob.solver_stats.solve_time

def convert_to_tensor(params):
    for i,param in enumerate(params):
        params[i] = torch.tensor(param).type(torch.FloatTensor)
    return params
            
# Autoregressive noise
def generate_noise_traj(batch,M,n,tt,wb,noise_type):
    if noise_type == 'uniform':
        W = np.random.uniform(low=-wb,high=wb,size=(n*tt,batch*M))
    elif noise_type == 'autoregressive':
        noise_alpha = 0.1
        W = np.zeros((n*tt,batch*M))
        W[0:n,:] = noise_alpha*np.random.uniform(low=-wb,high=wb,size=(n,batch*M))
        for i in range(1,tt):
            W[n*i:n*(i+1),:] = noise_alpha*np.random.uniform(low=-wb,high=wb,size=(n,batch*M)) + (1-noise_alpha)*W[n*(i-1):n*i,:]
        W = W*3
        W = np.maximum(W,-wb)
        W = np.minimum(W,wb)
        '''a = np.max(W)
        b = np.min(W)
        c = np.maximum(np.abs(a),np.abs(b))
        W = W/c*wb'''
    else:
        W = None
        print('invalid noise type')
    return W,torch.tensor(W).type(torch.FloatTensor)


    

#%% Train NN:
            
def train(syst,net,batch,M,lr=1e-3,max_epoch=200,noise_type='uniform'):
              
    n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = syst.sys_params
    
    def step(i,batch,M):
        # batch = number of initial conditions
        # M = number of disturbance scenarios per initial condition
        V0_train_batch = (torch.rand((n,batch))-.5)*2
        X0_train_batch = gauge_map(torch.tensor(Pr).type(torch.FloatTensor),torch.tensor(br*.99).type(torch.FloatTensor)@torch.ones((1,batch)),V0_train_batch)
        W_train_batch = torch.zeros((n*tau,batch*M))
        optimizer.zero_grad()
        U_hat_batch = net(X0_train_batch)
        loss = net.loss_fn(U_hat_batch,X0_train_batch,W_train_batch)
        loss.backward()
        optimizer.step()
        net.loss_traj.append(loss.item())
        #print([i,loss.item()])
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #print('\n Training:')
    for i in range(max_epoch):
        step(i,batch,M)
        #if i*10 % max_epoch == 0: syst.stdout.write('.'); syst.stdout.flush();

#%% Test rollout:

def test_rollout(syst,net,plot=False,noise_type='uniform'):
            
    n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = syst.sys_params
    Q,R,QQ,RR = syst.cost_matrices
    
    w_traj = np.zeros((n,T))
    w_traj_numpy,w_traj_torch = generate_noise_traj(1,1,n,T,wb,noise_type)

    #X0_torch = (((torch.rand((n,1))-.5)*2)*x0b).type(torch.FloatTensor)
    X0_torch = gauge_map(torch.tensor(Pr).type(torch.FloatTensor),torch.tensor(br).type(torch.FloatTensor),(torch.rand((n,1))-.5)*2)
    X_traj_nn = X0_torch.detach().numpy()
    X_t_nn = X0_torch
    U_traj_nn = np.array([[] for i in range(m)])
    
    X0_numpy = X0_torch.numpy()
    X_traj_mpc = X0_numpy
    X_t_mpc = X0_numpy
    U_traj_mpc = np.array([[] for i in range(m)])
    
    total_time_nn = 0
    total_time_mpc = 0
    
    for t in range(T):
                
        t1 = time.time()
        U_t_nn = net(X_t_nn)[0:m]
        t2 = time.time()
        total_time_nn += t2-t1
        t3 = time.time()
        U_t_mpc,solve_time_mpc = MPC(syst,X_t_mpc)
        t4 = time.time()
        U_t_mpc = U_t_mpc[0:m]
        total_time_mpc += t4-t3 #solve_time_mpc

        w_t_numpy = w_traj_numpy[n*t:n*(t+1)] ##np.random.uniform(low=-wb,high=wb,size=(n,1))
        w_t_torch = w_traj_torch[n*t:n*(t+1)] ##torch.tensor(w_t_numpy).type(torch.FloatTensor)
            
        X_plus_nn = Ad @ X_t_nn + Bd @ U_t_nn.view((m,1)) + w_t_torch
        #X_plus_nn = torch.clamp(X_plus_nn,-x0b,x0b)
        X_plus_nn = X_plus_nn/torch.maximum(torch.tensor(1),gauge(Pr_torch,br_torch,X_plus_nn)) # clamp to invariant set (projection was having numerical difficulties)
        X_traj_nn = np.hstack((X_traj_nn,X_plus_nn.detach().numpy()))
        X_t_nn = X_plus_nn
        U_traj_nn = np.hstack((U_traj_nn,U_t_nn.detach().numpy()))
        
        X_plus_mpc = Ad.numpy() @ X_t_mpc + Bd.numpy() @ U_t_mpc.reshape((m,1)) + w_t_numpy
        #X_plus_mpc = np.clip(X_plus_mpc,-x0b,x0b)
        X_traj_mpc = np.hstack((X_traj_mpc,X_plus_mpc))
        X_t_mpc = X_plus_mpc
        U_traj_mpc = np.hstack((U_traj_mpc,U_t_mpc))
        
        w_traj[:,t] = w_t_numpy.flatten()
        
        #print(t)
                    
    avg_time_nn = total_time_nn/T
    avg_time_mpc = total_time_mpc/T
    
    #cost_nn = (np.linalg.norm(U_traj_nn,ord='fro')**2 + np.linalg.norm(X_traj_nn,ord='fro')**2)
    #cost_mpc = np.linalg.norm(U_traj_mpc,ord='fro')**2 + np.linalg.norm(X_traj_mpc,ord='fro')**2
    
    cost_nn = np.trace(X_traj_nn.T@Q@X_traj_nn) + np.trace(U_traj_nn.T@R@U_traj_nn)
    cost_mpc = np.trace(X_traj_mpc.T@Q@X_traj_mpc) + np.trace(U_traj_mpc.T@R@U_traj_mpc)
    '''xnn = np.reshape(X_traj_nn,(n*(T+1),1),'F')
    unn = np.reshape(U_traj_nn,(m*(T),1),'F')
    xmpc = np.reshape(X_traj_mpc,(n*(T+1),1),'F')
    umpc = np.reshape(U_traj_mpc,(m*(T),1),'F')
    cost2_nn = xnn.T@QQ@xnn + unn.T@RR@unn
    cost2_mpc = xmpc.T@QQ@xmpc + umpc.T@RR@umpc
    print(cost_nn,cost2_nn)
    print(cost_mpc,cost2_mpc)'''

    #frac_subopt = np.round((cost_nn.detach().numpy() - cost_mpc)/cost_mpc,2)
    frac_subopt = (cost_nn - cost_mpc)/cost_mpc
    
    if plot:
        plt.figure()
        plt.step(range(T+1),X_traj_mpc[1,:].T,label = 'MPC')
        plt.step(range(T+1),X_traj_nn[1,:].T,label = 'NN')
        plt.title('State')
        plt.legend()
        
        plt.figure()
        plt.step(range(T),U_traj_mpc[0,:].T,label = 'MPC')
        plt.step(range(T),U_traj_nn[0,:].T,label = 'NN')
        plt.title('Control')
        plt.legend()
        
        if n == 2:
            plt.figure()
            Polytope(Pr,br).plot(fill=False,edgecolor=(0,0,0))
            plt.plot(X_traj_mpc[0,:],X_traj_mpc[1,:])
            plt.plot(X_traj_nn[0,:],X_traj_nn[1,:])
    
    return frac_subopt,avg_time_nn,avg_time_mpc,cost_nn,cost_mpc

#print(f'fraction suboptimality ((nn-mpc)/mpc): {frac_subopt}')



#%% Test and generate statistics:
    
N_test_standard = 20    

def test(syst,net,noise_type,N_test=N_test_standard):
    result_list = []
    #print('\n Testing:')
    for i in range(N_test):
        result = test_rollout(syst,net,plot=False,noise_type=noise_type)
        result_list.append(result)
        #print(i)
        #syst.stdout.write('.'); syst.stdout.flush();
    result_array = np.array(result_list)
    return result_array

@ray.remote
def test_once(syst,net,noise_type):
    result = test_rollout(syst,net,plot=False,noise_type=noise_type)
    return result

def test_parallel(syst,net,noise_type,N_test=N_test_standard):
    print('Testing')
    output_ids = []
    for i in range(N_test):
        output_ids.append(test_once.remote(syst,net,noise_type))
    output_list = ray.get(output_ids)
    result_array = np.array(output_list)
    return result_array

#%% Tune hyperparameters:

@ray.remote
def test_one_hyperparameter(syst,net_type,noise_type,i):
    # For first iteration, select "good" hyperparameters
    if i == 0:
        hidden_dim = 2**8
        lr = 1e-4
        M = 1
        batch = 1000
        
    elif i > 0:
        # Randomly generate hypeparameters:
        hidden_dim = int(2**np.random.uniform(low=6,high=10))
        lr = 10**np.random.uniform(low=-5,high=-3)
        M = 1
        batch = int(10**np.random.uniform(low=2,high=3.5))
    
    N_epoch = 10
    n_epoch = 10
    if net_type == 'gauge':
        net = gauge_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    elif net_type == 'penalty':
        net = penalty_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    elif net_type == 'projection':
        net = projection_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    else:
        return
    train(syst,net,batch,M,lr=lr,max_epoch=max_epoch,noise_type=noise_type)
    
    result = test(syst,net,noise_type)
    performance = np.mean(result,axis=0)[0]
    a = [i,hidden_dim,lr,batch,M,performance]
    
    print(net_type + ' hparam iter ' + str(i) + ' complete')
    
    return [a,net]

def tune_hyperparameters_parallel(syst,net_type = 'gauge'):
    
    aa = []
    net_list = []
    noise_type = 'autoregressive'
    if net_type == 'projection':
        N_search = 8
    elif net_type == 'penalty':
        N_search = 3
    elif net_type == 'gauge':
        N_search = 32
    output_ids = []
    for i in range(N_search):
        output_ids.append(test_one_hyperparameter.remote(syst,net_type,noise_type,i))
    output_list = ray.get(output_ids)
    a3 = np.array([z[0] for z in output_list])
    net_list = [z[1] for z in output_list]
    ind = np.argmin(a3[:,-1])
    best_net = net_list[ind]
    top_performance = a3[ind][-1]
    print(aa)
    return best_net,top_performance

#%% Open-loop validation

def validate_OL(syst,net):
    
    n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = syst.sys_params
    
    val_batch = 100
    
    V0_val = (torch.rand((n,val_batch))-.5)*2
    X0_val = gauge_map(Pr_torch,(br_torch*.99)@torch.ones((1,val_batch)),V0_val)
    U_val = net(X0_val).detach().numpy()
    U_opt = np.zeros((m*tau,val_batch))
    for i,x0 in enumerate(list(X0_val.T)):
        U_opt[:,i] = (MPC(syst,x0)[0]).flatten()
        
    # Compute the state trajectories:
    Mu,Mw,M0,F,g,H0,Hw = syst.MPC_params
    X_val = Mu@U_val + M0@X0_val
    X_opt = Mu@U_opt + M0@X0_val
        
    # Compute the trajectory costs:
    c_val = (np.linalg.norm(X_val,ord='fro')**2 + np.linalg.norm(U_val,ord='fro')**2)/val_batch
    c_opt = (np.linalg.norm(X_opt,ord='fro')**2 + np.linalg.norm(U_opt,ord='fro')**2)/val_batch
        
    # Compare NN to MPC:
    print(c_val)
    print(c_opt)
    val_score = (c_val-c_opt)/c_opt
    
    return val_score

@ray.remote
def test_one_hp_OL(syst,net_type,i):
    # For first iteration, select "good" hyperparameters
    if i == 0:
        hidden_dim = 2**8
        lr = 1e-4
        M = 1
        batch = 1000
        
    elif i > 0:
        # Randomly generate hypeparameters:
        hidden_dim = int(2**np.random.uniform(low=6,high=10))
        lr = 10**np.random.uniform(low=-5,high=-3)
        M = 1
        batch = int(10**np.random.uniform(low=2,high=3.5))
    
    N_epoch = 5
    n_epoch = 1
    if net_type == 'gauge':
        net = gauge_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    elif net_type == 'penalty':
        net = penalty_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    elif net_type == 'projection':
        net = projection_NN(syst,hidden_dim)
        max_epoch = n_epoch*N_epoch
    else:
        return
    train(syst,net,batch,M,lr=lr,max_epoch=max_epoch,noise_type='autoregressive')
    
    performance = validate_OL(syst,net)
    a = [i,hidden_dim,lr,batch,M,performance]
    
    print(net_type + ' hparam iter ' + str(i) + ' complete')
    
    return [a,net]


def tune_hyperparameters_parallel_OL(syst,net_type):
    aa = []
    net_list = []
    if net_type == 'projection':
        N_search =16
    elif net_type == 'penalty':
        N_search = 16
    elif net_type == 'gauge':
        N_search = 16
    output_ids = []
    for i in range(N_search):
        output_ids.append(test_one_hp_OL.remote(syst,net_type,i))
    output_list = ray.get(output_ids)
    a3 = np.array([z[0] for z in output_list])
    net_list = [z[1] for z in output_list]
    ind = np.argmin(a3[:,-1])
    best_net = net_list[ind]
    top_performance = a3[ind][-1]
    print(aa)
    return best_net,top_performance,a3[ind]
    
#%%

class mpc_model(object):
    
    def __init__(self,tau):
        
        # Get parameters:
        sys_params = interior_point.generate_sys_params(tau=tau)
        MPC_params = interior_point.generate_MPC_params(sys_params)
        IP_params = interior_point.IP_affine(sys_params,MPC_params)
        IP_matrices = interior_point.generate_M_ip(sys_params,IP_params[0],IP_params[1])
        
        # Convert parameters to tensor and repackage as needed:
        tau,n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,J,j,X0,Pt,bt,Pr,br,K_invariant = sys_params 
        Ad,Bd,u_bar,Pr_torch,br_torch = convert_to_tensor([Ad,Bd,u_bar,Pr,br])
        sys_params = (n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch)
        MPC_params = convert_to_tensor(list(MPC_params))
        IP_matrices = convert_to_tensor(list(IP_matrices))
        
        # Cost matrices:
        Q = np.eye(n)
        R = np.eye(m)
        QQ = torch.block_diag(*[torch.tensor(Q).type(torch.FloatTensor)]*tau)
        RR = torch.block_diag(*[torch.tensor(R).type(torch.FloatTensor)]*tau)
        cost_matrices = (Q,R,QQ,RR)
        
        # Set attributes:
        self.tau = tau
        self.sys_params = sys_params
        self.MPC_params = MPC_params
        self.IP_params = IP_params
        self.IP_matrices = IP_matrices
        self.cost_matrices = cost_matrices
        
        # How to extract parameters:
        # tau = syst.tau
        #n,m,wb,x0b,x0_bar,Ad,Bd,r,u_bar,x_bar,Pt,bt,Pr,br,Pr_torch,br_torch = syst.sys_params
        #Mu,Mw,M0,F,g,H0,Hw = syst.MPC_params
        #W,w,Fp2s,gp2s = syst.IP_params
        #K_weight,K_bias = syst.IP_matrices
        #Q,R,QQ,RR = syst.cost_matrices
        
#%%

if __name__ == '__main__':

    c = 100
    T = 50
    tau = 5
    net_list = []
    syst = mpc_model(tau)        
        
    #%% Closed-loop simulations:
    
    performance_plot = []
    time_plot = []
    best_net_list = []
    nets = ['gauge','projection'] 
    #nets = ['gauge','penalty','projection']
    for net_type in nets:
        best_net,cost = tune_hyperparameters_parallel(syst,net_type=net_type)
        result = test_parallel(syst,best_net,noise_type='autoregressive',N_test=100)
        performance_plot.append(result[:,3])
        time_plot.append(result[:,1])
        best_net_list.append(best_net)
    performance_plot.append(result[:,4])
    time_plot.append(result[:,2])
    net_list.append(best_net_list)
    
    #%% Plot results
    
    # Pareto plot:
    time_mean = [np.mean(z) for z in time_plot]
    time_std = [np.std(z) for z in time_plot]
    performance_mean = [np.mean(z) for z in performance_plot]
    performance_std = [np.std(z) for z in performance_plot]
    
    plt.figure(c+10,figsize=(6,4),dpi=1000)
    ax = plt.gca()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color = colors[0:4]
    #net_legend = ['Gauge NN','Penalty NN','Projection NN','Online MPC']
    net_legend = ['Gauge NN','Projection NN','Online MPC']

    box1 = plt.boxplot(performance_plot,vert=False,positions = time_mean,patch_artist=True,medianprops=dict(linewidth=0),notch=True,manage_ticks=False,widths = (max(time_mean)-min(time_mean)/len(net_legend))/7.5,showfliers=False)
    for patch, color in zip(box1['boxes'], color):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.legend(box1["boxes"], net_legend,fontsize=15,loc='upper right')
    plt.xlabel('Trajectory cost',font='Times',fontsize=20)
    plt.ylabel('Average solve time (sec)',font='Times',fontsize=20)
    plt.xticks(font='Times',fontsize=15)
    plt.yticks(font='Times',fontsize=15)
    #plt.ylim([0,max(time_mean) + min(time_mean)])
    plt.tight_layout()
    plt.grid()
    plt.savefig('pareto.png')
    
    # Training trajectories - no zoom:
    plt.figure(c+2000,figsize=(6,4),dpi=1000)
    ax = plt.gca()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color = colors[0:4]
    for i,this_net in enumerate(best_net_list):
        loss = np.array(this_net.loss_traj)
        loss_initial = loss[0]
        loss_initial_rounded = np.round(loss_initial,1)
        loss_normalized = loss/loss_initial
        #plt.plot(loss_normalized,color[i],label=net_legend[i] + f' ({loss_initial_rounded})')
        plt.semilogy(loss,color[i],label=net_legend[i])
    plt.legend(fontsize=15,loc='upper right')
    plt.xticks(font='Times',fontsize=15)
    plt.yticks(font='Times',fontsize=15)
    plt.xlabel('Iteration',font='Times',fontsize=20)
    plt.ylabel('Training Loss',font='Times',fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.savefig('loss.png')
    
    c += 1
    
    #%% Open-loop experiments
    
    '''results = []
    nets = ['gauge','penalty','projection']
    #nets = ['gauge','penalty']
    for net_type in nets:
        best_net,cost,hps = tune_hyperparameters_parallel_OL(syst,net_type=net_type)
        performance = validate_OL(syst,best_net)
        results.append([net_type,best_net,cost,hps,performance])'''
    