drive_dir = "." #"drive/MyDrive/CFRM/RL/SingleAgent-Stage2"

import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import datetime
import pytz, os

## Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = "cpu"
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = "cuda"

## Regimes
N_AGENT = 2
COST_POWER = 1.5

## Global Constants
S_VAL = 1 #245714618646 #1#

if COST_POWER == 2:
    TR = 0.2
    if N_AGENT == 10:
        XI_LIST = torch.tensor([-2.89, -1.49, -1.18, 1.4, 1.91, 2.7, -2.22, -3.15, 2.63, 2.29]).float() * (-10)
        GAMMA_LIST = torch.tensor([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]).float().to(device = DEVICE)
    elif N_AGENT == 5:
        XI_LIST = torch.tensor([-3, -2, -2, 3, 4]).float() * (-1)
        GAMMA_LIST = torch.tensor([1, 1.2, 1.4, 1.6, 1.8]).float().to(device = DEVICE)
else:
    if N_AGENT == 2:
        TR = 0.4
        XI_LIST = torch.tensor([3, -3]).float()
        GAMMA_LIST = torch.tensor([1, 2]).float().to(device = DEVICE)
    elif N_AGENT == 5:
        TR = 0.2
        XI_LIST = torch.tensor([-3, -2, -2, 3, 4]).float() * (-1)
        GAMMA_LIST = torch.tensor([1, 1.2, 1.4, 1.6, 1.8]).float().to(device = DEVICE)
    else:
        TR = 0.2
        XI_LIST = torch.tensor([-2.89, -1.49, -1.18, 1.4, 1.91, 2.7, -2.22, -3.15, 2.63, 2.29]).float() * (-10)
        GAMMA_LIST = torch.tensor([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]).float().to(device = DEVICE)
# TR = 0.2 #0.2 #0.2 for quad cost 2 agents and 0.4 for 1.5 cost 2 agents #20
T = 100
TIMESTAMPS = np.linspace(0, TR, T + 1)[:-1]
DT = TR / T
N_SAMPLE = 3000 #500 #128 #128
ALPHA = 1 #1 #
BETA = 2 #0.3 #0.5
# GAMMA_BAR = 8.30864e-14 * S_VAL
# KAPPA = 2.

# GAMMA_1 = GAMMA_BAR*(KAPPA+1)/KAPPA
# GAMMA_2 = GAMMA_BAR*(KAPPA+1)
GAMMA_1 = 1
GAMMA_2 = 2

# XI_LIST = torch.tensor([3, -3]).float()
# GAMMA_LIST = torch.tensor([GAMMA_1, GAMMA_2]).float().to(device = DEVICE)

# XI_LIST = torch.tensor([-2.89, -1.49, -1.18, 1.4, 1.91, 2.7, -2.22, -3.15, 2.63, 2.29]).float() * (-10) #torch.tensor([3, -3]).float() #torch.tensor([-3, -2, -2, 3, 4]).float() * (-1) #torch.tensor([3.01, 2.92, -2.86, 3.14, 2.90, -3.12, -2.88, 2.90, -2.93, -3.08]).float() #torch.tensor([3, -2, 2, -3]).float() #
# GAMMA_LIST = torch.tensor([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]).float().to(device = DEVICE) #torch.tensor([GAMMA_1, GAMMA_2]).float().to(device = DEVICE) #torch.tensor([1, 1.2, 1.4, 1.6, 1.8]).float().to(device = DEVICE) # #torch.tensor([1, 1, 1.3, 1.3, 1.6, 1.6, 1.9, 1.9, 2.2, 2.2]).float().to(device = DEVICE) #torch.tensor([1, 1, 2, 2]).float().to(device = DEVICE) #

XI_NORM_LIST = (torch.max(torch.abs(XI_LIST)) / torch.abs(XI_LIST)) ** 2

S = 1
## 1e-2: power 2 10 agents, power 1.5 2 agents
## 1e-3: power 2 2 agents
LAM = 1e-2 #1e-2 for 10 agents #1.08102e-10 * S_VAL #0.1 #

S_TERMINAL = 1 #1/3 #245.47
S_INITIAL = 0 #0 #250 #0#

assert len(XI_LIST) == len(GAMMA_LIST) and torch.max(GAMMA_LIST) == GAMMA_LIST[-1]

GAMMA_BAR = 1 / torch.sum(1 / GAMMA_LIST)
GAMMA_MAX = torch.max(GAMMA_LIST)
N_AGENT = len(XI_LIST)
# BETA = GAMMA_BAR*S*ALPHA**2 + S_TERMINAL/TR

## Setup Numpy Counterparts
GAMMA_LIST_NP = GAMMA_LIST.cpu().numpy().reshape((1, N_AGENT))
XI_LIST_NP = XI_LIST.numpy().reshape((1, N_AGENT))
GAMMA_BAR_NP = GAMMA_BAR.cpu().numpy()
GAMMA_MAX_NP = GAMMA_MAX.cpu().numpy()
###

## Load G
with open("eva.txt", "r") as f:
    G_MAP = torch.tensor([float(x.strip()) for x in f.readlines()]).to(device=DEVICE)

def get_W(dW_st, W_s0 = None):
    n_sample = dW_st.shape[0]
    if W_s0 is None:
        W_s0 = torch.zeros((n_sample, 1))
    else:
        W_s0 = W_s0.reshape((n_sample, 1)).cpu()
    W_st = torch.cumsum(torch.cat((W_s0, dW_st), dim=1), dim=1)
    return W_st.to(device = DEVICE)

## Get ground truth sigma
def InverseRiccati(t, R, LAM=LAM, GAMMA_BAR_NP=GAMMA_BAR_NP, GAMMA_LIST_NP=GAMMA_LIST_NP, GAMMA_MAX_NP=GAMMA_MAX_NP, ALPHA=ALPHA, N_AGENT=N_AGENT, XI_LIST_NP=XI_LIST_NP):
    RH = R[:N_AGENT]
    RF = R[N_AGENT:].reshape((N_AGENT, N_AGENT))
    const = (ALPHA + GAMMA_BAR_NP * np.sum((1 / GAMMA_LIST_NP[:,:-1] - 1 / GAMMA_MAX_NP) * RH[:-1]))
    dRH = np.zeros(N_AGENT)
    dRF = np.zeros((N_AGENT, N_AGENT))
    for n in range(N_AGENT - 1):
        ind = np.zeros(N_AGENT)
        ind[n] = 1
        dRH[n] = np.sum((GAMMA_LIST_NP[:,:-1] * (N_AGENT * ind[:-1] - 1) + GAMMA_MAX_NP) * XI_LIST_NP[:,:-1]) * const / N_AGENT - np.matmul(RF[n,:-1], RH[:-1].reshape((N_AGENT - 1, 1))) / LAM
    for n in range(N_AGENT - 1):
        for m in range(N_AGENT - 1):
            if n == m:
                ind = N_AGENT - 1
            else:
                ind = -1
            dRF[n, m] = (GAMMA_LIST_NP[:, m] * ind + GAMMA_MAX_NP) * const ** 2 / N_AGENT - np.matmul(RF[n,:-1], RF[:-1,m]) / LAM
    dR = np.hstack((dRH.reshape((-1,)), dRF.reshape((-1,))))
    return dR

def get_FH_exact():
    R_0 = np.zeros(N_AGENT * (N_AGENT + 1))
    timestamps = np.linspace(0, TR, T + 1)[:-1]
    res = solve_ivp(lambda t,R: InverseRiccati(t,R,LAM=LAM, GAMMA_BAR_NP=GAMMA_BAR_NP, GAMMA_LIST_NP=GAMMA_LIST_NP, GAMMA_MAX_NP=GAMMA_MAX_NP, ALPHA=ALPHA, N_AGENT=N_AGENT, XI_LIST_NP=XI_LIST_NP), t_span=[0, TR], y0=R_0, t_eval=timestamps, rtol=1e-5, atol=1e-8)
    RH = torch.tensor(res.y[:N_AGENT]).to(device = DEVICE)
    RF = torch.tensor(res.y[N_AGENT:]).to(device = DEVICE)
    F_exact, H_exact = RF.float(), RH.float()
    F_exact = torch.flip(F_exact,[1])
    H_exact = torch.flip(H_exact,[1])
    return F_exact, H_exact

## Get ground truth mu given sigma
def get_mu_from_sigma(sigma_st, phi_stn, W_st):
    assert phi_stn.shape[1] == sigma_st.shape[1]
    assert W_st.shape[1] == sigma_st.shape[1]
    T = phi_stn.shape[1]
    n_sample = phi_stn.shape[0]
    #sigma_st = torch.ones((n_sample, 1)).to(device = DEVICE) @ sigma_t.reshape((1, T))
    mu_st = torch.zeros((n_sample, T)).to(device = DEVICE)
    for n in range(N_AGENT):
        mu_st += 1 / N_AGENT * GAMMA_LIST[n] * sigma_st * (sigma_st * phi_stn.clone()[:,:,n] + XI_LIST[n] * W_st)
    return mu_st

## Training
class S_0(nn.Module):
    def __init__(self, s_init = S_INITIAL):
        super(S_0, self).__init__()
        self.s_0 = nn.Linear(1, 1)
        torch.nn.init.constant_(self.s_0.weight, s_init)
  
    def forward(self, x):
        return self.s_0(x)

class Net(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM_LST, OUTPUT_DIM=1):
        super(Net, self).__init__()
        self.layer_lst = nn.ModuleList()
#        self.bn = nn.ModuleList()

        self.layer_lst.append(nn.Linear(INPUT_DIM, HIDDEN_DIM_LST[0]))
#        self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[0],momentum=0.1))
        for i in range(1, len(HIDDEN_DIM_LST)):
            self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[i - 1], HIDDEN_DIM_LST[i]))
#            self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[i],momentum=0.1))
        self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[-1], OUTPUT_DIM))

    def forward(self, x):
        for i in range(len(self.layer_lst) - 1):
            x = self.layer_lst[i](x)
#            x = self.bn[i](x)
            # x = F.relu(x)
            x = torch.tanh(x)
        return self.layer_lst[-1](x)

## Model wrapper
class ModelFull(nn.Module):
    def __init__(self, predefined_model, is_discretized = False):
        super(ModelFull, self).__init__()
        self.model = predefined_model
        self.is_discretized = is_discretized
    
    def forward(self, tup):
        t, x = tup
        if self.is_discretized:
            return self.model[t](x)
        else:
            return self.model(x)

## Construct arbitrary neural network models with optimizer and scheduler
class ModelFactory:
    def __init__(self, time_len, algo, input_dim, hidden_lst, output_dim, lr, decay, scheduler_step, use_s0 = False, solver = "Adam", retrain = False, constant_len = 0, drive_dir = "."):
        assert solver in ["Adam", "SGD", "RMSprop"]
        assert algo in ["generator", "discriminator", "combo"]
        self.lr = lr
        self.decay = decay
        self.scheduler_step = scheduler_step
        self.solver = solver
        self.input_dim = input_dim
        self.hidden_lst = hidden_lst
        self.output_dim = output_dim
        self.model = None
        self.prev_ts = None
        self.algo = algo
        self.time_len = time_len
        self.use_s0 = use_s0
        self.constant_len = constant_len
        self.drive_dir = drive_dir

        if not retrain:
            self.model, self.prev_ts = self.load_latest()
#            print(self.prev_ts)

        if self.model is None:
            self.model = self.discretized_feedforward()
            if self.use_s0:
                self.model.append(S_0())
            self.model = ModelFull(self.model, is_discretized = True)
            self.model = self.model.to(device = DEVICE)

    ## TODO: Implement it -- Zhanhao Zhang
    def discretized_feedforward(self):
        model_list = nn.ModuleList()
        for _ in range(self.constant_len):
            model = Net(1, self.hidden_lst, self.output_dim)
            model_list.append(model)
        for _ in range(self.time_len - self.constant_len):
            model = Net(self.input_dim, self.hidden_lst, self.output_dim)
            model_list.append(model)
        return model_list
    
    ## TODO: Implement it -- Zhanhao Zhang
    def rnn(self):
        return None
    
    def update_model(self, model):
        self.model = model
    
    def prepare_model(self):
        if self.solver == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        elif self.solver == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        else:
            optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.scheduler_step, gamma = self.decay)
        return self.model, optimizer, scheduler, self.prev_ts
    
    def save_to_file(self, curr_ts = None):
        if curr_ts is None:
            curr_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
        model_save = self.model.cpu()
        torch.save(model_save, f"{self.drive_dir}/Models/{self.algo}__{curr_ts}.pt")
        self.model = self.model.to(device=DEVICE)
        return curr_ts
    
    def load_latest(self):
        ts_lst = [f.strip(".pt").split("__")[1] for f in os.listdir(f"{self.drive_dir}/Models/") if f.endswith(".pt") and f.startswith(self.algo)]
        ts_lst = sorted(ts_lst, reverse=True)
        if len(ts_lst) == 0:
            return None, None
        ts = ts_lst[0]
        print(f"Loading {self.drive_dir}/Models/{self.algo}__{ts}.pt")
        model = torch.load(f"{self.drive_dir}/Models/{self.algo}__{ts}.pt")
        model = model.to(device = DEVICE)
        return model, ts

class DynamicFactory():
    def __init__(self, dW_st, W_s0 = None):
        self.dW_st = dW_st
        self.W_st = get_W(dW_st, W_s0 = W_s0)
        self.dW_st = self.dW_st.to(device = DEVICE)
        self.n_sample = self.dW_st.shape[0]
        self.T = self.dW_st.shape[1]
        
        ## Auxiliary variables
        self.xi_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        for n in range(N_AGENT):
            self.xi_stn[:,:,n] = XI_LIST[n] * self.W_st[:,1:]
        self.mu_bar = GAMMA_BAR * ALPHA ** 2 * S
    
    def deep_hedging(self, gen_model, dis_model, use_true_mu = False, use_fast_var = False, combo_model = None, clearing_known = True, F_exact = None, H_exact = None, perturb_musigma = False, perturb_phidot = False):
        ## Setup variables
        phi_dot_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        phi_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        phi_stn[:,0,:] = S * GAMMA_BAR / GAMMA_LIST
        mu_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        sigma_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        stock_st = torch.zeros((self.n_sample, self.T + 1)).to(device = DEVICE)
        dummy_one = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        phi_bar_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        ## Initialize stock
        if combo_model is None:
            stock_st[:,0] = dis_model((-1, dummy_one)).reshape((-1,)) #-0.19 #(BETA - GAMMA_BAR * ALPHA ** 2 * S) * TR #
        else:
            stock_st[:,0] = combo_model((-1, dummy_one)).reshape((-1,))
        ## Begin iteration
        sigma_s = None
        if clearing_known:
            combo_offset = self.T * (N_AGENT - 1)
        else:
            combo_offset = self.T * N_AGENT
        ## DEBUGGING!!!
#         if F_exact is not None and H_exact is not None:
#             phi_dot_stn, phi_stn, _, _, _ = self.ground_truth(F_exact, H_exact)
        n_agent_itr = N_AGENT
        if clearing_known:
            n_agent_itr -= 1
        for t in range(self.T):
            curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
            ## Populate phi_bar
            for n in range(N_AGENT):
                phi_bar_stn[:,t,n] = self.mu_bar / GAMMA_LIST[n] / ALPHA ** 2 - XI_LIST[n] / ALPHA * self.W_st[:,t]
            delta_phi_stn = phi_stn[:,t,:n_agent_itr] - phi_bar_stn[:,t,:n_agent_itr]
            ## Mu Sigma perturbations
            perturb_sd = 2.0 #* ((t+1) * DT) ** 0.5
            perturb_phidot_sd = 100.0
            sigma_perturb = torch.normal(0.0, perturb_sd, (self.n_sample,))
            mu_perturb = torch.normal(0.0, perturb_sd, (self.n_sample,))
            phidot_perturb = torch.normal(0.0, perturb_phidot_sd, (self.n_sample,))
            # torch.normal(0, np.sqrt(DT), (sample_size, T))
            ## Discriminator - Sigma output
            if not use_fast_var:
                x_dis = curr_t.reshape((self.n_sample, 1)) #torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1)
            else:
                x_dis = curr_t.reshape((self.n_sample, 1)) #torch.cat((delta_phi_stn, self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #curr_t.reshape((self.n_sample, 1)) #
            if combo_model is None:
                sigma_s = torch.abs(dis_model((t, x_dis)).view((-1,)))
            else:
                sigma_s = torch.abs(combo_model((combo_offset + t, x_dis)).reshape((-1,)))
            sigma_st[:,t] = sigma_s
            # fast_var_stn = (phi_stn.clone()[:,t,:] * sigma_s.reshape((self.n_sample, 1)) + self.xi_stn[:,t,:]) * sigma_s.reshape((self.n_sample, 1))
            
            ## Discriminator - Mu output
            if use_true_mu:
                mu_s = get_mu_from_sigma(sigma_s.reshape((self.n_sample, 1)), phi_stn[:,t,:].reshape((self.n_sample, 1, N_AGENT)), self.W_st[:,t].reshape((self.n_sample, 1))).reshape((-1,))
            else:
                if not use_fast_var:
                    x_mu = torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1)
                else:
                    x_mu = torch.cat((delta_phi_stn, self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #torch.cat((phi_dot_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #torch.cat((fast_var_stn, curr_t), dim=1) #
                mu_s = dis_model((self.T + t, x_mu)).view((-1,))
            mu_st[:,t] = mu_s
            # if t < 10:
            #     mu_st[:,t] += GAMMA_BAR * (ALPHA ** 2) * S
            if perturb_musigma:
                sigma_st[:,t] += sigma_perturb
                mu_st[:,t] += mu_perturb
            stock_st[:,t+1] = stock_st[:,t] + mu_st[:,t] * DT + sigma_st[:,t] * self.dW_st[:,t]
            
            ## Generator output
            if not use_fast_var:
                x_gen = torch.cat((mu_st[:,t].view((self.n_sample, 1)), sigma_st[:,t].view((self.n_sample, 1)), self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #
            else:
                x_gen = torch.cat((delta_phi_stn, self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #torch.cat((fast_var_stn, mu_s.reshape((self.n_sample, 1)), self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1) #
            if combo_model is None:
                phi_dot_stn[:,t,:n_agent_itr] = gen_model((t, x_gen))
            else:
                phi_dot_stn[:,t,:n_agent_itr] = combo_model((t, x_gen))
            if perturb_phidot:
                phi_dot_stn[:,t,:] += perturb_phidot
            phi_stn[:,t+1,:n_agent_itr] = phi_stn[:,t,:n_agent_itr] + phi_dot_stn[:,t,:n_agent_itr] * DT
            if clearing_known:
                phi_dot_stn[:,t,-1] = -torch.sum(phi_dot_stn[:,t,:-1], axis = 1)
                phi_stn[:,t+1,-1] = S - torch.sum(phi_stn[:,t+1,:-1], axis = 1)
        return phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st
    
    def g_vec(self, x, q = 3/2):
        x_ind = torch.round((torch.abs(x) + 0) / 50 * 500000).long()
        x_inbound = (torch.abs(x) <= 50) + 0
        x_outbound = -torch.sign(x) * q * (q - 1) ** (-(q - 1) / q) * torch.abs(x) ** (2 * (q - 1) / q)
        return torch.sign(x) * G_MAP[x_ind * x_inbound] + x_outbound * (1 - x_inbound)
        
    def leading_order(self, power = 1.5):
        phi_dot_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        phi_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        phi_stn[:,0,:] = S * GAMMA_BAR / GAMMA_LIST
        delta_phi_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        mu_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        sigma_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        stock_st = torch.zeros((self.n_sample, self.T + 1)).to(device = DEVICE)
        phi_bar_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        gamma_hat = abs((GAMMA_1 - GAMMA_2) / (GAMMA_1 + GAMMA_2))
        gamma = (GAMMA_1 + GAMMA_2) / 2
        s_0_fricless = (BETA - ALPHA) * TR
        s0 = s_0_fricless + 1.976 * GAMMA_BAR * gamma_hat * gamma ** (3/7) * ALPHA ** (8/7) * S * LAM ** (4/7) * TR

        if N_AGENT > 2:
            for t in range(self.T + 1):
                for n in range(N_AGENT):
                    phi_stn[:,t,n] = self.mu_bar / GAMMA_LIST[n] / ALPHA ** 2 - XI_LIST[n] / ALPHA * self.W_st[:,t]
            mu_st[:,:] = self.mu_bar
            sigma_st[:,:] = ALPHA
            for t in range(self.T):
                phi_dot_stn[:,t,:] = (phi_stn[:,t+1,:] - phi_stn[:,t,:]) / DT
                stock_st[:,t+1] = stock_st[:,t] + mu_st[:,t] * DT + sigma_st[:,t] * self.dW_st[:,t]
        else:
            g_tilda_prime_0 = -1.771
            for t in range(self.T + 1):
                for n in range(N_AGENT):
                    phi_bar_stn[:,t,n] = self.mu_bar / GAMMA_LIST[n] / ALPHA ** 2 - XI_LIST[n] / ALPHA * self.W_st[:,t]
            for t in range(self.T):
                outer = -torch.sign(delta_phi_stn[:,t,:-1]) * (power * GAMMA_LIST[:-1] * XI_LIST[0] ** 4 / 8 / LAM / ALPHA ** 2) ** (1 / (power + 2))
                inner = 2 ** ((power - 1) / (power + 2)) * ((power * GAMMA_LIST[:-1] * ALPHA ** 2 / LAM) ** (1 / (power + 2))) * ((ALPHA / XI_LIST[0]) ** (2 * power / (power + 2))) * delta_phi_stn[:,t,:-1]
                phi_dot = outer * torch.abs(self.g_vec(inner, q = power)) ** (1 / (power - 1))
                d_delta_phi = phi_dot * DT + XI_LIST[0] / ALPHA * self.dW_st[:,t].reshape((self.n_sample, 1))
                delta_phi_stn[:,t+1,:-1] = delta_phi_stn[:,t,:-1] + d_delta_phi
                phi_stn[:,t+1,:-1] = delta_phi_stn[:,t+1,:-1] + phi_bar_stn[:,t+1,:-1]
                phi_stn[:,t+1,-1] = S - torch.sum(phi_stn[:,t+1,:-1], axis = 1)
                delta_phi_stn[:,t+1,-1] = phi_stn[:,t+1,-1] - phi_bar_stn[:,t+1,-1]
                phi_dot_stn[:,t,:] = (phi_stn[:,t+1,:] - phi_stn[:,t,:]) / DT
                
                # sigma_st[:,t] = ALPHA + -1.153 * (GAMMA_1 - GAMMA_2) / (GAMMA_1 + GAMMA_2) * (LAM / (1 ** (power - 1) * power)) ** (2 / (power + 2)) * ((GAMMA_1 + GAMMA_2) / 2 * ALPHA ** 2) ** (power / (power + 2)) * (ALPHA / XI_LIST[0]) ** ((4 - 2 * power) / (power + 2)) * XI_LIST[0] / ALPHA
                sigma_st[:,t] = ALPHA + (GAMMA_1 - GAMMA_2) / (GAMMA_1 + GAMMA_2) * (LAM / (2 ** (power - 1) * power)) ** (2 / (power + 2)) * ((GAMMA_1 + GAMMA_2) / 2 * ALPHA ** 2) ** (power / (power + 2)) * (ALPHA / XI_LIST[0]) ** ((4 - 2 * power) / (power + 2)) * g_tilda_prime_0 * XI_LIST[0] / ALPHA
                mu_st[:,t] = GAMMA_BAR * S * sigma_st[:,t] ** 2 + 1/2 * (GAMMA_1 - GAMMA_2) * sigma_st[:,t] ** 2 * delta_phi_stn[:,t,0] + 1/2 * XI_LIST[0] * sigma_st[:,t] / ALPHA * (GAMMA_1 - GAMMA_2) * (ALPHA - sigma_st[:,t]) * self.W_st[:,t+1]
                stock_st[:,t+1] = stock_st[:,t] + mu_st[:,t] * DT + sigma_st[:,t] * self.dW_st[:,t]
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        stock_st = stock_st + s0 #(target - stock_st[:,-1]).reshape((self.n_sample, 1))
        return phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st
    
    def ground_truth(self, F_exact, H_exact):
        ## Setup variables
        phi_dot_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        phi_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        phi_stn[:,0,:] = S * GAMMA_BAR / GAMMA_LIST
        mu_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        stock_st = torch.zeros((self.n_sample, self.T + 1)).to(device = DEVICE)
        ones = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        ## Begin iteration
#         stock_st[:,0] = (BETA - GAMMA_BAR * ALPHA ** 2 * S) * TR
        sigma_t = ALPHA + GAMMA_BAR * (1 / GAMMA_LIST[:-1] - 1 / GAMMA_MAX).reshape((1, N_AGENT - 1)) @ H_exact[:-1,:]
        sigma_st = ones @ sigma_t.reshape((1, T))
        for t in range(T):
            phi_dot_stn[:,t,:-1] = -1 / LAM * ((phi_stn[:,t,:-1] - ones @ (GAMMA_BAR / GAMMA_LIST[:-1] * S).reshape((1, N_AGENT - 1))) @ F_exact[:,t].reshape((N_AGENT, N_AGENT))[:-1,:-1].T + self.W_st[:,t].reshape((self.n_sample, 1)) @ H_exact[:-1,t].reshape((1, N_AGENT - 1)))
            phi_dot_stn[:,t,-1] = -torch.sum(phi_dot_stn[:,t,:-1], axis=1)
            phi_stn[:,t+1,:] = phi_stn[:,t,:] + phi_dot_stn[:,t,:] * DT
            mu_st[:,t] = get_mu_from_sigma(sigma_st[:,t].reshape((self.n_sample, 1)), phi_stn[:,t,:].reshape((self.n_sample, 1, N_AGENT)), self.W_st[:,t].reshape((self.n_sample, 1))).reshape((-1,))
            stock_st[:,t+1] = stock_st[:,t] + mu_st[:,t] * DT + sigma_st[:,t] * self.dW_st[:,t]
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        stock_st = stock_st + (target - stock_st[:,-1]).reshape((self.n_sample, 1))
        return phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st

    def frictionless_stock(self):
        mu_bar = GAMMA_BAR * (ALPHA ** 2) * S
        sigma_bar = ALPHA
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        stock_st = torch.zeros((self.n_sample, self.T + 1)).to(device = DEVICE)
        for t in range(T):
            stock_st[:,t+1] = stock_st[:,t] + mu_bar * DT + sigma_bar * self.dW_st[:,t]
        stock_st = stock_st + (BETA - ALPHA) * TR #(target - stock_st[:,-1]).reshape((self.n_sample, 1))
        return stock_st

    def frictionless_mu(self):
        mu_bar = GAMMA_BAR * (ALPHA ** 2) * S
        mu_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        for t in range(T):
            mu_st[:,t] = mu_bar
        return mu_st
    
    def pasting(self):
        pass

class LossFactory():
    def __init__(self, dW_st, W_s0 = None, normalize = False):
        self.dW_st = dW_st
        self.W_st = get_W(dW_st, W_s0 = W_s0)
        self.dW_st = self.dW_st.to(device = DEVICE)
        self.n_sample = self.dW_st.shape[0]
        self.T = self.dW_st.shape[1]
        self.normalize = normalize
    
    def utility_loss(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2):
        loss = 0
        for n in range(N_AGENT):
            loss_curr = self.utility_loss_single(phi_dot_stn, phi_stn, mu_st, sigma_st, n, power = power)
            loss += loss_curr
        return loss

    def utility_loss_single(self, phi_dot_stn, phi_stn, mu_st, sigma_st, n, power = 2):
        loss_curr = (torch.mean(-torch.sum(mu_st * phi_stn[:,:-1,n], axis = 1) + GAMMA_LIST[n] / 2 * torch.sum((sigma_st * phi_stn[:,:-1,n] + self.W_st[:,:-1] * XI_LIST[n]) ** 2, axis = 1) + LAM / 2 * torch.sum(torch.abs(phi_dot_stn[:,:,n]) ** power, axis = 1)) / self.T) / N_AGENT
        if self.normalize:
            loss_curr = loss_curr * XI_NORM_LIST[n]
        return loss_curr

    def utility_loss_matrix(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2):
        loss_arr = torch.zeros(self.T)
        for n in range(N_AGENT):
            loss_curr = (torch.mean(-(mu_st * phi_stn[:,:-1,n]) + GAMMA_LIST[n] / 2 * (sigma_st * phi_stn[:,:-1,n] + self.W_st[:,:-1] * XI_LIST[n]) ** 2 + LAM / 2 * torch.abs(phi_dot_stn[:,:,n]) ** power, axis = 0) / 1) / 1
            loss_arr += loss_curr
        return loss_arr

    def utility_loss_stats(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2):
        n_sample = phi_stn.shape[0]
        loss_arr = torch.zeros(n_sample)
        for n in range(N_AGENT):
            loss_curr = (torch.mean(-(mu_st * phi_stn[:,:-1,n]) + GAMMA_LIST[n] / 2 * (sigma_st * phi_stn[:,:-1,n] + self.W_st[:,:-1] * XI_LIST[n]) ** 2 + LAM / 2 * torch.abs(phi_dot_stn[:,:,n]) ** power, axis = 1) / self.T) / N_AGENT
            loss_arr += loss_curr
        loss_mean, loss_se = torch.mean(loss_arr), torch.std(loss_arr) / (n_sample ** 0.5)
        return loss_mean, loss_se
    
    def clearing_loss(self, phi_dot_stn, power = 2):
        loss = torch.abs(torch.sum(phi_dot_stn, axis = 2) / N_AGENT) ** power
        return torch.mean(loss)

    def clearing_loss_stats(self, phi_dot_stn, power = 2):
        n_sample = phi_dot_stn.shape[0]
        loss_arr = torch.mean(torch.abs(torch.sum(phi_dot_stn, axis = 2) / N_AGENT) ** power, axis = 1)
        return torch.mean(loss_arr), torch.std(loss_arr) / (n_sample ** 0.5)
    
    def clearing_loss_phi(self, phi_stn, power = 2):
        loss = torch.abs(torch.sum(phi_stn, axis = 2) - S) ** power
        return torch.mean(loss)

    def clearing_loss_same_delta(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2, delta = 0.01):
        loss_1 = self.utility_loss_matrix(phi_dot_stn, phi_stn, mu_st, sigma_st, power = power)
        loss_2 = self.utility_loss_matrix(phi_dot_stn, phi_stn + delta, mu_st, sigma_st, power = power)
        loss = torch.mean(((loss_2 - loss_1) / delta) ** 2)
        return loss

    def clearing_loss_delta(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2, delta = 0.01):
        loss_1 = self.utility_loss_matrix(phi_dot_stn, phi_stn, mu_st, sigma_st, power = power)
        loss_2 = self.utility_loss_matrix(phi_dot_stn + delta, phi_stn + delta, mu_st, sigma_st, power = power)
        loss = torch.mean(((loss_2 - loss_1) / delta) ** 2)
        return loss

    def clearing_loss_delta_accum(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2, delta = 1e-3):
        loss_1 = self.utility_loss_matrix(phi_dot_stn, phi_stn, mu_st, sigma_st, power = power)
        delta_dot = torch.ones((self.n_sample, self.T + 1, N_AGENT)) * delta
        delta_dot[:,0,:] = 0
        delta_cum = delta_dot.cumsum(dim = 1)
        loss_2 = self.utility_loss_matrix(phi_dot_stn + delta_dot[:,1:,:], phi_stn + delta_cum, mu_st, sigma_st, power = power)
        loss_1_cum = torch.flip(torch.flip(loss_1, dims = [0]).cumsum(dim = 0), dims = [0])
        loss_2_cum = torch.flip(torch.flip(loss_2, dims = [0]).cumsum(dim = 0), dims = [0])
        loss = torch.mean(((loss_2_cum - loss_1_cum) / delta_cum.mean(dim = 0).mean(dim = 1)[1:]) ** 2)
        return loss

    def clearing_loss_y(self, phi_dot_stn, phi_stn, mu_st, sigma_st, power = 2, normalize = False, y_coef = 1):
        ydot_stn = torch.zeros((self.n_sample, self.T, N_AGENT))
        const_stn = torch.ones((self.n_sample, self.T, N_AGENT))
        const_stn = torch.flip(torch.flip(const_stn, dims = [1]).cumsum(dim = 1), dims = [1])
        for n in range(N_AGENT):
            # mu_st += 1 / N_AGENT * GAMMA_LIST[n] * sigma_st * (sigma_st * phi_stn.clone()[:,:,n] + XI_LIST[n] * W_st)
            ydot_stn[:,:,n] += mu_st - GAMMA_LIST[n] * sigma_st * (sigma_st * phi_stn[:,:-1,n] + self.W_st[:,:-1] * XI_LIST[n])
        y_stn = torch.flip(torch.flip(ydot_stn, dims = [1]).cumsum(dim = 1), dims = [1]) #/ const_stn
        # y_tn = torch.mean(y_stn, dim = 0)
        # y_stn /= (const_stn ** 1)
        y_prime_stn = torch.abs(y_stn) ** (1 / (power - 1)) * torch.sign(y_stn)
        y_prime_stn /= const_stn
        loss_st = torch.sum(y_prime_stn / self.n_sample, dim = 2)
        loss = torch.sum(loss_st ** 2) * y_coef
        return loss
    
    def stock_loss(self, stock_st, power = 2):
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        loss = torch.abs(stock_st[:,-1] - target) ** power
        return torch.mean(loss)

    def stock_loss_stats(self, stock_st, power = 2):
        n_sample = stock_st.shape[0]
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        loss_arr = torch.abs(stock_st[:,-1] - target) ** power
        return torch.mean(loss_arr), torch.std(loss_arr) / (n_sample ** 0.5)
    
    def stock_loss_init(self, stock_st, power = 2):
        target = (BETA - GAMMA_BAR * ALPHA ** 2 * S) * TR
        loss = torch.abs(stock_st[:,0] - target) ** power
        return torch.mean(loss)

    def regularize_loss(self, data, C = 1e-3):
        return C * torch.mean(torch.abs(data) ** 2)

## Write training logs to file
def write_logs(ts_lst, train_args):
    with open(f"{drive_dir}/Logs.tsv", "a") as f:
        for i in range(1, len(ts_lst)):
            line = f"{ts_lst[i - 1]}\t{ts_lst[i]}\t{json.dumps(train_args)}\n"
            f.write(line)

## Visualize loss function through training
def visualize_loss(loss_arr, round, algo, ts, loss_truth):
    round += 1
    plt.plot(loss_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(y = loss_truth, color = "red")
    plt.title(f"Final Loss = {loss_arr[-1]:.2e}, True Loss = {loss_truth:.2e}")
    plt.savefig(f"{drive_dir}/Plots/loss_round={round}_{algo}_{ts}.png")
    plt.close()

## Visualize inference function through training
def visualize_infer(x_arr, y_arr_lst, name, xname, yname, label_lst, title = ""):
    idx = 0
    for y_arr, label in zip(y_arr_lst, label_lst):
        if len(label_lst) > 1:
            plt.plot(x_arr, y_arr, label = label)
        else:
            plt.scatter(x_arr, y_arr, label = label)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{drive_dir}/Plots/{name}.png")
    plt.close()

## Visualize the comparison of dynamics
def visualize_comparison(timestamps, arr_lst, round, ts, name, algo_lst, comment = None, expand = True):
    assert name in ["phi", "phi_dot", "phi_dot_short", "sigma", "mu", "s"]
    round += 1
    if name == "phi":
        title = "${\\varphi}_t$"
    elif name in ["phi_dot", "phi_dot_short"]:
        title = "$\dot{\\varphi}_t$"
    elif name == "sigma":
        title = "$\sigma_t$"
    elif name == "mu":
        title = "$\mu_t$"
    elif name == "s":
        title = "$S_t$"
    else:
        title = name
    if comment is not None:
        title2 = title + "\n" + str(comment)
    else:
        title2 = title
    if name == "phi_dot_short":
        title2 = title
    if name in ["phi_dot", "phi"] and expand:
        size = arr_lst[0].cpu().detach().numpy().shape
        if len(size) == 1:
            size = 1
        else:
            size = size[1]
        for i in range(size):
            ax = plt.subplot(111)
            for arr, algo in zip(arr_lst, algo_lst):
                arr = arr.cpu().detach().numpy()
                if name in ["phi", "phi_dot"]:
                    arr = arr * S_VAL
                if algo == "pasting":
                    algo = algo.replace("pasting", "ST-Hedging")
                if len(arr.shape) == 2 and arr.shape[1] == 1:
                    arr = arr.reshape((-1,))
                if len(arr.shape) == 1:
                    ax.plot(timestamps, arr,label = f"{algo}")
                else:
                    ax.plot(timestamps, arr[:, i], label = f"{algo}\n - Agent {i + 1}")
            #ax.set_xlabel("T")
            ax.set_xlabel(r"$t$", fontsize = 16)
            ax.xaxis.set_label_coords(0.63, 0.06)
            ax.set_xlim(0, TR)
            ax.set_ylabel(title)
            #ax.set_title(title2)
            ax.grid()
            box2 = ax.get_position()
            ax.legend(loc="center left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
#            ax.legend(loc="lower left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
#            ax.legend(loc="lower left")
            plt.savefig(f"{drive_dir}/Plots/comp_round={round}_{name}_agent{i+1}_{ts}.png", bbox_inches='tight')
            plt.close()
    else:
        ax = plt.subplot(111)
        for arr, algo in zip(arr_lst, algo_lst):
            arr = arr.cpu().detach().numpy()
            if name in ["phi", "phi_dot"]:
                arr = arr * S_VAL
            if algo == "pasting":
                algo = algo.replace("pasting", "ST-Hedging")
            if len(arr.shape) == 2 and arr.shape[1] == 1:
                arr = arr.reshape((-1,))
            if len(arr.shape) == 1:
                ax.plot(timestamps, arr,label = f"{algo}")
            else:
                for i in range(arr.shape[1]):
                    ax.plot(timestamps, arr[:, i], label = f"{algo}\n - Agent {i + 1}")
        #ax.set_xlabel("T")
        ax.set_xlabel(r"$t$", fontsize = 16)
        ax.xaxis.set_label_coords(0.63, 0.06)
        ax.set_xlim(0, TR)
        ax.set_ylabel(title)
        #ax.set_title(title2)
        ax.grid()
        box2 = ax.get_position()
        ax.legend(loc="center left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
#        ax.legend(loc="lower left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
        # if name in ["phi", "phi_dot"]:
        #     ax.legend(loc="upper left")
        # else:
        #     ax.legend(loc="lower left")
#        ax.legend()
        plt.savefig(f"{drive_dir}/Plots/comp_round={round}_{name}_{ts}.png", bbox_inches='tight')
        plt.close()

def prepare_generator(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver = "Adam", use_pretrained_gen = True, use_fast_var = False, clearing_known = True, drive_dir = "."):
    retrain = not use_pretrained_gen
    if not use_fast_var:
        input_dim = 4 #+ N_AGENT #2 + N_AGENT
    else:
        input_dim = 2 + N_AGENT #2 + N_AGENT
    n_model = T #T * (N_AGENT - 1)
#     if not clearing_known:
#         n_model += T
    output_dim = N_AGENT - 1
    if not clearing_known:
        output_dim += 1
    if clearing_known:
        input_dim -= 1
    model_factory = ModelFactory(n_model, "generator", input_dim, gen_hidden_lst, output_dim, gen_lr, gen_decay, gen_scheduler_step, use_s0 = False, solver = gen_solver, retrain = retrain, drive_dir = drive_dir)
    return model_factory

def prepare_discriminator(dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver = "Adam", use_pretrained_dis = True, use_true_mu = False, use_fast_var = False, clearing_known = True, drive_dir = "."):
    n_model = T + 1
    if not use_true_mu:
        n_model += T
    retrain = not use_pretrained_dis
    if not use_fast_var:
        input_dim = 2 + N_AGENT #2 + N_AGENT
    else:
        input_dim = 2 + N_AGENT
    if clearing_known:
        input_dim -= 1
    model_factory = ModelFactory(n_model, "discriminator", input_dim, dis_hidden_lst, 1, dis_lr, dis_decay, dis_scheduler_step, use_s0 = True, solver = dis_solver, retrain = retrain, constant_len = T, drive_dir = drive_dir)
    return model_factory

def prepare_combo(combo_hidden_lst, combo_lr, combo_decay, combo_scheduler_step, combo_solver = "Adam", use_pretrained_combo = True, use_true_mu = False, use_fast_var = False, clearing_known = True):
    n_model = T * (N_AGENT - 1) + T + 1
    if not clearing_known:
        n_model += T
    if not use_true_mu:
        n_model += T
    retrain = not use_pretrained_combo
    if not use_fast_var:
        input_dim = 2 + N_AGENT
    else:
        input_dim = 1 + N_AGENT
    model_factory = ModelFactory(n_model, "combo", input_dim, combo_hidden_lst, 1, combo_lr, combo_decay, combo_scheduler_step, use_s0 = True, solver = combo_solver, retrain = retrain, constant_len = 0)
    return model_factory

def slc(lst, idx):
    return lst[min(idx, len(lst) - 1)]

def train_single(generator, discriminator, optimizer, scheduler, epoch, sample_size, use_true_mu, use_fast_var, train_type, F_exact, H_exact, dis_loss = 1, ckpt_freq = 10000, model_factory = None, curr_ts = None, combo_model = None, clearing_known = True, normalize = False, utility_power = 2, normalize_y = False, y_coef = 1):
    assert train_type in ["generator", "discriminator", "combo"]
    loss_arr = []
    for itr in tqdm(range(epoch)):
        optimizer.zero_grad()
        dW_st = torch.normal(0, np.sqrt(DT), (sample_size, T))
        dynamic_factory = DynamicFactory(dW_st)
        loss_factory = LossFactory(dW_st, normalize = normalize)
        if train_type == "combo":
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(None, None, use_true_mu = use_true_mu, use_fast_var = use_fast_var, combo_model = combo_model, clearing_known = clearing_known)
        else:
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known, F_exact = F_exact, H_exact = H_exact, perturb_musigma = False, perturb_phidot = False) #perturb_musigma = train_type == "generator", perturb_phidot = train_type == "discriminator"
        if train_type == "generator":
            loss = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) + loss_factory.regularize_loss(phi_dot_stn, C = 1e-3) + loss_factory.clearing_loss(phi_dot_stn, power = dis_loss)
        elif train_type == "discriminator":
            stock_loss = loss_factory.stock_loss(stock_st, power = dis_loss)
            clearing_loss = loss_factory.clearing_loss_y(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power, normalize = normalize_y, y_coef = y_coef)
            if itr == 0:
                stock_clearing_loss_ratio = clearing_loss.data * 3000 / (stock_loss.data * 1) #3000 #min(stock_loss.data * 100 / clearing_loss.data, 1)
                # clearing_stock_loss_ratio = stock_loss.data / clearing_loss.data * 100
            if not use_true_mu:
                stock_loss *= stock_clearing_loss_ratio
            # clearing_loss *= clearing_stock_loss_ratio
            loss = stock_loss + clearing_loss + loss_factory.regularize_loss(sigma_st, C = 1e-3) + loss_factory.regularize_loss(mu_st, C = 1e-3) #+ loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power)
        else:
            loss = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) + loss_factory.stock_loss(stock_st, power = dis_loss) + loss_factory.clearing_loss(phi_dot_stn, power = dis_loss)
        assert not torch.isnan(loss.data)
        loss.backward()
        if train_type in ["generator", "combo"]:
            loss = loss * S_VAL
        else:
            loss = loss
        loss_arr.append(float(loss.data))
        optimizer.step()
        scheduler.step()
        ## Save checkpoint frequently
        if itr % ckpt_freq == 0 and itr > 0:
            if train_type == "generator":
                model_factory.update_model(generator)
            elif train_type == "discriminator":
                model_factory.update_model(discriminator)
            else:
                model_factory.update_model(combo_model)
            model_factory.save_to_file(curr_ts)
    if utility_power == 2:
        phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.ground_truth(F_exact, H_exact)
    else:
        phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.leading_order(power = utility_power)
    if train_type == "generator":
        model = generator
        loss_truth_final = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) * S_VAL
    elif train_type == "discriminator":
        model = discriminator
        loss_truth_final = loss_factory.stock_loss(stock_st, power = dis_loss)
    else:
        model = combo_model
        loss_truth_final = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) * S_VAL + loss_factory.stock_loss(stock_st, power = dis_loss) + loss_factory.clearing_loss(phi_dot_stn, power = dis_loss) * S_VAL
    loss_truth_final = float(loss_truth_final.data)
    return model, loss_arr, loss_truth_final

def training_pipeline(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, gen_epoch, gen_sample, use_pretrained_gen, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_epoch, dis_sample, use_pretrained_dis, combo_hidden_lst, combo_lr, combo_decay, combo_scheduler_step, combo_solver, combo_epoch, combo_sample, use_pretrained_combo, dis_loss = [1], y_coef_lst = [1], utility_power = 2, use_true_mu = False, use_fast_var = False, total_rounds = 1, normalize_up_to = 0, train_gen = True, train_dis = True, last_round_dis = True, visualize_obs = 0, seed = 0, ckpt_freq = 10000, use_combo = False, clearing_known = True, train_args = None):
    ## Generate Brownian paths for testing
    torch.manual_seed(seed)
    dW_st_eval = torch.normal(0, np.sqrt(DT), size = (N_SAMPLE, T))
    F_exact, H_exact = get_FH_exact()
    benchmark_name = "Ground Truth"
    if utility_power != 2:
        benchmark_name = "Leading Order"
    ## Begin training
    for gan_round in range(total_rounds):
        print(f"Round #{gan_round + 1}:")
        curr_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
        # if gan_round == 0:
        #     dynamic_factory = DynamicFactory(dW_st_eval)
        #     if utility_power == 2:
        #         phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.ground_truth(F_exact, H_exact)
        #     else:
        #         phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.leading_order(power = utility_power)
        #     torch.save(dW_st_eval, "ground_truth/dW.pt")
        #     torch.save(phi_dot_stn_truth, "ground_truth/phi_dot_stn_truth.pt")
        #     torch.save(phi_stn_truth, "ground_truth/phi_stn_truth.pt")
        #     torch.save(mu_st_truth, "ground_truth/mu_st_truth.pt")
        #     torch.save(sigma_st_truth, "ground_truth/sigma_st_truth.pt")
        #     torch.save(stock_st_truth, "ground_truth/stock_st_truth.pt")
        if not use_combo:
            if train_gen:
                print("\tTraining Generator...")
                model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, gan_round), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, gan_round), use_pretrained_gen = False, use_fast_var = use_fast_var, clearing_known = clearing_known)
                model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, gan_round), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, gan_round), use_pretrained_dis = use_pretrained_dis or not train_dis or gan_round > 0, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known)
                generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
                discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()
                generator, loss_arr_gen, loss_truth_final_gen = train_single(generator, discriminator, optimizer_gen, scheduler_gen, slc(gen_epoch, gan_round), slc(gen_sample, gan_round), use_true_mu, use_fast_var, "generator", F_exact, H_exact, dis_loss = slc(dis_loss, gan_round), ckpt_freq = ckpt_freq, model_factory = model_factory_gen, curr_ts = curr_ts, clearing_known = clearing_known, normalize = gan_round < normalize_up_to, utility_power = utility_power)
                model_factory_gen.update_model(generator)
                model_factory_gen.save_to_file(curr_ts)
                visualize_loss(loss_arr_gen, gan_round, "generator", curr_ts, loss_truth_final_gen)
            if train_dis and (gan_round < total_rounds - 1 or last_round_dis):
                print("\tTraining Discriminator...")
                model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, gan_round), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, gan_round), use_pretrained_gen = use_pretrained_gen or not train_gen or gan_round > 0, use_fast_var = use_fast_var, clearing_known = clearing_known)
                model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, gan_round), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, gan_round), use_pretrained_dis = False, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known)
                generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
                discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()
                discriminator, loss_arr_dis, loss_truth_final_dis = train_single(generator, discriminator, optimizer_dis, scheduler_dis, slc(dis_epoch, gan_round), slc(dis_sample, gan_round), use_true_mu, use_fast_var, "discriminator", F_exact, H_exact, dis_loss = slc(dis_loss, gan_round), ckpt_freq = ckpt_freq, model_factory = model_factory_dis, curr_ts = curr_ts, clearing_known = clearing_known, utility_power = utility_power, normalize_y = gan_round < 3)
                model_factory_dis.update_model(discriminator)
                model_factory_dis.save_to_file(curr_ts)
                visualize_loss(loss_arr_dis, gan_round, "discriminator", curr_ts, loss_truth_final_dis)
        else:
            model_factory_combo = prepare_combo(combo_hidden_lst, slc(combo_lr, gan_round), combo_decay, combo_scheduler_step, combo_solver = slc(combo_solver, gan_round), use_pretrained_combo = use_pretrained_combo or gan_round > 0, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known)
            combo, optimizer_combo, scheduler_combo, prev_ts_combo = model_factory_combo.prepare_model()
            print("\tTraining Discriminator...")
            combo, loss_arr_combo, loss_truth_final_combo = train_single(None, None, optimizer_combo, scheduler_combo, slc(combo_epoch, gan_round), slc(combo_sample, gan_round), use_true_mu, use_fast_var, "combo", F_exact, H_exact, dis_loss = slc(dis_loss, gan_round), ckpt_freq = ckpt_freq, model_factory = model_factory_combo, curr_ts = curr_ts, combo_model = combo, clearing_known = clearing_known, utility_power = utility_power, y_coef = slc(y_coef_lst, gan_round))
            model_factory_combo.update_model(combo)
            model_factory_combo.save_to_file(curr_ts)
            visualize_loss(loss_arr_combo, gan_round, "combo", curr_ts, loss_truth_final_combo)
        ## Evaluation per round
        dynamic_factory = DynamicFactory(dW_st_eval)
        loss_factory = LossFactory(dW_st_eval)
        model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, gan_round), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, gan_round), use_pretrained_gen = True, use_fast_var = use_fast_var, clearing_known = clearing_known)
        model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, gan_round), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, gan_round), use_pretrained_dis = True, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known)
        generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
        discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()
        if not use_combo:
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known, F_exact = F_exact, H_exact = H_exact)
        else:
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(None, None, use_true_mu = use_true_mu, use_fast_var = use_fast_var, combo_model = combo, clearing_known = clearing_known)
        loss_utility = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) * S_VAL
        loss_stock = loss_factory.stock_loss(stock_st, power = slc(dis_loss, gan_round))
        loss_clearing = loss_factory.clearing_loss(phi_dot_stn, power = utility_power)

        if utility_power == 2:
            phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.ground_truth(F_exact, H_exact)
        else:
            phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.leading_order(power = utility_power)
        stock_st_frictionless = dynamic_factory.frictionless_stock()
        mu_st_frictionless = dynamic_factory.frictionless_mu()
            
        loss_truth_utility = loss_factory.utility_loss(phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, power = utility_power) * S_VAL
        loss_truth_stock = loss_factory.stock_loss(stock_st_truth, power = slc(dis_loss, gan_round))
        loss_truth_clearing = loss_factory.clearing_loss(phi_dot_stn_truth, power = utility_power)
        ## Visualize
        comment = f"Model Loss: Utility = {loss_utility:.2e}, Stock = {loss_stock:.2e}, Clearing = {loss_clearing:.2e}\n{benchmark_name} Loss: Utility = {loss_truth_utility:.2e}, Stock = {loss_truth_stock:.2e}, Clearing = {loss_truth_clearing:.2e}\n"
        visualize_comparison(TIMESTAMPS, [phi_dot_stn[visualize_obs,:], phi_dot_stn_truth[visualize_obs,:]], gan_round, curr_ts, "phi_dot", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [phi_stn[visualize_obs,1:], phi_stn_truth[visualize_obs,1:]], gan_round, curr_ts, "phi", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [mu_st[visualize_obs,:], mu_st_truth[visualize_obs,:], mu_st_frictionless[visualize_obs,:]], gan_round, curr_ts, "mu", ["Model", benchmark_name, "Frictionless"], comment = comment)
        visualize_comparison(TIMESTAMPS, [sigma_st[visualize_obs,:], sigma_st_truth[visualize_obs,:]], gan_round, curr_ts, "sigma", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [stock_st[visualize_obs,1:], stock_st_truth[visualize_obs,1:], stock_st_frictionless[visualize_obs,1:]], gan_round, curr_ts, "s", ["Model", benchmark_name, "Frictionless"], comment = comment)
        ## Save logs to file
        if not use_combo:
            write_logs([prev_ts_gen, curr_ts], train_args)
        else:
            write_logs([prev_ts_combo, curr_ts], train_args)
    if total_rounds == 0:
        model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, 0), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, 0), use_pretrained_gen = True, use_fast_var = use_fast_var, clearing_known = clearing_known)
        model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, 0), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, 0), use_pretrained_dis = True, use_true_mu = use_true_mu, use_fast_var = use_fast_var)
        generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
        discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()

        gan_round = 0
        curr_ts = "curr"
        dynamic_factory = DynamicFactory(dW_st_eval)
        loss_factory = LossFactory(dW_st_eval)
        if not use_combo:
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known, F_exact = F_exact, H_exact = H_exact)
        else:
            phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(None, None, use_true_mu = use_true_mu, use_fast_var = use_fast_var, combo_model = combo, clearing_known = clearing_known)
        loss_utility = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power) * S_VAL
        loss_stock = loss_factory.stock_loss(stock_st, power = slc(dis_loss, gan_round))
        loss_clearing = loss_factory.clearing_loss(phi_dot_stn, power = utility_power)
        if utility_power == 2:
            phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.ground_truth(F_exact, H_exact)
        else:
            phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.leading_order(power = utility_power)
        stock_st_frictionless = dynamic_factory.frictionless_stock()
        mu_st_frictionless = dynamic_factory.frictionless_mu()
        loss_truth_utility = loss_factory.utility_loss(phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, power = utility_power) * S_VAL
        loss_truth_stock = loss_factory.stock_loss(stock_st_truth, power = slc(dis_loss, gan_round))
        loss_truth_clearing = loss_factory.clearing_loss(phi_dot_stn_truth, power = utility_power)
        ## Visualize
        comment = f"Model Loss: Utility = {loss_utility:.2e}, Stock = {loss_stock:.2e}, Clearing = {loss_clearing:.2e}\n{benchmark_name} Loss: Utility = {loss_truth_utility:.2e}, Stock = {loss_truth_stock:.2e}, Clearing = {loss_truth_clearing:.2e}\n"
        visualize_comparison(TIMESTAMPS, [phi_dot_stn[visualize_obs,:], phi_dot_stn_truth[visualize_obs,:]], gan_round, curr_ts, "phi_dot", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [phi_stn[visualize_obs,1:], phi_stn_truth[visualize_obs,1:]], gan_round, curr_ts, "phi", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [mu_st[visualize_obs,:], mu_st_truth[visualize_obs,:]], gan_round, curr_ts, "mu", ["Model", benchmark_name], comment = comment)
        # visualize_comparison(TIMESTAMPS, [mu_st[visualize_obs,:], mu_st_truth[visualize_obs,:], mu_st_frictionless[visualize_obs,:]], gan_round, curr_ts, "mu", ["Model", benchmark_name, "Frictionless"], comment = comment)
        visualize_comparison(TIMESTAMPS, [sigma_st[visualize_obs,:], sigma_st_truth[visualize_obs,:]], gan_round, curr_ts, "sigma", ["Model", benchmark_name], comment = comment)
        visualize_comparison(TIMESTAMPS, [stock_st[visualize_obs,1:], stock_st_truth[visualize_obs,1:], stock_st_frictionless[visualize_obs,1:]], gan_round, curr_ts, "s", ["Model", benchmark_name, "Frictionless"], comment = comment)
        # visualize_comparison(TIMESTAMPS, [phi_dot_stn[visualize_obs,:,[0,1]], phi_dot_stn_truth[visualize_obs,:,[0,1]]], gan_round, curr_ts, "phi_dot_short", ["Model", benchmark_name], comment = "")
        visualize_comparison(TIMESTAMPS, [phi_dot_stn[visualize_obs,:,[2,7,8]], phi_dot_stn_truth[visualize_obs,:,[2,7,8]]], gan_round, curr_ts, "phi_dot_short", ["Model", benchmark_name], comment = "")
    return generator, discriminator

def inference(generator, discriminator, randomized = True, clearing_known = False):
    N_AGENT = 2 #10
    agent_num = 0
    n_agent_itr = N_AGENT
    if clearing_known:
        n_agent_itr -= 1
    if randomized:
        N_SAMPLE = 50000
        delta_phi_stn = torch.normal(0, 3, size = (N_SAMPLE, T, N_AGENT)) #torch.zeros((N_SAMPLE, T + 1, N_AGENT)).to(device = DEVICE) #
        torch.manual_seed(0)
        dW_st = torch.normal(0, np.sqrt(DT), size = (N_SAMPLE, T)) #torch.zeros((N_SAMPLE, T)).to(device = DEVICE) #
    else:
        N_SAMPLE = 100
        ts = 0
        rg = 3
        delta_phi_stn = torch.zeros((N_SAMPLE, T + 1, N_AGENT)).to(device = DEVICE) #torch.normal(0, 3, size = (N_SAMPLE, T, N_AGENT)) #
        for ts in range(T):
            delta_phi_stn[:,ts,agent_num] = torch.from_numpy(np.linspace(-rg, rg, N_SAMPLE))
        dW_st = torch.zeros((N_SAMPLE, T)).to(device = DEVICE) #torch.normal(0, np.sqrt(DT), size = (N_SAMPLE, T)) #
    phi_bar_stn = torch.zeros((N_SAMPLE, T, N_AGENT)).to(device = DEVICE)
    phi_stn = torch.zeros((N_SAMPLE, T, N_AGENT)).to(device = DEVICE)
    phi_stn[:,0,:] = S * GAMMA_BAR / GAMMA_LIST
    W_st = get_W(dW_st, W_s0 = None)
    mu_st = torch.zeros((N_SAMPLE, T)).to(device = DEVICE)
    sigma_st = torch.zeros((N_SAMPLE, T)).to(device = DEVICE)
    mu_bar = GAMMA_BAR * ALPHA ** 2 * S
    ## mu_st += 1 / N_AGENT * GAMMA_LIST[n] * sigma_st * (sigma_st * phi_stn.clone()[:,:,n] + XI_LIST[n] * W_st)
    mu_st_true = torch.zeros((N_SAMPLE, T)).to(device = DEVICE)
    phi_gamma_st = torch.zeros((N_SAMPLE, T)).to(device = DEVICE)
    REPEAT = 1
    musigma_t_all = 0
    musigma_t_true_all = 0
    mu_t_all = 0
    mu_t_true_all = 0
    with torch.no_grad():
        for _ in tqdm(range(REPEAT)):
            for t in range(T):
                for n in range(N_AGENT):
                    phi_bar_stn[:,t,n] = mu_bar / GAMMA_LIST[n] / ALPHA ** 2 - XI_LIST[n] / ALPHA * W_st[:,t]
                phi_stn[:,t,:] = phi_bar_stn[:,t,:] + delta_phi_stn[:,t,:]
                for n in range(N_AGENT):
                    phi_gamma_st[:,t] += GAMMA_LIST[n] * phi_stn[:,t,n]
                curr_t = torch.ones((N_SAMPLE, 1)).to(device = DEVICE)
                x_mu = torch.cat((delta_phi_stn[:,t,:n_agent_itr], W_st[:,t].reshape((N_SAMPLE, 1)), curr_t), dim=1)
                mu_s = discriminator((T + t, x_mu)).view((-1,))
                x_dis = curr_t.reshape((N_SAMPLE, 1))
                sigma_s = torch.abs(discriminator((t, x_dis)).view((-1,)))
                mu_st[:,t] = mu_s
                sigma_st[:,t] = sigma_s
                for n in range(N_AGENT):
                    mu_st_true[:,t] += 1 / N_AGENT * GAMMA_LIST[n] * sigma_s * (sigma_s * phi_stn[:,t,n] + XI_LIST[n] * W_st[:,t])
            musigma_st = mu_st / sigma_st ** 2
            musigma_t = musigma_st.mean(dim = 0)
            musigma_st_true = mu_st_true / sigma_st ** 2
            musigma_t_true = musigma_st_true.mean(dim = 0)
            mu_t = mu_st.mean(dim = 0)
            mu_t_true = mu_st_true.mean(dim = 0)
            ## Update
            musigma_t_all += musigma_t / REPEAT
            musigma_t_true_all += musigma_t_true / REPEAT
            mu_t_all += mu_t / REPEAT
            mu_t_true_all += mu_t_true / REPEAT
    # x_arr, y_arr, name, xname, yname
    if randomized:
        visualize_infer(TIMESTAMPS, [musigma_t, musigma_t_true], "musigma_t", "T", "mu over sigma^2", ["Model", "Truth"])
        # visualize_infer(TIMESTAMPS, [mu_t, mu_t_true], "mu_t", "T", "mu", ["Model", "Truth"])
        # mu_s = mu_st.mean(dim = 1)
        # mu_s_true = mu_st_true.mean(dim = 1)
        # phi_gamma_s = phi_gamma_st.mean(dim = 1)
        # visualize_infer(phi_gamma_s, [mu_s, mu_s_true], "mu_phigamma", "\sum_n phi_n * gamma_n", "mu", ["Model", "Truth"], title = "")
    else:
        # ts = 80
        # slope = (mu_st[-1,ts] - mu_st[0,ts]) / (rg * 2)
        # slope_true = (mu_st_true[-1,ts] - mu_st_true[0,ts]) / (rg * 2)
        # visualize_infer(delta_phi_stn[:,ts,agent_num], [mu_st[:,ts], mu_st_true[:,ts]], "mu_fast", "Fast Variable", "mu", ["Model", "Truth"], title = f"Model Slope = {slope}\nTrue Slope = {slope_true}")
        #####
        mu_s = mu_st.mean(dim = 1)
        mu_s_true = mu_st_true.mean(dim = 1)
        phi_gamma_s = phi_gamma_st.mean(dim = 1)
        slope = (mu_s[-1] - mu_s[0]) / (rg * 2)
        slope_true = (mu_s_true[-1] - mu_s_true[0]) / (rg * 2)
        visualize_infer(delta_phi_stn[:,ts,agent_num], [mu_s, mu_s_true], "mu_fast", "Fast Variable", "mu", ["Model", "Truth"], title = f"Model Slope = {slope}\nTrue Slope = {slope_true}")
        # visualize_infer(phi_gamma_s, [mu_s, mu_s_true], "mu_phigamma", "\sum_n phi_n * gamma_n", "mu", ["Model", "Truth"], title = "")
    return mu_st, sigma_st, delta_phi_stn, mu_st_true

def transfer_learning():
    pass

def compute_trajectory(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_loss, use_true_mu = False, use_fast_var = False, seed = 0, clearing_known = True, utility_power = 2, drive_dir = "."):
    ## Generate Brownian paths for testing
    torch.manual_seed(seed)
    dW_st_eval = torch.normal(0, np.sqrt(DT), size = (N_SAMPLE, T))
    F_exact, H_exact = get_FH_exact()
    benchmark_name = "Ground Truth"
    if utility_power != 2:
        benchmark_name = "Leading Order"
    model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, 0), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, 0), use_pretrained_gen = True, use_fast_var = use_fast_var, clearing_known = clearing_known, drive_dir = drive_dir)
    model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, 0), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, 0), use_pretrained_dis = True, use_true_mu = use_true_mu, use_fast_var = use_fast_var, drive_dir = drive_dir)
    generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
    discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()

    dynamic_factory = DynamicFactory(dW_st_eval)
    loss_factory = LossFactory(dW_st_eval)
    phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var, clearing_known = clearing_known, F_exact = F_exact, H_exact = H_exact)
    loss_utility_mean, loss_utility_se = loss_factory.utility_loss_stats(phi_dot_stn, phi_stn, mu_st, sigma_st, power = utility_power)
    loss_utility_mean, loss_utility_se = loss_utility_mean * S_VAL, loss_utility_se * S_VAL
    loss_stock_mean, loss_stock_se = loss_factory.stock_loss_stats(stock_st, power = 2)
    loss_clearing_mean, loss_clearing_se = loss_factory.clearing_loss_stats(phi_dot_stn, power = 2)
    if utility_power == 2:
        phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.ground_truth(F_exact, H_exact)
    else:
        phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.leading_order(power = utility_power)
    stock_st_frictionless = dynamic_factory.frictionless_stock()
    mu_st_frictionless = dynamic_factory.frictionless_mu()
    loss_truth_utility_mean, loss_truth_utility_se = loss_factory.utility_loss_stats(phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, power = utility_power)
    loss_truth_utility_mean, loss_truth_utility_se = loss_truth_utility_mean * S_VAL, loss_truth_utility_se * S_VAL
    loss_truth_stock_mean, loss_truth_stock_se = loss_factory.stock_loss_stats(stock_st_truth, power = slc(dis_loss, 0))
    loss_truth_clearing_mean, loss_truth_clearing_se = loss_factory.clearing_loss_stats(phi_dot_stn_truth, power = 2)
    return phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st, loss_utility_mean, loss_utility_se, loss_stock_mean, loss_stock_se, loss_clearing_mean, loss_clearing_se, phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth, loss_truth_utility_mean, loss_truth_utility_se, loss_truth_stock_mean, loss_truth_stock_se, loss_truth_clearing_mean, loss_truth_clearing_se

def plot_all_trajectories(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_loss, use_fast_var = False, seed = 0, clearing_known = False, utility_power = 2, **train_args):
    drive_dir = f"{N_AGENT}agents_power{utility_power}"
    phi_dot_stn_nomu, phi_stn_nomu, mu_st_nomu, sigma_st_nomu, stock_st_nomu, loss_utility_mean_nomu, loss_utility_se_nomu, loss_stock_mean_nomu, loss_stock_se_nomu, loss_clearing_mean_nomu, loss_clearing_se_nomu, phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth, loss_truth_utility_mean, loss_truth_utility_se, loss_truth_stock_mean, loss_truth_stock_se, loss_truth_clearing_mean, loss_truth_clearing_se = compute_trajectory(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_loss, use_true_mu = False, use_fast_var = use_fast_var, seed = seed, clearing_known = clearing_known, utility_power = utility_power, drive_dir = drive_dir + "_mu_unknown")
    s0_nomu = torch.mean(stock_st_nomu[:,0])
    s0_truth = torch.mean(stock_st_truth[:,0])
    if utility_power == 2 or N_AGENT == 2:
        phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st, loss_utility_mean, loss_utility_se, loss_stock_mean, loss_stock_se, loss_clearing_mean, loss_clearing_se, phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth, loss_truth_utility_mean, loss_truth_utility_se, loss_truth_stock_mean, loss_truth_stock_se, loss_truth_clearing_mean, loss_truth_clearing_se = compute_trajectory(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_loss, use_true_mu = True, use_fast_var = use_fast_var, seed = seed, clearing_known = clearing_known, utility_power = utility_power, drive_dir = drive_dir)
        s0 = torch.mean(stock_st[:,0])

    visualize_obs = 0
    if utility_power == 2:
        benchmark_name = "Ground Truth"
    else:
        if N_AGENT == 2:
            visualize_obs = -1
            benchmark_name = "Leading Order"
        else:
            benchmark_name = "Frictionless"
    if utility_power == 1.5 and N_AGENT > 2:
        dct = {
            "Neg Utility Mean": [float(loss_utility_mean_nomu.detach()), float(loss_truth_utility_mean.detach())],
            "Stock Loss Mean": [float(loss_stock_mean_nomu.detach()), float(loss_truth_stock_mean.detach())],
            "Clearing Loss Mean": [float(loss_clearing_mean_nomu.detach()), float(loss_truth_clearing_mean.detach())],
            "Neg Utility SE": [float(loss_utility_se_nomu.detach()), float(loss_truth_utility_se.detach())],
            "Stock Loss SE": [float(loss_stock_se_nomu.detach()), float(loss_truth_stock_se.detach())],
            "Clearing Loss SE": [float(loss_clearing_se_nomu.detach()), float(loss_truth_clearing_se.detach())],
            "S0": [float(s0_nomu.detach()), float(s0_truth.detach())],
            "Type": ["Mu Unknown", benchmark_name]
        }
    else:
        dct = {
            "Neg Utility Mean": [float(loss_utility_mean_nomu.detach()), float(loss_utility_mean.detach()), float(loss_truth_utility_mean.detach())],
            "Stock Loss Mean": [float(loss_stock_mean_nomu.detach()), float(loss_stock_mean.detach()), float(loss_truth_stock_mean.detach())],
            "Clearing Loss Mean": [float(loss_clearing_mean_nomu.detach()), float(loss_clearing_mean.detach()), float(loss_truth_clearing_mean.detach())],
            "Neg Utility SE": [float(loss_utility_se_nomu.detach()), float(loss_utility_se.detach()), float(loss_truth_utility_se.detach())],
            "Stock Loss SE": [float(loss_stock_se_nomu.detach()), float(loss_stock_se.detach()), float(loss_truth_stock_se.detach())],
            "Clearing Loss SE": [float(loss_clearing_se_nomu.detach()), float(loss_clearing_se.detach()), float(loss_truth_clearing_se.detach())],
            "S0": [float(s0_nomu.detach()), float(s0.detach()), float(s0_truth.detach())],
            "Type": ["Mu Unknown", "Mu Known", benchmark_name]
        }
    df = pd.DataFrame.from_dict(dct)
    df.to_csv(f"Tables/{drive_dir}.csv", index = False)
    
    if N_AGENT == 2:
        AGENT_LST = [0, 1]
    elif utility_power == 2:
        AGENT_LST = [1, 3] #[0, 3]
    else:
        AGENT_LST = list(range(N_AGENT))
    if utility_power == 1.5 and N_AGENT > 2:
        visualize_comparison(TIMESTAMPS, [mu_st_nomu[visualize_obs,:]], 0, drive_dir, "mu", ["$\mu$ Unknown"], comment = "")
        visualize_comparison(TIMESTAMPS, [sigma_st_nomu[visualize_obs,:]], 0, drive_dir, "sigma", ["$\mu$ Unknown"], comment = "")
        visualize_comparison(TIMESTAMPS, [phi_dot_stn_nomu[visualize_obs,:,agent] for agent in AGENT_LST], 0, drive_dir, "phi_dot", [f"Agent {agent + 1}" for agent in AGENT_LST], comment = "", expand = False)
        visualize_comparison(TIMESTAMPS, [phi_stn_nomu[visualize_obs,:-1,agent] for agent in AGENT_LST], 0, drive_dir, "phi", [f"Agent {agent + 1}" for agent in AGENT_LST], comment = "", expand = False)
    else:
        visualize_comparison(TIMESTAMPS, [mu_st_nomu[visualize_obs,:], mu_st[visualize_obs,:], mu_st_truth[visualize_obs,:]], 0, drive_dir, "mu", ["$\mu$ Unknown", "$\mu$ Known", benchmark_name], comment = "")
        visualize_comparison(TIMESTAMPS, [sigma_st_nomu[visualize_obs,:], sigma_st[visualize_obs,:], sigma_st_truth[visualize_obs,:]], 0, drive_dir, "sigma", ["$\mu$ Unknown", "$\mu$ Known", benchmark_name], comment = "")
        # if utility_power == 1.5:
        visualize_comparison(TIMESTAMPS, [phi_dot_stn_nomu[visualize_obs,:,agent] for agent in AGENT_LST] + [phi_dot_stn[visualize_obs,:,agent] for agent in AGENT_LST] + [phi_dot_stn_truth[visualize_obs,:,agent] for agent in AGENT_LST], 0, drive_dir, "phi_dot", [f"$\mu$ Unknown\n - Agent {agent + 1}" for agent in AGENT_LST] + [f"$\mu$ Known\n - Agent {agent + 1}" for agent in AGENT_LST] + [f"{benchmark_name}\n - Agent {agent + 1}" for agent in AGENT_LST], comment = "", expand = False)
        visualize_comparison(TIMESTAMPS, [phi_stn_nomu[visualize_obs,:-1,agent] for agent in AGENT_LST] + [phi_stn[visualize_obs,:-1,agent] for agent in AGENT_LST] + [phi_stn_truth[visualize_obs,:-1,agent] for agent in AGENT_LST], 0, drive_dir, "phi", [f"$\mu$ Unknown\n - Agent {agent + 1}" for agent in AGENT_LST] + [f"$\mu$ Known\n - Agent {agent + 1}" for agent in AGENT_LST] + [f"{benchmark_name}\n - Agent {agent + 1}" for agent in AGENT_LST], comment = "", expand = False)
        # else:
        visualize_comparison(TIMESTAMPS, [phi_dot_stn_nomu[visualize_obs,:], phi_dot_stn[visualize_obs,:], phi_dot_stn_truth[visualize_obs,:]], 0, drive_dir, "phi_dot", ["$\mu$ Unknown", "$\mu$ Known", benchmark_name], comment = "", expand = True)
        visualize_comparison(TIMESTAMPS, [phi_stn_nomu[visualize_obs,:-1], phi_stn[visualize_obs,:-1], phi_stn_truth[visualize_obs,:-1]], 0, drive_dir, "phi", ["$\mu$ Unknown", "$\mu$ Known", benchmark_name], comment = "", expand = True)

## Begin Training
train_args = {
    "gen_hidden_lst": [50, 50, 50],
    "dis_hidden_lst": [50, 50, 50],
    "combo_hidden_lst": [50, 50, 50],
    "gen_lr": [1e-2, 1e-2, 1e-2, 1e-2, 1e-3],
    "gen_epoch": [500, 1000, 1000, 10000],#[500, 1000, 10000, 50000],
    "gen_decay": 0.1,
    "gen_scheduler_step": 10000,
    "dis_lr": [1e-2, 1e-2, 1e-2, 1e-2, 1e-3],
    "dis_epoch": [500, 1000, 1000, 10000],#[500, 2000, 10000, 50000],
    "dis_loss": [2, 2, 2],
    "utility_power": COST_POWER, #2,
    "dis_decay": 0.1,
    "dis_scheduler_step": 20000,
    "combo_lr": [1e-3],
    "combo_epoch": [100000],#[500, 1000, 10000, 50000],
    "combo_decay": 0.1,
    "combo_scheduler_step": 50000,
    "gen_sample": [1000],#[3000, 1000],
    "dis_sample": [1000],#[3000, 1000],
    "combo_sample": [128, 128],
    "y_coef_lst": [1],#[10, 10, 10, 10],
    "gen_solver": ["Adam"],
    "dis_solver": ["Adam"],
    "combo_solver": ["Adam"],
    "total_rounds": 10,#10,
    "normalize_up_to": 100,
    "visualize_obs": 0,
    "train_gen": True,
    "train_dis": True,
    "use_pretrained_gen": True,
    "use_pretrained_dis": True,
    "use_pretrained_combo": True,
    "use_true_mu": False,
    "use_fast_var": True,
    "last_round_dis": True,
    "seed": 0,
    "ckpt_freq": 10000,
    "use_combo": False,
    "clearing_known": False
}
# generator, discriminator = training_pipeline(train_args = train_args, **train_args)
# inference(generator, discriminator, randomized = False, clearing_known = train_args["clearing_known"])
plot_all_trajectories(**train_args)
