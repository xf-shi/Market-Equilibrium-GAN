drive_dir = "." #"drive/MyDrive/CFRM/RL/SingleAgent-Stage2"

import json
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

## Global Constants
S_VAL = 245714618646 #1#

TR = 20
T = 100
TIMESTAMPS = np.linspace(0, TR, T + 1)[:-1]
DT = TR / T
N_SAMPLE = 300 #128
ALPHA = 1 #1 #
BETA = 0.5
GAMMA_BAR = 8.30864e-14 * S_VAL
KAPPA = 2.

GAMMA_1 = GAMMA_BAR*(KAPPA+1)/KAPPA
GAMMA_2 = GAMMA_BAR*(KAPPA+1)

XI_LIST = torch.tensor([3, -3]).float()
GAMMA_LIST = torch.tensor([GAMMA_1, GAMMA_2]).float()

S = 1
LAM = 1.08102e-10 * S_VAL #0.1 #

S_TERMINAL = 245.47
S_INITIAL = 250 #0#

BETA = GAMMA_BAR*S*ALPHA**2 + S_TERMINAL/TR

assert len(XI_LIST) == len(GAMMA_LIST) and torch.max(GAMMA_LIST) == GAMMA_LIST[-1]

GAMMA_BAR = 1 / torch.sum(1 / GAMMA_LIST)
GAMMA_MAX = torch.max(GAMMA_LIST)
N_AGENT = len(XI_LIST)

## Setup Numpy Counterparts
GAMMA_LIST_NP = GAMMA_LIST.numpy().reshape((1, N_AGENT))
XI_LIST_NP = XI_LIST.numpy().reshape((1, N_AGENT))
GAMMA_BAR_NP = GAMMA_BAR.numpy()
GAMMA_MAX_NP = GAMMA_MAX.numpy()
###

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
            ind = -1
            if n == m:
                ind += N_AGENT
            dRF[n, m] = (GAMMA_LIST_NP[:, m] * ind + GAMMA_MAX_NP) * const ** 2 / N_AGENT - np.matmul(RF[n,:-1], RF[:-1,m]) / LAM
    dR = np.hstack((dRH.reshape((-1,)), dRF.reshape((-1,))))
    return dR

def get_FH_exact():
    R_0 = np.zeros(N_AGENT * (N_AGENT + 1))
    timestamps = np.linspace(0, TR, T + 1)[:-1]
    res = solve_ivp(lambda t,R: InverseRiccati(t,R,LAM=LAM, GAMMA_BAR_NP=GAMMA_BAR_NP, GAMMA_LIST_NP=GAMMA_LIST_NP, GAMMA_MAX_NP=GAMMA_MAX_NP, ALPHA=ALPHA, N_AGENT=N_AGENT, XI_LIST_NP=XI_LIST_NP), t_span=[0, TR], y0=R_0, t_eval=timestamps)
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
            x = F.relu(x)
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
    def __init__(self, time_len, algo, input_dim, hidden_lst, output_dim, lr, decay, scheduler_step, use_s0 = False, solver = "Adam", retrain = False):
        assert solver in ["Adam", "SGD", "RMSprop"]
        assert algo in ["generator", "discriminator"]
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
        for _ in range(self.time_len):
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
        torch.save(model_save, f"{drive_dir}/Models/{self.algo}__{curr_ts}.pt")
        self.model = self.model.to(device=DEVICE)
        return curr_ts
    
    def load_latest(self):
        ts_lst = [f.strip(".pt").split("__")[1] for f in os.listdir(f"{drive_dir}/Models/") if f.endswith(".pt") and f.startswith(self.algo)]
        ts_lst = sorted(ts_lst, reverse=True)
        if len(ts_lst) == 0:
            return None, None
        ts = ts_lst[0]
        model = torch.load(f"{drive_dir}/Models/{self.algo}__{ts}.pt")
        model = model.to(device = DEVICE)
        return model, ts

class DynamicFactory():
    def __init__(self, dW_st, W_s0 = None):
        self.dW_st = dW_st
        self.W_st = get_W(dW_st, W_s0 = W_s0)
        self.n_sample = self.dW_st.shape[0]
        self.T = self.dW_st.shape[1]
        
        ## Auxiliary variables
        self.xi_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        for n in range(N_AGENT):
            self.xi_stn[:,:,n] = XI_LIST[n] * self.W_st[:,1:]
        self.mu_bar = GAMMA_BAR * ALPHA ** 2 * S
    
    def deep_hedging(self, gen_model, dis_model, use_true_mu = False, use_fast_var = False):
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
        stock_st[:,0] = dis_model((-1, dummy_one)).reshape((-1,))
        ## Begin iteration
        for t in range(self.T):
            curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
            ## Populate phi_bar
            for n in range(N_AGENT):
                phi_bar_stn[:,t,n] = self.mu_bar / GAMMA_LIST[n] / ALPHA ** 2 - XI_LIST[n] / ALPHA * self.W_st[:,t]
            delta_phi_stn = phi_stn[:,t,:] - phi_bar_stn[:,t,:]
            ## Discriminator output
            if not use_fast_var:
                x_dis = torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1)
            else:
                x_dis = torch.cat((delta_phi_stn, curr_t), dim=1)
            sigma_st[:,t] = dis_model((t, x_dis)).reshape((-1,))
            if use_true_mu:
                mu_st[:,t] = get_mu_from_sigma(sigma_st[:,t].reshape((self.n_sample, 1)), phi_stn[:,t,:].reshape((self.n_sample, 1, N_AGENT)), self.W_st[:,t].reshape((self.n_sample, 1))).reshape((-1,))
            else:
                if not use_fast_var:
                    x_mu = torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1)
                else:
                    x_mu = torch.cat((delta_phi_stn, curr_t), dim=1)
                mu_st[:,t] = dis_model((self.T + t, x_mu)).reshape((-1,))
            stock_st[:,t+1] = stock_st[:,t] + mu_st[:,t] * DT + sigma_st[:,t] * self.dW_st[:,t]
            ## Generator output
            for n in range(N_AGENT - 1):
                if not use_fast_var:
                    x_gen = torch.cat((phi_stn[:,t,:], self.W_st[:,t].reshape((self.n_sample, 1)), curr_t), dim=1)
                else:
                    x_gen = torch.cat((delta_phi_stn, curr_t), dim=1)
                phi_dot_stn[:,t,n] = gen_model((n * self.T + t, x_gen)).reshape((-1,))
                phi_stn[:,t+1,n] = phi_stn[:,t,n] + phi_dot_stn[:,t,n] * DT
            phi_dot_stn[:,t,-1] = -torch.sum(phi_dot_stn[:,t,:-1], axis = 1)
            phi_stn[:,t+1,-1] = S - torch.sum(phi_stn[:,t+1,:-1], axis = 1)
        return phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st
    
    def leading_order(self):
        pass
    
    def ground_truth(self, F_exact, H_exact):
        ## Setup variables
        phi_dot_stn = torch.zeros((self.n_sample, self.T, N_AGENT)).to(device = DEVICE)
        phi_stn = torch.zeros((self.n_sample, self.T + 1, N_AGENT)).to(device = DEVICE)
        phi_stn[:,0,:] = S * GAMMA_BAR / GAMMA_LIST
        mu_st = torch.zeros((self.n_sample, self.T)).to(device = DEVICE)
        stock_st = torch.zeros((self.n_sample, self.T + 1)).to(device = DEVICE)
        ones = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        ## Begin iteration
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
    
    def pasting(self):
        pass

class LossFactory():
    def __init__(self, dW_st, W_s0 = None):
        self.dW_st = dW_st
        self.W_st = get_W(dW_st, W_s0 = W_s0)
        self.n_sample = self.dW_st.shape[0]
        self.T = self.dW_st.shape[1]
    
    def utility_loss(self, phi_dot_stn, phi_stn, mu_st, sigma_st):
        loss = 0
        for n in range(N_AGENT):
            loss += (torch.mean(-torch.sum(mu_st * phi_stn[:,1:,n], axis = 1) + GAMMA_LIST[n] / 2 * torch.sum((sigma_st * phi_stn[:,1:,n] + self.W_st[:,1:] * XI_LIST[n]) ** 2, axis = 1) + LAM / 2 * torch.sum(phi_dot_stn[:,:,n] ** 2, axis = 1)) / self.T) / N_AGENT
        return loss
    
    def stock_loss(self, stock_st, power = 2):
        target = BETA * TR + ALPHA * self.W_st[:,-1]
        loss = torch.abs(stock_st[:,-1] - target) ** power
        return torch.mean(loss)

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

## Visualize the comparison of dynamics
def visualize_comparison(timestamps, arr_lst, round, ts, name, algo_lst, comment = None):
    assert name in ["phi", "phi_dot", "sigma", "mu", "s"]
    round += 1
    if name == "phi":
        title = "${\\varphi}_t$"
    elif name == "phi_dot":
        title = "$\dot{\\varphi}_t$"
    elif name == "sigma":
        title = "$\sigma_t$"
    elif name == "mu":
        title = "$\mu_t$"
    else:
        title = "$S_t$"
    if comment is not None:
        title2 = title + "\n" + str(comment)
    else:
        title2 = title
    if name == "phi_dot":
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
                    ax.plot(timestamps, arr[:, i], label = f"{algo} - Agent {i + 1}")
            ax.set_xlabel("T")
            ax.set_ylabel(title)
            ax.set_title(title2)
            ax.grid()
            box2 = ax.get_position()
            ax.legend(loc="lower left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
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
                    ax.plot(timestamps, arr[:, i], label = f"{algo} - Agent {i + 1}")
        ax.set_xlabel("T")
        ax.set_ylabel(title)
        ax.set_title(title2)
        ax.grid()
        box2 = ax.get_position()
        ax.legend(loc="lower left", bbox_to_anchor=(box2.width*1.3,box2.height*0.5))
        plt.savefig(f"{drive_dir}/Plots/comp_round={round}_{name}_{ts}.png", bbox_inches='tight')
        plt.close()

def prepare_generator(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver = "Adam", use_pretrained_gen = True, use_fast_var = False):
    retrain = not use_pretrained_gen
    if not use_fast_var:
        input_dim = 2 + N_AGENT
    else:
        input_dim = 1 + N_AGENT
    model_factory = ModelFactory(T, "generator", input_dim, gen_hidden_lst, 1, gen_lr, gen_decay, gen_scheduler_step, use_s0 = False, solver = gen_solver, retrain = retrain)
    return model_factory

def prepare_discriminator(dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver = "Adam", use_pretrained_dis = True, use_true_mu = False, use_fast_var = False):
    n_model = T + 1
    if not use_true_mu:
        n_model += T
    retrain = not use_pretrained_dis
    if not use_fast_var:
        input_dim = 2 + N_AGENT
    else:
        input_dim = 1 + N_AGENT
    model_factory = ModelFactory(n_model, "discriminator", input_dim, dis_hidden_lst, 1, dis_lr, dis_decay, dis_scheduler_step, use_s0 = True, solver = dis_solver, retrain = retrain)
    return model_factory

def slc(lst, idx):
    return lst[min(idx, len(lst) - 1)]

def train_single(generator, discriminator, optimizer, scheduler, epoch, sample_size, use_true_mu, use_fast_var, train_type, F_exact, H_exact, dis_loss = 1, ckpt_freq = 10000, model_factory = None, curr_ts = None):
    assert train_type in ["generator", "discriminator"]
    loss_arr = []
    for itr in tqdm(range(epoch)):
        optimizer.zero_grad()
        dW_st = torch.normal(0, np.sqrt(DT), (sample_size, T))
        dynamic_factory = DynamicFactory(dW_st)
        loss_factory = LossFactory(dW_st)
        phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var)
        if train_type == "generator":
            loss = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st)
        else:
            loss = loss_factory.stock_loss(stock_st, power = dis_loss)
        assert not torch.isnan(loss.data)
        loss.backward()
        if train_type == "generator":
            loss = loss * S_VAL
        loss_arr.append(float(loss.data))
        optimizer.step()
        scheduler.step()
        ## Save checkpoint frequently
        if itr % ckpt_freq == 0 and itr > 0:
            if train_type == "generator":
                model_factory.update_model(generator)
            else:
                model_factory.update_model(discriminator)
            model_factory.save_to_file(curr_ts)
    phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.ground_truth(F_exact, H_exact)
    if train_type == "generator":
        model = generator
        loss_truth_final = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st) * S_VAL
    else:
        model = discriminator
        loss_truth_final = loss_factory.stock_loss(stock_st, power = dis_loss)
    return model, loss_arr, loss_truth_final

def training_pipeline(gen_hidden_lst, gen_lr, gen_decay, gen_scheduler_step, gen_solver, gen_epoch, gen_sample, use_pretrained_gen, dis_hidden_lst, dis_lr, dis_decay, dis_scheduler_step, dis_solver, dis_epoch, dis_sample, use_pretrained_dis, dis_loss = [1], use_true_mu = False, use_fast_var = False, total_rounds = 1, train_gen = True, train_dis = True, last_round_dis = True, visualize_obs = 0, seed = 0, ckpt_freq = 10000, train_args = None):
    ## Generate Brownian paths for testing
    torch.manual_seed(seed)
    dW_st_eval = torch.normal(0, np.sqrt(DT), size = (N_SAMPLE, T))
    F_exact, H_exact = get_FH_exact()
    ## Begin training
    for gan_round in range(total_rounds):
        print(f"Round #{gan_round + 1}:")
        curr_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
        model_factory_gen = prepare_generator(gen_hidden_lst, slc(gen_lr, gan_round), gen_decay, gen_scheduler_step, gen_solver = slc(gen_solver, gan_round), use_pretrained_gen = use_pretrained_gen or not train_gen or gan_round > 0, use_fast_var = use_fast_var)
        model_factory_dis = prepare_discriminator(dis_hidden_lst, slc(dis_lr, gan_round), dis_decay, dis_scheduler_step, dis_solver = slc(dis_solver, gan_round), use_pretrained_dis = use_pretrained_dis or not train_dis or gan_round > 0, use_true_mu = use_true_mu, use_fast_var = use_fast_var)
        generator, optimizer_gen, scheduler_gen, prev_ts_gen = model_factory_gen.prepare_model()
        discriminator, optimizer_dis, scheduler_dis, prev_ts_dis = model_factory_dis.prepare_model()
        if train_gen:
            print("\tTraining Generator...")
            generator, loss_arr_gen, loss_truth_final_gen = train_single(generator, discriminator, optimizer_gen, scheduler_gen, slc(gen_epoch, gan_round), slc(gen_sample, gan_round), use_true_mu, use_fast_var, "generator", F_exact, H_exact, dis_loss = slc(dis_loss, gan_round), ckpt_freq = ckpt_freq, model_factory = model_factory_gen, curr_ts = curr_ts)
            model_factory_gen.update_model(generator)
            model_factory_gen.save_to_file(curr_ts)
            visualize_loss(loss_arr_gen, gan_round, "generator", curr_ts, loss_truth_final_gen)
        if train_dis and (gan_round < total_rounds - 1 or last_round_dis):
            print("\tTraining Discriminator...")
            discriminator, loss_arr_dis, loss_truth_final_dis = train_single(generator, discriminator, optimizer_dis, scheduler_dis, slc(dis_epoch, gan_round), slc(dis_sample, gan_round), use_true_mu, use_fast_var, "discriminator", F_exact, H_exact, dis_loss = slc(dis_loss, gan_round), ckpt_freq = ckpt_freq, model_factory = model_factory_dis, curr_ts = curr_ts)
            model_factory_dis.update_model(discriminator)
            model_factory_dis.save_to_file(curr_ts)
            visualize_loss(loss_arr_dis, gan_round, "discriminator", curr_ts, loss_truth_final_dis)
        ## Evaluation per round
        dynamic_factory = DynamicFactory(dW_st_eval)
        loss_factory = LossFactory(dW_st_eval)
        phi_dot_stn, phi_stn, mu_st, sigma_st, stock_st = dynamic_factory.deep_hedging(generator, discriminator, use_true_mu = use_true_mu, use_fast_var = use_fast_var)
        loss_utility = loss_factory.utility_loss(phi_dot_stn, phi_stn, mu_st, sigma_st) * S_VAL
        loss_stock = loss_factory.stock_loss(stock_st, power = slc(dis_loss, gan_round))

        phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth, stock_st_truth = dynamic_factory.ground_truth(F_exact, H_exact)
        loss_truth_utility = loss_factory.utility_loss(phi_dot_stn_truth, phi_stn_truth, mu_st_truth, sigma_st_truth) * S_VAL
        loss_truth_stock = loss_factory.stock_loss(stock_st_truth, power = slc(dis_loss, gan_round))
        ## Visualize
        comment = f"Model Utility Loss = {loss_utility:.2e}, Stock Loss = {loss_stock:.2e}\nGround Truth Utility Loss = {loss_truth_utility:.2e}, Stock Loss = {loss_truth_stock:.2e}\n"
        visualize_comparison(TIMESTAMPS, [phi_dot_stn[visualize_obs,:], phi_dot_stn_truth[visualize_obs,:]], gan_round, curr_ts, "phi_dot", ["Model", "Ground Truth"], comment = comment)
        visualize_comparison(TIMESTAMPS, [phi_stn[visualize_obs,1:], phi_stn_truth[visualize_obs,1:]], gan_round, curr_ts, "phi", ["Model", "Ground Truth"], comment = comment)
        visualize_comparison(TIMESTAMPS, [mu_st[visualize_obs,:], mu_st_truth[visualize_obs,:]], gan_round, curr_ts, "mu", ["Model", "Ground Truth"], comment = comment)
        visualize_comparison(TIMESTAMPS, [sigma_st[visualize_obs,:], sigma_st_truth[visualize_obs,:]], gan_round, curr_ts, "sigma", ["Model", "Ground Truth"], comment = comment)
        visualize_comparison(TIMESTAMPS, [stock_st[visualize_obs,1:], stock_st_truth[visualize_obs,1:]], gan_round, curr_ts, "s", ["Model", "Ground Truth"], comment = comment)
        ## Save logs to file
        write_logs([prev_ts_gen, curr_ts], train_args)

def transfer_learning():
    pass

## Begin Training
train_args = {
    "gen_hidden_lst": [50, 50, 50],
    "dis_hidden_lst": [50, 50, 50],
    "gen_lr": [1e-2, 1e-2, 1e-2],
    "gen_epoch": [100, 100, 500, 1000, 5000, 10000],
    "gen_decay": 0.1,
    "gen_scheduler_step": 5000,
    "dis_lr": [1e-2, 1e-2, 1e-2],
    "dis_epoch": [100,100, 500, 2000, 10000, 20000],
    "dis_loss": [1],
    "dis_decay": 0.1,
    "dis_scheduler_step": 10000,
    "gen_sample": [128, 128],
    "dis_sample": [128, 128],
    "gen_solver": ["Adam"],
    "dis_solver": ["Adam"],
    "total_rounds": 2,
    "visualize_obs": 0,
    "train_gen": True,
    "train_dis": True,
    "use_pretrained_gen": False,
    "use_pretrained_dis": False,
    "use_true_mu": False,
    "use_fast_var": False,
    "last_round_dis": False,
    "seed": 0,
    "ckpt_freq": 10000
}
training_pipeline(train_args = train_args, **train_args)
