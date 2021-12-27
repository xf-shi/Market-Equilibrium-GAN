# Market-Equilibrium-GAN
This repo contains the framework for obtaining optimal trading strategies among multiple agents while finding stock market equilibrium. The framework is implemented using the generative adversarial infrastructure with deep neural networks discretized by time steps.

## Basic Setup

The special case with following assumptions is considered:

* the dynamic of the market satisfies that return <img src="https://latex.codecogs.com/gif.latex?\mu_t" /> and voalatility <img src="https://latex.codecogs.com/gif.latex?\sigma_t" /> are unknown;
* the cost parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" /> for each trading agent is constant, but different agents might have different <img src="https://latex.codecogs.com/gif.latex?\lambda" />;
* the endowment volatility is in the form of <img src="https://latex.codecogs.com/gif.latex?\xi_t={\xi}W_t" /> where <img src="https://latex.codecogs.com/gif.latex?{\xi}" /> is constant for each trading agent. Again, different trading agents might have different endowment volatilities; 
* the frictionless strategy satisfies that   <img src="https://latex.codecogs.com/gif.latex?\bar{b}_t=0" /> and <img src="https://latex.codecogs.com/gif.latex?\bar{a}_t=-{\xi}/{\sigma}" />

The general variables and the market parameters and their corresponding values in the code are summarized below:
| Variable | Value | Meaning |
| --- | --- | --- |
| `q`  | 2 | power of the trading cost, q |
| `S_OUTSTANDING` | 1 | total shares in the market, s |
| `TIME` | 1 | trading horizon, T |
| `TIME_STEP` | 252 |  time discretization, N |
| `DT ` | 1/252 | <img src="https://latex.codecogs.com/gif.latex?\Delta%20t={T}/{N}" />  |
| `GAMMA` | <img src="https://latex.codecogs.com/gif.latex?\gamma_1=1" />, <img src="https://latex.codecogs.com/gif.latex?\gamma_2=1" />, <img src="https://latex.codecogs.com/gif.latex?\gamma_3=2" /> | risk aversion, <img src="https://latex.codecogs.com/gif.latex?\gamma" /> |
| `XI` | <img src="https://latex.codecogs.com/gif.latex?{\xi}_1=1" />, <img src="https://latex.codecogs.com/gif.latex?{\xi}_2=2" />, <img src="https://latex.codecogs.com/gif.latex?{\xi}_3=-3" /> | endowment volatility parameter, <img src="https://latex.codecogs.com/gif.latex?{\xi}" /> |
| `PHI_INITIAL` | - | initial holding,  <img src="https://latex.codecogs.com/gif.latex?\varphi_{0-}" /> |
| `ALPHA` | 1 | market volatility,  <img src="https://latex.codecogs.com/gif.latex?\sigma " /> |
| `MU_BAR` | - | market return,  <img src="https://latex.codecogs.com/gif.latex?\mu " /> |
| `LAM` | 0.1 | trading cost parameter, <img src="https://latex.codecogs.com/gif.latex?\lambda " /> |
| `test_samples` | 300 | number of test sample path, batch_size |
