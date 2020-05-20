import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal
import csv
class GaussianMLP(torch.nn.Module):
    def __init__(self, dimensions, activation = nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []

        for j in range(len(dimensions)-1):
            act = activation if j < len(dimensions)-2 else output_activation
            layers += [nn.Linear(dimensions[j], dimensions[j+1]), act()]

        self.log_std = nn.Parameter(torch.as_tensor(-0.5*np.ones(dimensions[-1],dtype=np.float32)))
        self.perceptron =  nn.Sequential(*layers)

    def forward(self, state):
        pi = self.perceptron(state)
        st_dev = torch.exp(self.log_std)
        return Normal(pi,st_dev)

class CategoricalMLP(torch.nn.Module):
    def __init__(self,dimensions,activation = nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []

        for j in range(len(dimensions)-1):
            act = activation if j < len(dimensions)-2 else output_activation
            layers += [nn.Linear(dimensions[j], dimensions[j+1]), act()]

        self.perceptron =  nn.Sequential(*layers)

    def forward(self, state):
        return torch.squeeze(self.perceptron(state),-1)

def discount(arr, gamma, last_val = 0):
    T = len(arr)
    discounted = [last_val]*(T+1)

    for i in range(T):
        discounted[-i-2] = arr[-i-1] + gamma*discounted[-i-1]

    return discounted[:-1]

def compute_advatages(states,rewards,vals,gamma,last_val):
    T = len(states)
    deltas = [0]*T
    for i in range(T):
        nextval = last_val
        if(i != T-1):
            nextval = vals[i+1]
        deltas[i] = rewards[i] + gamma*nextval - vals[i]

    a = discount(deltas,gamma,last_val)
    return a

def ppo(env_fn, num_epochs=10000, steps_per_epoch=400, gamma=0.98, lam = 0.95, epsilon = 0.2, 
        pi_step=0.0001, v_step = 0.001, sgd_iterations=20, plot_fn = None, path=None):
    env = env_fn()

    state_dims = env.observation_space.shape
    act_dims = env.action_space.shape

    hidden_sizes = (64,64)

    pi_network = GaussianMLP(state_dims + hidden_sizes + act_dims)
    v_network = CategoricalMLP(state_dims + hidden_sizes + (1,))

    pi_optimizer = Adam(pi_network.parameters(), lr = pi_step)
    v_optimizer = Adam(v_network.parameters(), lr = v_step)

    state = env.reset()

    for e in range(num_epochs):
        print("Epoch %s"%e)

        states    = np.zeros((steps_per_epoch, *state_dims), dtype=np.float32)
        actions   = np.zeros((steps_per_epoch, *act_dims), dtype=np.float32)
        rewards   = np.zeros(steps_per_epoch, dtype=np.float32)
        vals      = np.zeros(steps_per_epoch, dtype=np.float32)
        log_probs = np.zeros(steps_per_epoch, dtype=np.float32)

        for s in range(steps_per_epoch):
            pi = pi_network(torch.as_tensor(state, dtype=torch.float32))
            action = pi.sample()
            logp =  pi.log_prob(action).sum(axis=-1)

            new_state, reward, done, info = env.step(action)
            v = v_network(torch.as_tensor(state, dtype=torch.float32))

            states[s]    = state
            actions[s]   = action
            rewards[s]   = reward
            vals[s]      = v
            log_probs[s] = logp

            state = new_state

        last_val = v_network(torch.as_tensor(state, dtype=torch.float32))

        #update
        states     = torch.as_tensor(states, dtype=torch.float32)
        actions    = torch.as_tensor(actions, dtype=torch.float32)
        rewards    = torch.as_tensor(rewards, dtype=torch.float32)
        vals       = torch.as_tensor(vals, dtype=torch.float32)
        log_probs  = torch.as_tensor(log_probs, dtype=torch.float32)

        #compute advantages
        adv        = torch.as_tensor(compute_advatages(states,rewards,vals, gamma*lam, last_val), dtype=torch.float32)
        returns    = torch.as_tensor(discount(rewards,gamma,last_val), dtype=torch.float32)


        pi_losses = []
        v_losses  = []
        for i in range(sgd_iterations):

            #calculate loss pi
            pi_optimizer.zero_grad()
            pi = pi_network(states)
            new_log_probs = pi.log_prob(actions).sum(axis=-1)
            ratio = torch.exp(new_log_probs - log_probs)
            l_cpi = torch.clamp(ratio,1-epsilon,1+epsilon)*adv
            loss_pi = -1*(torch.min(ratio*adv, l_cpi)).mean()
            pi_losses.append(loss_pi.item())
            loss_pi.backward()
            pi_optimizer.step()
            
        for i in range(sgd_iterations):
            v_optimizer.zero_grad()
            loss_v = ((v_network(states) - returns)**2).mean()
            v_losses.append(loss_v.item())
            loss_v.backward()
            v_optimizer.step()

        mean_loss_pi = np.array(pi_losses).mean()
        mean_loss_v  = np.array(v_losses).mean()

        if(e%10 == 0):
            plot_fn(env = env, epoch = e, pi = pi_network, v_net=v_network)
            save_metrics(e,rewards,vals,mean_loss_pi,mean_loss_v,path)

        if(e%50 == 0 and path is not None):
            print("saving...")
            save(pi_network, v_network, path)
        state = env.reset()

def save(pi_network, v_network,path):
    torch.save(pi_network.state_dict(),path+"policy.pt")
    torch.save(v_network.state_dict(),path+"value_fn.pt")

def save_metrics(epoch,rewards,vals,mean_loss_pi,mean_loss_v,path):
    cum_r = np.sum(rewards.detach().numpy())
    min_r = rewards.min().item()
    max_r = rewards.max().item()
    mean_r = rewards.mean().item()
    min_v = vals.min().item()
    max_v = vals.max().item()
    mean_v = vals.mean().item()
    with open(path+'metrics.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([epoch,cum_r,min_r,mean_r,max_r,min_v,mean_v,max_v,mean_loss_pi,mean_loss_v])