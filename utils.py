import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import os
import torch.nn as nn
import pickle
import csv
import time
from torch.distributions.normal import Normal
from ast import literal_eval as make_tuple

class GaussianMLP(torch.nn.Module):
    def __init__(self, dimensions, activation = nn.Tanh, output_activation=nn.Identity):
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
    def __init__(self,dimensions,activation = nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        layers = []

        for j in range(len(dimensions)-1):
            act = activation if j < len(dimensions)-2 else output_activation
            layers += [nn.Linear(dimensions[j], dimensions[j+1]), act()]

        self.perceptron =  nn.Sequential(*layers)

    def forward(self, state):
        return torch.squeeze(self.perceptron(state),-1)

def plot_pusher_policy(env = None, epoch = 0, pi = None, v_net = None, path=None):

    obj_pos=env._get_obs()[3:]
    bound = 1
    step = 0.1
    X, Y  = np.meshgrid(np.arange(-bound, bound+step, step), np.arange(-bound, bound+step, step))
    Z = np.ones(X.shape)


    def func(X,Y,Z):
        shape = X.shape
        X = X.reshape(*X.shape,1)
        Y = Y.reshape(*Y.shape,1)
        Z = Z.reshape(*Z.shape,1)
        coords = np.concatenate((X,Y,Z), axis=2).reshape((-1,3))
        states = np.array([])
        for c in coords:
            state  = np.concatenate([c,obj_pos]).reshape(-1,6)
            states = np.append(states,state).reshape(-1,*state.shape)
        o = torch.as_tensor(states, dtype=torch.float32)
        action = pi(o).mean.detach().numpy().reshape(-1,2)/20
        z = v_net(o).detach().numpy().reshape(shape)
        u = action[:,0].reshape(shape)
        v = action[:,1].reshape(shape)
        # d = 2*(u**2 + v**2)**0.5

        return u,v,z

    U,V,Z = func(X,Y,Z)

    d = 1
    C  = d*np.ones(X.shape)

    norm = mpl.colors.Normalize()
    norm.autoscale(C)
    cmap = cm.get_cmap('Greys')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(C)
    C = m.to_rgba(x = C,norm=True) 
    fig, ax = plt.subplots()

    z_min, z_max = Z.min(), Z.max()
    c = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=z_min, vmax=z_max)

    q = ax.quiver(X, Y, U, V, color='black', units='xy')
    ax.set_aspect('equal')


    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    plt.title('Epoch = %s'%epoch,fontsize=10)

    filename = path + 'plots/pi_%s'%(str(epoch))

    ax.scatter([env.goal[0]],[env.goal[1]])
    ax.scatter([env.init[0]],[env.init[1]], c='red')
    ax.scatter([obj_pos[0]],[obj_pos[1]], c='green')
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_reacher_policy(env = None, epoch = 0, pi = None, v_net = None, path=None):

    print("******************************8Observation Space: ",env.observation_space.shape)
    bound = 1
    step = 0.1
    X, Y  = np.meshgrid(np.arange(-bound, bound+step, step), np.arange(-bound, bound+step, step))
    Z = np.ones(X.shape)


    def func(X,Y,Z):
        shape = X.shape
        X = X.reshape(*X.shape,1)
        Y = Y.reshape(*Y.shape,1)
        Z = Z.reshape(*Z.shape,1)
        coords = np.concatenate((X,Y,Z), axis=2).reshape((-1,3))
        o = torch.as_tensor(coords, dtype=torch.float32)
        action = pi(o).mean.detach().numpy().reshape(-1,2)/20
        z = v_net(o).detach().numpy().reshape(shape)
        u = action[:,0].reshape(shape)
        v = action[:,1].reshape(shape)
        # d = 2*(u**2 + v**2)**0.5

        return u,v,z

    U,V,Z = func(X,Y,Z)

    d = 1
    C  = d*np.ones(X.shape)

    norm = mpl.colors.Normalize()
    norm.autoscale(C)
    cmap = cm.get_cmap('Greys')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(C)
    C = m.to_rgba(x = C,norm=True) 
    fig, ax = plt.subplots()

    z_min, z_max = Z.min(), Z.max()
    c = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=z_min, vmax=z_max)

    q = ax.quiver(X, Y, U, V, color='black', units='xy')
    ax.set_aspect('equal')


    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    plt.title('Epoch = %s'%epoch,fontsize=10)
    filename = path+'plots/pi_%s'%(str(epoch))
    ax.scatter([env.goal[0]],[env.goal[1]])
    ax.scatter([env.init[0]],[env.init[1]], c='red')
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()


def create_paths(env_name,suffix):
    path = './results/%s/experiment-%s/plots/'%(env_name,suffix)
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def load_hyperParameters(filename):
    h={}
    if filename.endswith('.pickle'):
        f = open(filename,"wb")
        h = pickle.load(f)
        f.close()
    elif(filename.endswith('.csv')):
        types = ['int','int','float','float','float','float','float','int','tuple','str']
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i,row in enumerate(reader):
                print(row)
                key = row[0]
                val = row[1]
                if types[i] == 'int':
                    val = int(val)
                if types[i] == 'float':
                    val = float(val)
                if types[i] == 'tuple':
                    val = make_tuple(val)
                h[key]=val
    else:
        print("Hyper Parameter file needs to be either .pickle or .csv")
        raise
    return h


def save_hyperParameters(h,path):
    filename = path + 'hyperparameters.pickle'
    f = open(filename,"wb")
    pickle.dump(h, f)
    f.close()


    filename = path + 'hyperparameters.csv'
    w = csv.writer(open(filename, "w"))
    for key, val in dict.items(h):
        w.writerow([key, val])



def save_pi_and_val(pi_network, v_network,path):

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

def plot_metrics(path,show=False):
    X            = []
    returns      = []
    min_reward   = []
    mean_reward  = []
    max_reward   = []
    min_val      = []
    mean_val     = []
    max_val      = []
    losses_pi    = []
    losses_v     = []

    # red dashes, blue squares and green triangles
    with open(path+'metrics.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            epoch,cum_r,min_r,mean_r,max_r,min_v,mean_v,max_v,mean_loss_pi,mean_loss_v = row
            X.append(int(epoch))
            returns.append(float(cum_r))
            min_reward.append(float(min_r))
            mean_reward.append(float(mean_r))
            max_reward.append(float(max_r))
            min_val.append(float(min_v))   
            mean_val.append(float(mean_v))
            max_val.append(float(max_v))
            losses_v.append(float(mean_loss_v))
            losses_pi.append(float(mean_loss_pi))


    figure, axes = plt.subplots(nrows=5, ncols=1)
    axes[0].set_ylabel("Return")
    axes[0].plot(X,returns)

    axes[1].set_ylabel("Reward")
    axes[1].plot(X,max_reward, c='blue',label='Max')
    axes[1].plot(X,mean_reward, label='Mean')
    axes[1].plot(X,min_reward, c='red', label='Min')

    axes[2].set_ylabel("Val")
    axes[2].plot(X,max_val, c='blue', label='Max')
    axes[2].plot(X,mean_val, label='Mean')
    axes[2].plot(X,min_val, c='red', label='Min')

    axes[3].set_ylabel("Loss V")
    axes[3].plot(X,losses_v)
    axes[4].set_ylabel("Loss Pi")
    axes[4].plot(X,losses_pi)

    for ax in axes:
        ax.set_xticks(ax.get_xticks()[::max(1,int(len(X)/10.0))])

    figure.tight_layout()
    if show:
        plt.show()
    plt.savefig(path+'metrics_plot.png')
    plt.close()

def save_rf(path,rf_str):
    with open(path+'reward_function.txt','w') as f:
        f.write(rf_str)