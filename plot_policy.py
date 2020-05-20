import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch

def plot_pusher_policy(obj_pos, env = None, epoch = 0, pi = None, v_net = None):

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

    q = ax.quiver(X, Y, U, V,C, units='xy')
    ax.set_aspect('equal')


    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    plt.title('Epoch = %s'%epoch,fontsize=10)

    filename = './pusher_results/plots/pi_' + str(epoch)
    ax.scatter([env.goal[0]],[env.goal[1]])
    ax.scatter([env.init[0]],[env.init[1]], c='red')
    ax.scatter([obj_pos[0]],[obj_pos[1]], c='green')
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_reacher_policy(env = None, epoch = 0, pi = None, v_net = None):

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
        action = pi(o).mean.detach().numpy()/20
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

    q = ax.quiver(X, Y, U, V,C, units='xy')
    ax.set_aspect('equal')


    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    plt.title('Epoch = %s'%epoch,fontsize=10)

    filename = './reacher_results/plots/pi_' + str(epoch)
    ax.scatter([env.goal[0]],[env.goal[1]])
    ax.scatter([env.init[0]],[env.init[1]], c='red')
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()
