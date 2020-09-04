from pusher import PusherEnv
from reacher import ReacherEnv
from reacher_wall import ReacherWallEnv
from ppo import CategoricalMLP, GaussianMLP
import torch
import argparse
import os
from utils import *


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='run trained agent from previous experiments')

    parser.add_argument('-d',dest='dir', default=None,
                        help='specify a directory where the models and the hyperparams are savaed\n\
                                example: -d ./results/pusher/experiment-example/')

    parser.add_argument('--env', help='ENV = [pusher|reacher]')

    parser.add_argument('--suffix', default=None,
                        help='specify a suffix corresponding to the experiment you want to run\n\
                               the results will be saved in ./results/ENV/experiment-SUFFIX/')
    parser.add_argument('-n', dest='episodes', default=10, type=int,
                        help='number of episodes you want to run')

    args = parser.parse_args()

    path = ""
    if(args.dir is not None):
        path = args.dir
    elif(args.suffix is not None):
        path = '.results/%s/experiment-%s/'%(args.env,suffix)
    else:
        print("Either the path (-d) or the suffix (--suffix) need to e specified")
        raise


    if(args.env == 'reacher'):
        env=ReacherEnv(render=True)
    elif(args.env == 'pusher'):
        env = PusherEnv(render=True)
    elif(args.env == 'reacher_wall'):
        env = ReacherWallEnv(render=True)
    else:
        print("Environment should be either --reacher or --pusher")
        raise




    filename = path + 'hyperparameters.csv'
    if not os.path.exists(filename):
        print("No experiment at %s"%(path))
        raise

    h = load_hyperParameters(filename)

    state_dims = env.observation_space.shape
    act_dims = env.action_space.shape
    hidden_sizes = h['hidden_sizes']

    model = GaussianMLP(state_dims + hidden_sizes + act_dims)
    model.load_state_dict(torch.load(path + 'policy.pt'))
    model.eval()

    for i in range(args.episodes):
        o = env.reset()
        for j in range(h['steps_per_epoch']):
            action = model(torch.as_tensor(o, dtype=torch.float32)).mean.detach()
            o, reward, done, info = env.step(action)
