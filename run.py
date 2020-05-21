from pusher import PusherEnv
from reacher import ReacherEnv
from ppo import CategoricalMLP, GaussianMLP
import torch
import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='run trained agent from previous experiments')

    parser.add_argument('--pusher', dest='env', action='store_const',
                        const='pusher', default='reacher',
                        help='run a pusher environment')

    parser.add_argument('--suffix', default=None,
                        help='specify a suffix corresponding to the experiment you want to run')

    args = parser.parse_args()

    suffix = args.suffix
    while(suffix is None):
        print("You must provide the suffix of the experiment you want to run")
        suffix = input("Enter a suffix...  ")

    if(args.env!='pusher' and args.env!='reacher'):

    env = PusherEnv(render=True)

    if(args.env == 'reacher'):
        env=ReacherEnv(render=True)


    path = '.results/%s/experiment-%s/'%(args.env,suffix)
    if not os.path.exists(filename):
        print("No %s Experiment with suffix \"%s\" was found."%(args.env, suffix))
        return

    filename = path + 'hyperparameters.csv'
    h = load_hyperParameters(filename)

    state_dims = env.observation_space.shape
    act_dims = env.action_space.shape
    hidden_sizes = h['hidden_sizes']

    model = GaussianMLP(state_dims + hidden_sizes + act_dims)
    model.load_state_dict(torch.load(h['path'] + 'policy.pt'))
    model.eval()

    o = env.reset()
    while(True):
        action = model(torch.as_tensor(o, dtype=torch.float32)).sample()
        o, reward, done, info = env.step(action)