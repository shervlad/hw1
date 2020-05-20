from spinup.utils.test_policy import load_policy_and_env, run_policy
from pusher import PusherEnv
from reacher import ReacherEnv
from ppo import CategoricalMLP, GaussianMLP
import torch
# _, get_action = load_policy_and_env('/home/vld/Desktop/hw1/reacher_results/')
# env = PusherEnv(render=True)

env = ReacherEnv(render=True)

state_dims = env.observation_space.shape
act_dims = env.action_space.shape
hidden_sizes = (64,64)

model = GaussianMLP(state_dims + hidden_sizes + act_dims)
model.load_state_dict(torch.load("./reacher_results/pyt_save/pi_network.pt"))
model.eval()

o = env.reset()

while(True):
    action = model(torch.as_tensor(o, dtype=torch.float32)).sample()
    o, reward, done, info = env.step(action)