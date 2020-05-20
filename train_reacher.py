from pusher import PusherEnv
from reacher import ReacherEnv
from plot_policy import plot_reacher_policy
import random
#from spinup import ppo_pytorch as ppo
from ppo import ppo
import time
import torch
# env_fn = lambda : PusherEnv(render=False)
env_fn = lambda : ReacherEnv(render=False)

ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)
logger_kwargs = dict(output_dir='reacher_results', exp_name='pusher_ppo')
# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)
#ppo(env_fn=env_fn, 
#    ac_kwargs=ac_kwargs, 
#    seed=0, 
#    steps_per_epoch=1000, 
#    epochs=100, 
#    gamma=0.98, 
#    clip_ratio=0.2, 
#    pi_lr=3e-4,
#    vf_lr=1e-3, 
#    train_pi_iters=80, 
#    train_v_iters=80, 
#    lam=0.97, 
#    max_ep_len=1000,
#    target_kl=0.1, 
#    logger_kwargs=logger_kwargs,
#    save_freq=10)
plot_fn = lambda env, epoch, pi, v_net : plot_reacher_policy(env=env, epoch = epoch, pi=pi, v_net=v_net)
ppo(env_fn=env_fn, plot_fn = plot_fn, path='./reacher_results/model/')