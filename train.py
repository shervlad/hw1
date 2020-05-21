from utils import *
import argparse
import time
from ppo import ppo
from pusher import PusherEnv
from reacher import ReacherEnv
import shutil
from reward_functions import reward_functions


def main():
    defaultTimestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description='train reacher and pusher using PPO')
    parser.add_argument('--env',
                        help='[pusher|reacher] - the environment you want to train')

    parser.add_argument('--hparams',dest='hparams',
                        help='(.csv or .pickle) specify the file that stores the hyperparameters you want to use for the trainig session ')

    parser.add_argument('--suffix', default=defaultTimestr,
                        help='will be appended to the results folder. \
                                Serves as a unique identifier for this experiment.  \n\
                                the results will be saved in ./results/ENV/experiment-SUFFIX/')

    parser.add_argument('--render', dest='render', default=False, action='store_const', const=True,
                        help='set this flag if you want to render the environment')

    parser.add_argument('--rf', dest='rf', default=0, type=int, 
                        help='specify the index of the reward function you want to use.\n \
                                --env ENV --show_rfs to see rfs available for ENV')
    parser.add_argument('--show_rfs', dest='show_rfs', default=False, action='store_const', const=True,
                        help='view abailable reward functions')

    parser.add_argument('--seed',  default=0, type=int, 
                        help='Seed for initializing random parameters')
    args = parser.parse_args()

    if(args.env!='pusher' and args.env!='reacher'):
        print('%s is not a valid environment. Has to be \"pusher\" or \"reacher\"' %args.env)
    
    if(args.show_rfs):
        for i,rfstr in enumerate(reward_functions[args.env]['str']):
            print("%s  :  %s"%(i,rfstr))
        return

    if(args.rf >= len(reward_functions[args.env]['rfs'])):
        print("Reward function index out of range!\
                --show_rfs to view available reward functions")
        raise


    rf     = reward_functions[args.env]['rfs'][args.rf]
    rf_str = reward_functions[args.env]['str'][args.rf]
    h = {
        'num_epochs'          : 10000,
        'steps_per_epoch'     : 400,
        'gamma'               : 0.98,
        'lam'                 : 0.95,
        'epsilon'             : 0.2,
        'pi_step'             : 0.0001,
        'v_step'              : 0.001,
        'sgd_iterations'      : 20,
        'hidden_sizes'        : (64,64)
        }

    path = './results/%s/experiment-%s/'%(args.env,args.suffix)

    while(os.path.exists(path) and len(os.listdir(path))>0):
        print("An experiment with the suffix %s already exists."%args.suffix)
        answer = input("woudl you like to replace it? (yes/no)  ")
        while(answer != 'yes' and answer!='no'):
            answer = input("woudl you like to replace it? (yes/no)  ")
        if(answer=='yes'):
            shutil.rmtree(path)
        else:
            suffix = input("Type a new suffix:  ")
            path = './results/%s/experiment-%s/'%(args.env,suffix)



    if(args.hparams is not None):
        filename = args.hparams
        if not os.path.exists(filename):
            print("The hypermarameter file you provided does not exist")
            raise
        h = load_hyperParameters(filename)


    h['seed'] = args.seed
    create_paths(args.env,args.suffix)
    save_hyperParameters(h,path)
    h['path'] = path
    save_rf(path,rf_str)
    if(args.env == 'pusher'):

        h['env_fn'] = lambda : PusherEnv(render=args.render,reward_fn=rf)
        h['plot_fn']   =  plot_pusher_policy

    elif(args.env == 'reacher'):

        h['env_fn'] = lambda : ReacherEnv(render=args.render,reward_fn=rf)
        h['plot_fn']   =  plot_reacher_policy

    ppo(**h)

if __name__ == '__main__':
    main()