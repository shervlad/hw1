from utils import *
import argparse
import time
from ppo import ppo
from pusher import PusherEnv
from reacher import ReacherEnv
import shutil

if __name__ == '__main__':

    defaultTimestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description='train reacher and pusher using PPO')
    parser.add_argument('--pusher', dest='env', action='store_const',
                        const='pusher', default='reacher',
                        help='start a training session')

    parser.add_argument('--hparams', dest='hparams',
                        help='(.csv or .pickle) specify the file that stores the hyperparameters you want to use for the trainig session ')

    parser.add_argument('--suffix', default=defaultTimestr,
                        help='specify a suffix to be appended to the saved metrics, Neural Net Parameters, and plots')

    parser.add_argument('--render', dest='render', default=False, action='store_const', const=True,
                        help='specify a suffix to be appended to the saved metrics, Neural Net Parameters, and plots')

    args = parser.parse_args()

    if(args.env!='pusher' and args.env!='reacher'):
        print('%s is not a valid environment. Has to be \"pusher\" or \"reacher\"' %args.env)

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

    while(os.path.exists(path)):
        print("An experiment with the suffix %s already exists."%args.suffix)
        answer = input("woudl you like to replace it? (yes/no)  ")
        while(answer != 'yes' and answer!='no'):
            answer = input("woudl you like to replace it? (yes/no)  ")
        if(answer=='yes'):
            shutil.rmtree(path)
        else:
            suffix = input("Type a new suffix:  ")
            path = './results/%s/experiment-%s/'%(args.env,suffix)


    create_paths(args.env,args.suffix)



    if(args.hparams is not None):
        filename = args.hparams
        if not os.path.exists(filename):
            print("The hypermarameter file you provided does not exist")
            raise
        h = load_hyperParameters(filename)


    save_hyperParameters(h,path)

    if(args.env == 'pusher'):

        h['env_fn'] = lambda : PusherEnv(render=args.render)
        h['plot_fn']   =  plot_pusher_policy

    elif(args.env == 'reacher'):


        h['env_fn'] = lambda : ReacherEnv(render=args.render)
        h['plot_fn']   =  plot_reacher_policy

    h['path'] = path
    ppo(**h)
