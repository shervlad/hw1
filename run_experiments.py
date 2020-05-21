import os
import argparse
import subprocess

exp1 = {
    'env'    : 'pusher',
    'suffix' : 'exp1',
    'rf'     :  '0',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '10'
}

exp2 = {
    'env'    : 'pusher',
    'suffix' : 'exp2',
    'rf'     :  '0',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '20'
}

exp3 = {
    'env'    : 'pusher',
    'suffix' : 'exp3',
    'rf'     :  '1',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '30'
}

exp4 = {
    'env'    : 'pusher',
    'suffix' : 'exp4',
    'rf'     :  '1',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '40'
}

exp5 = {
    'env'    : 'pusher',
    'suffix' : 'exp5',
    'rf'     :  '2',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '50'
}
exp6 = {
    'env'    : 'pusher',
    'suffix' : 'exp6',
    'rf'     :  '2',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '60'
}
exp7 = {
    'env'    : 'pusher',
    'suffix' : 'exp7',
    'rf'     :  '3',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '70'
}
exp8 = {
    'env'    : 'pusher',
    'suffix' : 'exp8',
    'rf'     :  '3',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '80'
}
exp9 = {
    'env'    : 'pusher',
    'suffix' : 'exp9',
    'rf'     :  '4',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '90'
}
exp10 = {
    'env'    : 'pusher',
    'suffix' : 'exp10',
    'rf'     :  '4',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '100'
}
exp11 = {
    'env'    : 'pusher',
    'suffix' : 'exp11',
    'rf'     :  '5',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '110'
}

exp12 = {
    'env'    : 'pusher',
    'suffix' : 'exp12',
    'rf'     :  '5',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '120'
}
exp13 = {
    'env'    : 'pusher',
    'suffix' : 'exp13',
    'rf'     :  '6',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '130'
}
exp14 = {
    'env'    : 'pusher',
    'suffix' : 'exp14',
    'rf'     :  '6',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '140'
}
exp15 = {
    'env'    : 'pusher',
    'suffix' : 'exp15',
    'rf'     :  '7',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '150'
}
exp16 = {
    'env'    : 'pusher',
    'suffix' : 'exp16',
    'rf'     :  '7',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '160'
}

Exp1 = {
    'env'    : 'reacher',
    'suffix' : 'exp1',
    'rf'     :  '0',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '10'
}

Exp2 = {
    'env'    : 'reacher',
    'suffix' : 'exp2',
    'rf'     :  '0',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '20'
}

Exp3 = {
    'env'    : 'reacher',
    'suffix' : 'exp3',
    'rf'     :  '1',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '30'
}

Exp4 = {
    'env'    : 'reacher',
    'suffix' : 'exp4',
    'rf'     :  '1',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '40'
}

Exp5 = {
    'env'    : 'reacher',
    'suffix' : 'exp5',
    'rf'     :  '2',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '50'
}

Exp6 = {
    'env'    : 'reacher',
    'suffix' : 'exp6',
    'rf'     :  '2',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '60'
}

Exp7 = {
    'env'    : 'reacher',
    'suffix' : 'exp7',
    'rf'     :  '3',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '70'
}

Exp8 = {
    'env'    : 'reacher',
    'suffix' : 'exp8',
    'rf'     :  '3',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '80'
}

Exp9 = {
    'env'    : 'reacher',
    'suffix' : 'exp9',
    'rf'     :  '4',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '90'
}

Exp10 = {
    'env'    : 'reacher',
    'suffix' : 'exp10',
    'rf'     :  '4',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '100'
}

Exp11 = {
    'env'    : 'reacher',
    'suffix' : 'exp11',
    'rf'     :  '5',
    'hparams':  './hyperparameters/hpprm1.csv',
    'seed'   :  '110'
}

Exp12 = {
    'env'    : 'reacher',
    'suffix' : 'exp12',
    'rf'     :  '5',
    'hparams':  './hyperparameters/hpprm2.csv',
    'seed'   :  '120'
}


pusher_experiments = [exp1, exp2, exp3, exp4, exp5, exp6,\
                      exp7, exp8, exp8, exp9, exp10, exp11,\
                      exp12, exp13, exp14, exp15, exp16, exp13 ]

reacher_experiments = [Exp1, Exp2, Exp3, Exp4, Exp5, Exp6,\
                      Exp7, Exp8, Exp8, Exp9, Exp10, Exp11, Exp12]

experiments = {'pusher':pusher_experiments,'reacher':reacher_experiments}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train reacher and pusher using PPO')

    parser.add_argument('--env')
    parser.add_argument('--experiments', type=int, nargs='+',
                        help='list of indices of the experiments you want to run')
    args = parser.parse_args()
    if(args.env!='pusher' and args.env != 'reacher'):
        print('--env ENV must be either pusher or reacher')
        raise

    for i in args.experiments:
        if i>=len(experiments[args.env]):
            print("experiment index %s is out of range\
                    indices must be in range [%s,%s]"%(i,0,len(experiments[args.env])))
            raise
    
    for i in args.experiments:
        exp = experiments[args.env][i]
        with open('./logs/log-exp%s.txt'%i, 'w') as f:
            process = subprocess.Popen(['python', 
                                        'train.py', 
                                        '--env',
                                        exp['env'],
                                        '--suffix',
                                        exp['suffix'],
                                        '--rf',
                                        exp['rf'],
                                        '--hparams',
                                        exp['hparams'],
                                        '--seed',
                                        exp['seed']],
                                        stdout=f)


    while True:
        output = process.stdout.readline()
        if(output == '' and process.poll() is not None):
            break
        if output:
            print(output.strip())