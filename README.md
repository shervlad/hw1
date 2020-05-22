train.py, run.py, run_experiments.py all have Command Line Interfaces.

To learn how to train:

>python train.py -h 

To learn how to run a trained network:

>python run.py -h

Examples how to run the policy you obtain from an experiment:
```
python run.py --env pusher -d ./results/pusher/experiment-exp1/

python run.py --env reacher -d ./results/reacher/experiment-exp2/
```
To learn how to train based on a predefined experiments:
```
python run_experiments.py -h

```

Results are saved in /results/ENV/experiment-NAME/

Every result folder contains:

hyperparameters.csv , hyperparameters.pickle - the hyperparameters used for this experiment

metrics.csv - the following metrics are saved every 10th epoch:

Cummulative Reward, Min Reward, Mean Reward, Max Reward, Min Val, Mean Val, Max Val,


plots/ : plots of the policy are saved every 100th epoch. The arroes represent the action at that position. The color represents the value function.
For the pusher policy the position of the object is fixed.


policy.pt - contains the saved parameters of the learned policy.
valye_fn.pt - contains the saved parameters of the learned value function
