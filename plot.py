from utils import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run trained agent from previous experiments')

    parser.add_argument('-d',dest='dir',
                        help='specify a directory where the metrics.csv file is located')
    parser.add_argument('--show', dest='show', default=False, action='store_const', const=True,
                        help='when this flaf is set, the plot will show on the screen')


    args = parser.parse_args()


    plot_metrics(args.dir,args.show)



    
