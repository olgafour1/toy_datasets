import argparse
import os
import random

import numpy as np
import tensorflow as tf
import ast

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train Graph Att net')

    parser.add_argument('--siam_lr', dest='siam_init_lr',
                        help='initial learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--prob', dest='prob',
                        help='initial learning rate',
                        default=0.5, type=float)
    parser.add_argument('--mode', dest='mode',
                        help='select eucidean or siamese distance',
                        choices=["euclidean","siamese"],
                        default="siamese", type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory where the weights of the model are stored',
                        default="Saved_model", type=str)
    parser.add_argument("--weight_file", default=False,
                        action="store_true",
                        help="true if there is a weight file")
    parser.add_argument('--siamese_weights_path', dest='siamese_weights_path',
                        help='directory where the images are stored',
                        default="colon_weights", type=str)
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=1e-4, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train GRAPH MIL',
                        default=100, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)
    parser.add_argument('--seed_value', dest='seed_value',
                        help='use same seed value for reproducability',
                        default=12321, type=int)
    parser.add_argument('--run', dest='run',
                        help='number of experiments to be run',
                        default=5, type=int)
    parser.add_argument('--k', dest='k',
                        help='number of neighbors taken into account',
                        default=3, type=int,
                        choices=range(1,11), metavar="[1-11]")
    parser.add_argument('--folds', dest='n_folds',
                        help='number of folds for cross fold validation',
                        default=10, type=int)
    parser.add_argument('--data_path', dest='data_path',
                        help='directory where the images are stored',
                        default="ColonCancer", type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',
                        help='the name of the experiment needed for the logs',
                        default="eiclidean_aug", type=str)
    parser.add_argument('--siam_batch_size', dest='siam_batch_size',
                        help='batch size used bu the siamese network',
                        default=128, type=int)
    parser.add_argument('--siam_k', dest='siam_k',
                        help='neighbours for siamese network',
                        default=3, type=int)
    parser.add_argument('--siam_epochs', dest='siam_epochs',
                        help='epochs for siamese network',
                        default=50, type=int)
    parser.add_argument('--siam_pixel_distance', dest='siam_pixel_dist',
                        help=' threshold for all patch_dist<= siam_pixel_dist closest neighbors to x_i,'
                             ' (x_i, k) is considered a positive pair by siamese net',
                        default=20, type=int)
    parser.add_argument('--extention', dest='ext',
                        help='shape of the image',
                        default="bmp", type=str)
    parser.add_argument('--input_shape', dest="input_shape",
                        help='shape of the image',
                        default=(27, 27, 3), type=int, nargs=3)
    parser.add_argument('--data', dest="data",
                        help='name of the dataset',
                        choices=["colon","ucsb"],
                        default="colon", type=str)
    parser.add_argument('--pooling_mode', dest="pooling_mode",
                        help='pooling operator selected',
                        choices=["max", "ave","sum"],
                        default="sum", type=str)
    parser.add_argument('--arch', help='list containing a list of the networks layers',
                        type=ast.literal_eval,
                        default=[
                            {'type': 'Conv2D', "channels": 36, 'kernel': (4, 4)},
                            {'type': 'MaxPooling2D', 'pool_size': (2, 2)},
                            {'type': 'Conv2D', "channels": 48, 'kernel': (3, 3)},
                            {'type': 'MaxPooling2D', 'pool_size': (2, 2)},
                            {'type': 'Flatten'},
                            {'type': 'relu', 'size': 512},
                            {'type': 'Dropout', 'rate': 0.5},
                            {'type': 'relu', 'size': 512},
                            {'type': 'Dropout', 'rate': 0.5}
                        ])


    args = parser.parse_args()
    return args


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
