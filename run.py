from args import parse_args, set_seed
from utl.dataset import load_files
from utl.networks import GraphAttnet
from cell_segmenation_util import *
import time
import numpy as np

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def model_training(args,dataset, irun, ifold,detection_model):
    train_bags = dataset['train']
    test_bags = dataset['test']

    net = GraphAttnet(args, useMulGpue=False)

    t1 = time.time()

    net.train(train_bags, irun=irun, ifold=ifold,detection_model=detection_model)

    test_net = GraphAttnet(args, useMulGpue=False)
    test_loss, test_acc, auc, precision, recall = net.predict(test_bags,detection_model=detection_model, test_model=test_net.model,irun=irun, ifold=ifold)

    t2 = time.time()
    print('run time:', (t2 - t1) / 60.0, 'min')
    print('test_acc={:.3f}'.format(test_acc))

    return test_loss, test_acc, recall, precision, auc


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(args.seed_value)

    acc = np.zeros((args.run, args.n_folds), dtype=float)
    precision = np.zeros((args.run, args.n_folds), dtype=float)
    recall = np.zeros((args.run, args.n_folds), dtype=float)
    auc = np.zeros((args.run, args.n_folds), dtype=float)
    loss = np.zeros((args.run, args.n_folds), dtype=float)
    
    for irun in range(args.run):
        if args.data=="ucsb":
            model_path = os.path.join(os.getcwd(), 'segm_models')
            model = restored_model(os.path.join(model_path,'nucles_model_v3.meta'), model_path)
        else:
            model=None
        datasets = load_files(dataset_path=args.data_path,n_folds=args.n_folds, rand_state=irun,ext=args.ext)
        
        for ifold in range(args.n_folds):

            print('run=', irun, '  fold=', ifold)

            loss[irun][ifold],acc[irun][ifold], recall[irun][ifold], precision[irun][ifold], auc[irun][ifold] = \
                model_training(args,datasets[ifold], irun=irun, ifold=ifold, detection_model=model)
        if model:
            model.close_sess()
        print("k-nn without siamese")
        print("number of neighbors used {}:".format(args.k))
        print('mi-net mean loss = ', np.mean(loss))
        print('std = ', np.std(loss))
        print('mi-net mean accuracy = ', np.mean(acc))
        print('std = ', np.std(acc))
        print('mi-net mean precision = ', np.mean(precision))
        print('std = ', np.std(precision))
        print('mi-net mean recall = ', np.mean(recall))
        print('std = ', np.std(recall))
        print('mi-net mean auc = ', np.mean(auc))
        print('std = ', np.std(auc))
