import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file, get_checkpoint_path
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import json
from model_resnet import *
from my_utils import load_weight_file_for_test, select_model, feature_evaluation
import data.feature_loader as feat_loader


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('mytest')

    model = select_model(params)

    params.checkpoint_dir, params.checkpoint_dir_test = get_checkpoint_path(params)
    checkpoint_dir_test = params.checkpoint_dir_test

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    json.dump(vars(params), open(params.checkpoint_dir+'/configs.json','w'))


    ##### from save_features.py (except maml)#####
    # three situation for recognition36 dataset ["novel", "novel_car", "novel_plane"]
    split = 'novel'
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split


    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    acc_all = []

    model = load_weight_file_for_test(model, params)
    total_params = sum(p.numel() for p in model.parameters())
    print("{} model {} backbone have {} parameters.".format(model.__class__.__name__, params.model, total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} model {} backbone have {} training parameters.".format(model.__class__.__name__, params.model, total_trainable_params))

    if params.method in ['maml', 'maml_approx']:
        datamgr          = SetDataManager(params.image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params, isAircraft=(params.dataset == 'aircrafts'))
        loadfile         = os.path.join('filelists', params.test_dataset, 'novel.json')
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)

        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

    else:  ## eg: for Protonet
        ### from test.py ###
        novel_file = os.path.join( checkpoint_dir_test.replace("checkpoints","features"), split_str +"_shuffle_True.hdf5") #defaut split = novel, but you can also test base or val classes
        print('load novel file from:',novel_file)
        
        cl_data_file = feat_loader.init_loader(novel_file)
        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = params.test_n_query, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        
        with open(os.path.join( checkpoint_dir_test.replace("checkpoints","features"), split_str +"_test.txt") , 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.test_dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.test_dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )


