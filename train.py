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
from my_utils import select_dataloader_for_train, select_model, load_presaved_model_for_train, feature_evaluation, save_features, print_model_params
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import json
from model_resnet import *

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):    
    if params.optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    elif params.optimization == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, nesterov=True, momentum=0.9, weight_decay=params.wd)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       
    writer = SummaryWriter(log_dir=params.checkpoint_dir)
    for epoch in range(start_epoch, stop_epoch):
        start_epoch_time = time.time()

        model.train()
        model.train_loop(epoch, base_loader, optimizer, writer) #model are called by reference, no need to return 
        print("a epoch traininig process cost {}s".format(time.time() - start_epoch_time))
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        start_test_time = time.time()
        # TODO: can change the val frequency to reduce training time
        if ((epoch+1) % 10==0) or (epoch==stop_epoch-1):
            if params.jigsaw:
                acc, acc_jigsaw = model.test_loop( val_loader)
                writer.add_scalar('val/acc', acc, epoch)
                writer.add_scalar('val/acc_jigsaw', acc_jigsaw, epoch)
            elif params.rotation:
                acc, acc_rotation = model.test_loop( val_loader)
                writer.add_scalar('val/acc', acc, epoch)
                writer.add_scalar('val/acc_rotation', acc_rotation, epoch)
            else:    
                acc = model.test_loop( val_loader)
                writer.add_scalar('val/acc', acc, epoch)
            print("val acc:", acc)
            print("a epoch test process cost{}s".format(time.time() - start_test_time))
            if acc > max_acc :  #for baseline and baseline++, we don't use validation here so we let acc = -1
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

            if ((epoch+1) % params.save_freq==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            print("a epoch cost {} s".format(time.time() - start_epoch_time))

    # return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    isAircraft = (params.test_dataset == 'aircrafts')

    model = select_model(params)
    base_loader, val_loader = select_dataloader_for_train(params)

    model = model.cuda()
    print_model_params(model, params)

    params.checkpoint_dir = get_checkpoint_path(params)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    model, start_epoch, stop_epoch = load_presaved_model_for_train(model, params)

    json.dump(vars(params), open(params.checkpoint_dir+'/configs.json','w'))
    train(base_loader, val_loader,  model, start_epoch, stop_epoch, params) # can comment this line for test 


    ##### from save_features.py (except maml)#####
    split = 'novel'
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    image_size = params.image_size
    acc_all = []

    if params.loadfile != '':
        modelfile   = params.loadfile
        checkpoint_dir = params.loadfile
    else:
        checkpoint_dir = params.checkpoint_dir # checkpoint path
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir) # return the best.tar file

    if params.method in ['maml', 'maml_approx']:
        if modelfile is not None:
            tmp = torch.load(modelfile)
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(tmp['state'])
        print('modelfile:',modelfile)

        datamgr          = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params, isAircraft=isAircraft)
        loadfile         = os.path.join('filelists', params.dataset, 'novel.json')
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)
    else:  # eg: for Protonet
        if params.save_iter != -1:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), "novel_" + str(params.save_iter)+ ".hdf5")
        else:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), "novel.hdf5") # path for use the best model to produce feature

        datamgr          = SimpleDataManager(image_size, batch_size = params.test_bs, isAircraft=isAircraft)
        loadfile         = os.path.join('filelists', params.dataset, 'novel.json')
        data_loader      = datamgr.get_data_loader(loadfile, aug = False)

        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        model.feature.load_state_dict(state)
        model.eval()
        model = model.cuda()
        model.eval()

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print('save outfile at:', outfile)

        save_features(model, data_loader, outfile)

        ### from test.py ###

        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        print('load novel file from:',novel_file)
        import data.feature_loader as feat_loader
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc, _ = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        
        with open(os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +"_test.txt") , 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )


