import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py
import json

import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file, get_checkpoint_path
from model_resnet import *
from my_utils import save_features, print_model_params, select_model, load_weight_file_for_test

if __name__ == '__main__':
    params = parse_args('mytest')

    isAircraft = (params.test_dataset == 'aircrafts')

    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    image_size = params.image_size

    # three situation for recognition36 dataset ["novel", "novel_car", "novel_plane"]
    split = 'novel'

    loadfile = os.path.join('filelists', params.test_dataset, split+".json")
    with open(loadfile, 'r') as f:
        meta = json.load(f)
    print("dataset length:", len(meta['image_names']))


    # if params.json_seed is not None:
    #     checkpoint_dir = '%s/checkpoints/%s_%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.json_seed, params.date, params.model, params.method)
    # else:
    params.checkpoint_dir, params.checkpoint_dir_test = get_checkpoint_path(params)
    checkpoint_dir_test = params.checkpoint_dir_test
    checkpoint_dir = params.checkpoint_dir

    if not os.path.isdir(checkpoint_dir_test):
        os.makedirs(checkpoint_dir_test)


    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir_test.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5")
    else:
        outfile = os.path.join( checkpoint_dir_test.replace("checkpoints","features"), split + "_shuffle_False_bn_16.hdf5")


    datamgr         = SimpleDataManager(image_size, batch_size = params.test_bs, isAircraft=isAircraft, shuffle=False)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)

    model = select_model(params)
    model = load_weight_file_for_test(model, params)
    model = model.cuda()
    
    model.feature.eval()
    model.eval()
    model.cuda()
    model.eval()
    print_model_params(model, params)
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    print('outfile is', outfile)
    save_features(model, data_loader, outfile)
