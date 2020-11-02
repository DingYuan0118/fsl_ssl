import torch
import os
import numpy as np
import backbone
import time
import random
from PIL import Image
from data.datamgr import TransformLoader
import json
from pathlib import Path
from collections import defaultdict

from data.datamgr import SimpleDataManager, SetDataManager
import data.feature_loader as feat_loader
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file, get_checkpoint_path
from model_resnet import *
from my_utils import select_model, load_weight_file_for_test, read_json_file, produce_subjson_file, save_features, feature_evaluation

if __name__ == "__main__":
    np.random.seed(10)
    params = parse_args('mytest')
    image_size = params.image_size
    isAircraft = (params.dataset == 'aircrafts')
    model = select_model(params)

    model.cuda()

    params.checkpoint_dir, params.checkpoint_dir_test = get_checkpoint_path(params)
    checkpoint_dir_test = params.checkpoint_dir_test
    checkpoint_dir = params.checkpoint_dir

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)


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
    print("{} model \033[1;33;m{}\033[0m backbone have \033[1;33;m{}\033[0m parameters.".format(model.__class__.__name__, params.model, total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} model \033[1;33;m{}\033[0m backbone have \033[1;33;m{}\033[0m training parameters.".format(model.__class__.__name__, params.model, total_trainable_params))

    if params.method in ['maml', 'maml_approx']:
        datamgr          = SetDataManager(params.image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params, isAircraft=(params.dataset == 'aircrafts'))
        loadfile         = os.path.join('filelists', params.test_dataset, 'novel.json')
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)

        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)
        params = parse_args('mytest')
        loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')

    else:
        loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')
        sub_meta, meta = read_json_file(loadfile)

        random.seed(0)
        # from visualize_datapre.py,already tested
        # params
        test_n_way = params.test_n_way
        test_n_shot = params.test_n_shot
        test_n_query = params.test_n_query
        image_size = params.image_size


        classes_id = sub_meta.keys()
        # for test function
        selected_classes_id = random.sample(classes_id, test_n_way)
        # selected_classes_id = random.sample(classes_id, len(classes_id))
        # selected_classes_id = classes_id

        
        # init 
        sub_json_name, sub_json_path = produce_subjson_file(selected_classes_id, sub_meta, meta, params)
        # set the shuffle False to recognize the support sample
        sub_datamgr  = SimpleDataManager(image_size, batch_size = params.test_bs, isAircraft=isAircraft, shuffle=False)
        sub_data_loader  = sub_datamgr.get_data_loader(sub_json_path, aug = False) 
        
        output_path = checkpoint_dir_test.replace("checkpoints", "features")
        output_name = sub_json_name.replace(".json", ".hdf5")
        output_file = Path(os.path.join(output_path, output_name)).as_posix()

        # pass if hdf5 file already exists
        if not os.path.exists(output_file):
            save_features(model, sub_data_loader, output_file)

        sub_novel_file = output_file
        # sub_novel_file = os.path.join( checkpoint_dir_test.replace("checkpoints","features"), split_str +"_shuffle_false.hdf5")
        print('load novel file from:',sub_novel_file)
        
        sub_cl_data_file = feat_loader.init_loader(sub_novel_file)
        acc = feature_evaluation(sub_cl_data_file, model, n_query = params.test_n_query, adaptation = params.adaptation, **few_shot_params)
        print(acc)


        
        




        



        

