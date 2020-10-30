import torch
import os
import numpy as np
import backbone
import time
import random
from PIL import Image
from data.datamgr import TransformLoader

from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file, get_checkpoint_path
from model_resnet import *
from visualize_datapre import visualize_datapre

if __name__ == "__main__":
    np.random.seed(10)
    params = parse_args('mytest')
    image_size = params.image_size
    isAircraft = (params.dataset == 'aircrafts')
    if params.method in ['baseline', 'baseline++']:
        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                            jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                            loss_type = 'dist', jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        # n_query = max(1, int(params.n_query * params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                    jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params, use_bn=(not params.no_bn), pretrain=params.pretrain)
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True

            BasicBlock.maml = True
            Bottleneck.maml = True
            ResNet.maml = True

            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model.cuda()
    if params.test_dataset == params.transfered_dataset:
        params.checkpoint_dir = get_checkpoint_path(params)
        params.checkpoint_dir_test = params.checkpoint_dir
    else :
        params.checkpoint_dir, params.checkpoint_dir_test = get_checkpoint_path(params)
    checkpoint_dir_test = params.checkpoint_dir_test
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)


    split = 'novel'
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split


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

    assert modelfile, "can not find model weight file in {}".format(checkpoint_dir)
    print("use model weight file: ", modelfile)

    # iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    # acc_all = []


        print("{} model {} backbone have {} parameters.".format(model.__class__.__name__, params.model, total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{} model {} backbone have {} training parameters.".format(model.__class__.__name__, params.model, total_trainable_params))
    
        params = parse_args('mytest')
        loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')

        sub_meta, class_name = visualize_datapre(params, loadfile)
        random.seed(0)
        
        # from visualize_datapre.py,already tested
        # params
        test_n_way = params.test_n_way
        test_n_shot = params.test_n_shot
        test_n_query = params.test_n_query
        image_size = params.image_size


        classes_id = sub_meta.keys()
        selected_classes_id = random.sample(classes_id, test_n_way)

        # random selected images_path
        selected_imgs = {}
        for i in selected_classes_id:
            sub_imgs = sub_meta[i]
            sub_selected_imgs = random.sample(sub_imgs, test_n_shot + test_n_query)
            selected_imgs[i] = sub_selected_imgs

        trans_loader = TransformLoader(image_size)
        transform = trans_loader.get_composed_transform(aug=False)

        # got the source image and transformed image as a dict
        src_imgs = {}
        transformed_src_imgs = {}

        start = time.time()
        for i in selected_imgs.keys():
            src_imgs[i] = []
            transformed_src_imgs[i] = []
            for image_path in selected_imgs[i]:
                img = Image.open(image_path).convert('RGB')
                transformed_img = transform(img)
                src_imgs[i].append(img)
                transformed_src_imgs[i].append(transformed_img)
        print("load image data cost {:.4f}s".format(time.time() - start))
        data = []

        # stack in dim 0, in down order. 
        for key in transformed_src_imgs.keys():
            sub_data = transformed_src_imgs[key]
            sub_data = torch.stack(sub_data, dim=0)
            data.append(sub_data)
        test_data_episode = torch.stack(data)

        model.n_support = test_n_shot
        model.n_query = test_n_query
        scores = model.set_forward(test_data_episode)

        pred = scores.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( test_n_way ), test_n_query )
        acc = np.mean(pred == y)*100
        print(acc)


        

