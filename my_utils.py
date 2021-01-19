from methods.baselinefinetune import BaselineFinetune
import numpy as np
from collections import defaultdict
import os
import h5py
import random
import json
from pathlib import Path

import backbone
import torch
from torch.autograd import Variable
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
# from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, get_resume_file, get_best_file, get_assigned_file
from model_resnet import *


def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    # add some printing to understand the program
    print("Message about the feature file:\n")
    f.visititems(print)

    f.close()


def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys() # cl_data_file is a dict: {class_id: sample_paths}

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) ) # z_all changed from list size(n_way, n_support+n_query, feature_dim)

    model.n_way = n_way  # predefined in 
    model.n_support = n_support
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    class_acc = {}
    for i, cl in enumerate(select_class):
        class_acc[cl] = np.mean((pred == y)[i * n_query : (i+1) * n_query])*100
    
    acc = np.mean(pred == y)*100
    return acc, class_acc


def select_model(params):
    """
    select which model to use based on params
    """
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    if params.method in ['baseline', 'baseline++'] :
        if params.dataset == 'CUB':
            params.num_classes = 200
        elif params.dataset == 'cars':
            params.num_classes = 196
        elif params.dataset == 'aircrafts':
            params.num_classes = 100
        elif params.dataset == 'dogs':
            params.num_classes = 120
        elif params.dataset == 'flowers':
            params.num_classes = 102
        elif params.dataset == 'miniImagenet':
            params.num_classes = 100
        elif params.dataset == 'tieredImagenet':
            params.num_classes = 608

        if params.script not in ['test', 'mytest']:
            if params.method == 'baseline':
                model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                                jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)
            elif params.method == 'baseline++':
                model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                                loss_type = 'dist', jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)
        else :
            if params.method == 'baseline':
                model           = BaselineFinetune( model_dict[params.model], **few_shot_params, tracking=params.tracking)
            elif params.method == 'baseline++':
                model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params, tracking=params.tracking)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 
        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params, use_bn=(not params.no_bn), pretrain=params.pretrain, tracking=params.tracking)
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
    return model


def select_dataloader_for_train(params):
    """
    select dataloader to define the Data reading mode based on params
    """
    isAircraft = (params.dataset == 'aircrafts')    
        
    base_file = os.path.join('filelists', params.dataset, params.base+'.json')
    val_file   = os.path.join('filelists', params.dataset, 'val.json')
     
    image_size = params.image_size

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(params.n_query * params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params, isAircraft=isAircraft)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params, isAircraft=isAircraft)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
    
    return base_loader, val_loader


def load_presaved_model_for_train(model, params):
    """
    
    """
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
            del tmp
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = 'checkpoints/%s/%s_%s' %(params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')
    
    if params.loadfile != '':
        print('Loading model from: ' + params.loadfile)
        checkpoint = torch.load(params.loadfile)
        ## remove last layer for baseline
        pretrained_dict = {k: v for k, v in checkpoint['state'].items() if 'classifier' not in k and 'loss_fn' not in k}
        print('Load model from:',params.loadfile)
        model.load_state_dict(pretrained_dict, strict=False)
    return model, start_epoch, stop_epoch


def load_weight_file_for_test(model, params):
    """
    choose the weight file for test process
    """
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

    else:  ## eg: for Protonet and others
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        ## for protonets

        model.feature.load_state_dict(state)
        model.eval()
        model = model.cuda()
        model.eval()
    return model


def read_json_file(loadfile):
    """
    data prepare process for visualize
    """
    with open(loadfile, 'r') as f:
        meta = json.load(f)

    # Implicit sort the list from small to large 
    cl_list = np.unique(meta['image_labels']).tolist()

    sub_meta = {}
    for cl in cl_list:
        # init dict
        sub_meta[cl] = []

    for x,y in zip(meta['image_names'],meta['image_labels']):
        # create subdataset for each class
        sub_meta[y].append(x)
    return sub_meta, meta


def manual_compute_scores(test_data_episode, model, test_n_way, test_n_shot, test_n_query):
    """
    params:
        test_data_episode (tensor):  shape(test_n_way, test_n_shot+test_n_query, channels, img_size, imgsize) , eg:(5, 21, 3, 224, 224)
        model :  used to computer classfier scores
        test_n_way, test_n_shot, test_n_query: few-shot params
    """
    model.n_support = test_n_shot
    model.n_query = test_n_query
    scores = model.set_forward(test_data_episode)

    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( test_n_way ), test_n_query )
    acc = np.mean(pred == y)*100
    print("acc:{:.4f}".format(acc))
    return acc


def visualize_support_imgs(selected_imgs, class_names, test_n_shot, image_size):
    """
    use html to draw pictures in a tabel, according to the CIFAR10 site
    params:
        selected_imgs : a dict format like {class_id_1: [img_path_1, img_path_2···], ···}
    """
    with open("dataset_table.html", "w") as fo:
        fo.write("<html>\n")
        fo.write("<body>\n\n")
        fo.write("<table>\n")
        for key in selected_imgs.keys():
            fo.write("\t<tr>\n")
            fo.write("\t\t<td><font size='5'>{}</font></td>\n".format(class_names[key]))
            for img_path in selected_imgs[key][:test_n_shot]:
                fo.write("\t\t<td><img src={0} height={1} width={1} /></td>\n".format(img_path, image_size))
            fo.write("\t</tr>\n")
        fo.write("</table>\n")
        fo.write("</body>\n")
        fo.write("</html>\n")
    print("\033[1;32;m{}\033[0m generated.".format("dataset_table.html"))


def produce_subjson_file(selected_classes_id, sub_meta, meta, params):
    """
    produce the sub dataset json file
    
    params:
        selected_classes_id (list): the id of selected classes. 
        sub_meta (dict): a dict format like {class_id_1: [img_path_1, img_path_2···], ···}
        meta (dict): the dataset dict from a json file
    """

    sub_json = defaultdict(list)
    sub_json["label_names"] = meta['label_names']
    
    for id in selected_classes_id:
        sub_json["image_names"].extend(sub_meta[id])
        sub_json["image_labels"].extend(np.repeat(id, len(sub_meta[id])).tolist())
    
    sub_json_name = "select_{}_sub.json".format("_".join([str(i) for i in selected_classes_id]))
    sub_json_path = Path(os.path.join("filelists/{}".format(params.test_dataset), sub_json_name)).as_posix()
    # save sub classes json file

    # produce some message
    count = 0
    for id in selected_classes_id:
        count = count + len(sub_meta[id])
        # highlight fir digit
        print("class \033[1;32;m{}\033[0m has \033[1;33;m{}\033[0m samples".format(meta['label_names'][id], len(sub_meta[id])))
    print("selected dataset has total \033[1;33;m{}\033[0m samples".format(count))

    # write teh sub json file
    with open(sub_json_path, "w") as fo:
        json_str = json.dumps(sub_json)
        fo.write(json_str)

    return sub_json_name, sub_json_path


def print_class_acc(class_acc, class_names):
    """
    params: 
        class_acc: a dict format like {class_id : acc}
        class_names: a List cointain the class_name by the order in dataset
    """
    for key in class_acc.keys():
        print("class \033[1;32;m{}\033[0m accuracy is \033[1;33;m{:.5f}\033[0m".format(class_names[key], class_acc[key]))
    print("total acc: \033[1;33;m{:.5f}\033[0m".format(np.mean(list(class_acc.values()))))


def print_model_params(model, params):
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(q.numel() for q in model.buffers())
    print("{} model {} backbone have {} parameters.".format(model.__class__.__name__, params.model, total_params + total_buffers))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} model {} backbone have {} training parameters.".format(model.__class__.__name__, params.model, total_trainable_params))