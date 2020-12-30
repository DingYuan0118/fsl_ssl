import numpy as np
import os
import glob
import argparse
import backbone
from model_resnet import *
import json

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            resnet18 = 'resnet18',
            resnet18_pytorch = 'resnet18_pytorch',
            resnet50_pytorch = 'resnet50_pytorch'
            ) 

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--script'      , default=script,           help="used for choose checkpoint ")
    parser.add_argument('--dataset'     , default='CUB',            help='CUB/cars/flowers/dogs/aircrafts/miniImagenet/tieredImagenet')
    parser.add_argument('--model'       , default='resnet18',       help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='protonet',       help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,      help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,      help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,      help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',      help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--jigsaw'      , action='store_true',      help='multi-task training')
    parser.add_argument('--lbda'        , default=0.5, type=float,  help='lambda for the jigsaw loss, (1-lambda) for proto loss')
    parser.add_argument('--lr'          , default=0.001, type=float,help='learning rate')
    parser.add_argument('--optimization', default='Adam', type=str, help='Adam or SGD')
    parser.add_argument('--loadfile'    , default='', type=str,     help='load pre-trained model')
    parser.add_argument('--finetune'    , action='store_true',      help='finetuning from jigsaw to protonet')
    parser.add_argument('--random'      , action='store_true',      help='random init net')
    parser.add_argument('--n_query'     , default=16, type=int,     help='number of query, 16 is used in the paper')
    parser.add_argument('--image_size'  , default=224, type=int,    help='224 is used in the paper')
    parser.add_argument('--date'        , default='', type=str,     help='date of the exp')
    parser.add_argument('--rotation'    , action='store_true',      help='multi-task training')
    parser.add_argument('--tracking'    , action='store_true',      help='tracking batchnorm stats')
    parser.add_argument('--split'        , default='novel',         help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
    parser.add_argument('--save_iter'    , default=-1, type=int,    help='saved feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--adaptation'   , action='store_true',     help='further adaptation in test time or not')

    parser.add_argument('--bs'          , default=16, type=int,     help='batch size used for unlabeled dataset, also when method==baseline')
    parser.add_argument('--no_bn'        , action='store_true',     help='not using batch norm if True')
    parser.add_argument('--pretrain'     , action='store_true',     help='use imagenet pre-train model')
    parser.add_argument('--grey'     , action='store_true',         help='use grey iamge')
    parser.add_argument('--test_bs'      , default=64, type=int,    help='batch size for testing w/o batchnorm')
    parser.add_argument('--dataset_unlabel', default=None,          help='CUB/cars/flowers/dogs/aircrafts/miniImagenet/tieredImagenet')
    
    parser.add_argument('--base'     , default='base',              help='name of the json file of the base set')
    parser.add_argument('--base_unlabel'     , default='base',      help='name of the json file of the base set for unlabeled dataset dataloader')


    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int,help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=100, type=int,help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,  help='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=600, type=int,help='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument('--resume'      , action='store_true',  help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true',  help='continue from baseline, neglected if resume is true') #never used in the paper
    
    if script == "mytest":
        parser.add_argument("--test_dataset", default="recognition36", help="for custom few shot dataset test for transfer situation")
        parser.add_argument("--transfered_dataset", default="cars", help="the base training dataset for novel test dataset")
        parser.add_argument("--test_n_query", default=16, type=int, help="the number of query for test")
        parser.add_argument("--test_n_shot", default=5, type=int, help="the number of support for test")

    parser.add_argument('--layer', default=-1, type=int)
        

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def get_checkpoint_path(params):
    # for test process
    if params.script == "mytest":
        checkpoint_dir = 'checkpoints/%s/%s_%s_%s' %(params.transfered_dataset, params.date, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        if not params.method in ['baseline', 'baseline++'] :
            checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)

        ## Use another dataset (dataloader) for unlabeled data
        if params.dataset_unlabel is not None:
            checkpoint_dir += params.dataset_unlabel
            checkpoint_dir += str(params.bs)

        ## Track bn stats
        if params.tracking:
            checkpoint_dir += '_tracking'

        if params.pretrain:
            checkpoint_dir += "_pretrain"
        
        if params.no_bn:
            checkpoint_dir += '_no_bn'

        ## Add jigsaw
        if params.jigsaw:
            checkpoint_dir += '_jigsaw_lbda%.2f'%(params.lbda)
            checkpoint_dir += params.optimization
        ## Add rotation
        if params.rotation:
            checkpoint_dir += '_rotation_lbda%.2f'%(params.lbda)
            checkpoint_dir += params.optimization

        checkpoint_dir += '_lr%.4f'%(params.lr)
        if params.finetune:
            checkpoint_dir += '_finetune'

        print('checkpoint path:',checkpoint_dir)
        if params.test_dataset == params.transfered_dataset:
            check_dir_test = checkpoint_dir
            return checkpoint_dir, check_dir_test
        else:
            check_dir_test = checkpoint_dir.replace(params.transfered_dataset, params.test_dataset)
            check_dir_test += "_{}".format(params.transfered_dataset)
            print('test checkpoint path:', check_dir_test)
            return checkpoint_dir, check_dir_test
    
    elif params.script in ["train", "test"]:
        checkpoint_dir = 'checkpoints/%s/%s_%s_%s' %(params.dataset, params.date, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        if not params.method in ['baseline', 'baseline++'] :
            checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)

        ## Use another dataset (dataloader) for unlabeled data
        if params.dataset_unlabel is not None:
            checkpoint_dir += params.dataset_unlabel
            checkpoint_dir += str(params.bs)

        ## Track bn stats
        if params.tracking:
            checkpoint_dir += '_tracking'

        if params.pretrain:
            checkpoint_dir += "_pretrain"
            
        if params.no_bn:
            checkpoint_dir += '_no_bn'

        ## Add jigsaw
        if params.jigsaw:
            checkpoint_dir += '_jigsaw_lbda%.2f'%(params.lbda)
            checkpoint_dir += params.optimization
        ## Add rotation
        if params.rotation:
            checkpoint_dir += '_rotation_lbda%.2f'%(params.lbda)
            checkpoint_dir += params.optimization

        checkpoint_dir += '_lr%.4f'%(params.lr)
        if params.finetune:
            checkpoint_dir += '_finetune'

        print('checkpoint path:',checkpoint_dir)
        return checkpoint_dir


def combine_dataset_from_different_domain(source1, source2):
    """
    combine different dataset.

    params:
        source1: dataset1 json path
        source2: dataset2 json path
    """
    with open(source1, "r") as f:
        dataset1 = json.load(f)
    with open(source2, "r") as f:
        dataset2 = json.load(f)
    
    dataset = {}
    keys = dataset1.keys()
    for key in keys:
        # init dict for deep copy
        # ! do not use dataset1 to init dataset, this will cause a reference problem
        dataset[key] = []
    for key in keys:
        dataset[key].extend(dataset1[key])

    # combine dataset 2
    for key in keys:
        if key != "image_labels":
            dataset[key].extend(dataset2[key])  
        else:
            dataset[key].extend((np.array(dataset2[key]) + len(dataset1["label_names"])).tolist())
    return dataset


def combine_dataset_from_same_domain(source1, source2):
    """
    combine different dataset from the same domain, such as base.json and val.json.

    ! maybe not useful
    params:
        source1: dataset1 json path
        source2: dataset2 json path
    """
    pass


# test functions
if __name__ == "__main__":
    source1 = "filelists\\cars\\base.json"
    source2 = "filelists\\cars\\base.json"
    dataset = combine_dataset_from_different_domain(source1, source2)
    with open("combine.json", "w") as f:
        json.dump(dataset, f)
    print(dataset.keys())