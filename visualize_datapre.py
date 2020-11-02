# 暂时弃用

import os
import numpy as np
import json
from io_utils import parse_args
import random 
from PIL import Image
import torch

import time

from data.datamgr import TransformLoader


def visualize_datapre(loadfile):
    """
    data prepare process for visualize
    """
    with open(loadfile, 'r') as f:
        meta = json.load(f)

    cl_list = np.unique(meta['image_labels']).tolist()

    sub_meta = {}
    for cl in cl_list:
        # init dict
        sub_meta[cl] = []

    for x,y in zip(meta['image_names'],meta['image_labels']):
        # create subdataset for each class
        sub_meta[y].append(x)
    return sub_meta, meta['label_names']



if __name__ == "__main__":
    # test function
    params = parse_args('mytest')
    loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')

    sub_meta, class_names = visualize_datapre(params, loadfile)
    random.seed(0)

    # params
    n_way = params.test_n_way
    n_shot = params.n_shot
    n_query = params.n_query
    image_size = params.image_size


    classes_id = sub_meta.keys()
    selected_classes_id = random.sample(classes_id, n_way)

    # random selected images_path
    selected_imgs = {}
    for i in selected_classes_id:
        sub_imgs = sub_meta[i]
        sub_selected_imgs = random.sample(sub_imgs, n_shot + n_query)
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
    
    