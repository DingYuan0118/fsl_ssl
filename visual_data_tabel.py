# 暂时弃用

from visualize_datapre import visualize_datapre
from io_utils import parse_args
import os, random
from my_utils import visualize_support_imgs

if __name__ == "__main__":

    params = parse_args('mytest')
    loadfile = os.path.join('filelists', params.test_dataset, 'novel.json')
    sub_meta, class_names = visualize_datapre(params, loadfile)
    random.seed(0)

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

    visualize_support_imgs(selected_imgs, class_names, test_n_shot, image_size)


         