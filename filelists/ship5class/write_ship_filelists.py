import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path
import copy



cwd = os.getcwd() 
train_data_path = 'filelists/ship5class/5class'
test_data_path = 'filelists/ship5class/5class_test'
savedir = 'filelists/ship5class/'
dataset_list = ["train", "test", "novel"]
train_class_list = listdir(train_data_path)
test_class_list = listdir(test_data_path)
assert train_class_list == test_class_list, "train test classes don't match"

class_names = []
for class_name in train_class_list:
    class_names.append(class_name)
class_names_unique = np.unique(class_names).tolist()

train_classfile_list = []
test_classfile_list = []
total_classfile_list = []

for class_name in class_names_unique:
    train_image_name = []
    test_image_name = []
    train_classfile_path = Path(join(train_data_path, class_name)).as_posix()
    test_classfile_path = Path(join(test_data_path, class_name)).as_posix()
    for train_file_name in listdir(train_classfile_path):
        if train_file_name[-4:] == ".jpg":
            train_image_name.append(Path(join(train_classfile_path, train_file_name)).as_posix())
    for test_file_name in listdir(test_classfile_path):
        if test_file_name[-4:] == ".jpg":
            test_image_name.append(Path(join(test_classfile_path, test_file_name)).as_posix())
    
    train_classfile_list.append(train_image_name)
    test_classfile_list.append(test_image_name)
for i, j in copy.deepcopy(zip(train_classfile_list, test_classfile_list)):
    i.extend(j)
    total_classfile_list.append(i)

print(len(train_classfile_list))
print(len(test_classfile_list))
print(len(total_classfile_list))



for dataset in dataset_list:
    file_list = []
    label_list = []
    if dataset == "train":
        classfile_list_all = train_classfile_list
    elif dataset == "test":
        classfile_list_all = test_classfile_list
    else:
        classfile_list_all = total_classfile_list

    for i, classfile_list in enumerate(classfile_list_all):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in class_names_unique])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
