import os
import cv2


data_path = 'filelists/recognition36/images'
boxes_path = "filelists/recognition36/boxlabels_txt"

data_list = os.listdir(data_path)
bound_box = os.listdir(boxes_path)

des_path = 'filelists/recognition36/image_crops'
if not os.path.isdir(des_path):
    os.makedirs(des_path)

for img_path, box_path in zip(data_list, bound_box):
    img = cv2.imread(os.path.join(data_path, img_path))
    labels = []
    with open(os.path.join(boxes_path, box_path), "r") as f:
        for line in f.readlines():
            if line == "":
                continue
            labels.append(line)
    for label in labels:
        x1, y1, x2, y2 = map(int, label.split(",")[:4])
        # TODO : expand x1,y1,x2,y2 to a normal size
        img_crop = img[y1:y2, x1:x2, :]  # attention! change the XY coordinate
        cv2.imwrite(os.path.join(des_path, img_path), img_crop)




