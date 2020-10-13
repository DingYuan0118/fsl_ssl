import os
import numpy as np

data_path = "./images"
filenames = os.listdir(data_path)
labels = []
for file in filenames:
    label = file[:9]
    labels.append(label)

print(len(labels))
labels_unique = np.unique(labels)
print(len(labels_unique))

for label in labels_unique[1:]:
    if not os.path.isdir(os.path.join(data_path, label)):
        os.makedirs(os.path.join(data_path, label))

for file in filenames:
    if not os.path.isfile(os.path.join(data_path, file)):
        continue
    img_label = file[:9]
    os.rename(os.path.join(data_path, file), os.path.join(data_path, img_label, file))


    