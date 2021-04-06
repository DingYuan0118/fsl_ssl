# 观察RandomResizeCrop()对于原始高分辨率的小目标是否有影响

from torch import tensor
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from PIL import Image
import matplotlib.pyplot as plt

# aug transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
# transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

class TransformLoader:
    """
    get a composed tranform including ('RandomResizedCrop()', 'ImageJitter',
     'RandomHorizontalFlip', 'ToTensor', 'Normalize', 'Resize')
    """
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        # elif transform_type=='Scale':
        #     return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Resize':
            return method(int(self.image_size))
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            # transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor'] # 无归一化
        else:
            # transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            # transform_list = ['Resize','CenterCrop', 'ToTensor'] # 无归一化

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

tranloader = TransformLoader(224)

transforms_noaug = tranloader.get_composed_transform(aug=False)
transforms_aug = tranloader.get_composed_transform(aug=True)

# img = Image.open("filelists/recognition36/images/LG_car_ARM_tiger_pit_0_1.jpg") #recognition
img = Image.open("filelists/miniImagenet/images/n01532829/n0153282900000005.jpg") # imagenet

plt.figure("原图")
plt.imshow(img)


transforms_noaug_img = transforms_noaug(img)
plt.figure("无增强变换")
plt.imshow(transforms.ToPILImage()(transforms_noaug_img))

transforms_aug_img = transforms_aug(img)
plt.figure("有增强变换")
plt.imshow(transforms.ToPILImage()(transforms_aug_img))
plt.show()

print()