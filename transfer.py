from __future__ import print_function, division
from data.dataset import CustomDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import argparse

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))  # (C,H,W) to (H,W,C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig("visulize_data.jpg")
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, use_sche=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print_freq = 5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if ((i+1) % print_freq==0 or (i+1) == len(dataloaders[phase])) and phase == "train":
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(dataloaders[phase]), loss.item()))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                if use_sche:
                    scheduler.step() # remove scheduler
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('training acc', epoch_acc, epoch)

            if phase == 'val':
                writer.add_scalar('val acc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        epoch_end = time.time()
        print('A epoch cost in {:.0f}m {:.0f}s'.format(
            (epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60))
        print()
    # torch.save(best_model_wts, "resnet_18_classes_10.pth")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def model_finetune(pretrain=True, num_epochs=25, num_classes=5, use_sche=False):
    model_ft = models.resnet18(pretrained=pretrain)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[80, 160], gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=num_epochs, use_sche=use_sche)
    return model_ft


def model_as_extractor(pretrain=True, num_epochs=25, num_classes=5, use_sche=False):
    model_conv = torchvision.models.resnet18(pretrained=pretrain)
    if pretrain == True:
        for param in model_conv.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=[160, 240], gamma=0.1)
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=num_epochs, use_sche=use_sche)
    return model_conv

def test_model(model, dataloader):
    running_corrects = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes["test"]
    return test_acc

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', default="ship5class", help='choose dataset')
    parse.add_argument('-f', '--flag', default="finetune", choices=['finetune', 'extractor'], help="Use finetune or extractor")
    parse.add_argument('--pretrain', action="store_true", help="Use pretrain or not")
    parse.add_argument('--num_train', default=5 ,type=int, help="Order the number of shots")
    parse.add_argument('--num_epochs', default=25, type=int, help="num of epochs")
    parse.add_argument('--eval', action='store_true', help='set the eval mode')
    parse.add_argument('--use_sche', default=False, action='store_true', help="use scheduler or not")
    parse.add_argument('--batch_size', default=4, type=int, help="batch size used for train")
    parse.add_argument('--num_workers', default=8, type=int, help="control the number of num_workers, For linux default is 8, for windows default is 0")

    arg = parse.parse_args()
    num_train = arg.num_train
    num_epochs = arg.num_epochs
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_file = os.path.join('filelists', arg.dataset, 'novel.json')
    print("use data file:{}".format(data_file))

    train_dataset = CustomDataset(data_file=data_file, mode="train", num_train=num_train,\
                                num_val=10, transform=data_transforms["train"])

    val_dataset = CustomDataset(data_file=data_file, mode="val", num_train=num_train,\
                                num_val=10, transform=data_transforms["val"])

    test_dataset = CustomDataset(data_file=data_file, mode="test", num_train=num_train,\
                                num_val=10, transform=data_transforms["val"])

    dataloaders = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=arg.num_workers),\
                    "val" : torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                                  shuffle=True, num_workers=arg.num_workers),\
                    "test" : torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                                  shuffle=True, num_workers=arg.num_workers)}
    
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)}
    print("datasize: train={train} samples, val={val} samples, test={test} samples".format(**dataset_sizes))
    class_names = train_dataset.label_names
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # inputs, labels = next(iter(dataloaders['train']))
    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in labels])

    # use --flag to set pretrain type ('finetune' or 'extractor')
    # use --eval to set eval mode
    # use --pretrain to set pretrain mode or not
    # use --epoch to set number of epoch
    # use --shot ti set number of shots
    checkpoint_path = "checkpoints/{0}/_resnet18_transfer_{1}_{3}_epochs_{2}_shots".format(arg.dataset, arg.flag, arg.num_train, arg.num_epochs)
    if arg.use_sche:
        checkpoint_path += "_use_sche"
    writer = SummaryWriter(log_dir=checkpoint_path)
    if arg.flag == "finetune":
        model_ft = model_finetune(pretrain=arg.pretrain, num_epochs=num_epochs, num_classes=num_classes, use_sche=arg.use_sche)
        torch.save(model_ft.state_dict, os.path.join(checkpoint_path, "batch_size{}_best.tar".format(arg.batch_size)))
        test_acc = test_model(model_ft, dataloaders["test"])

    elif arg.flag == "extractor":
        model_conv = model_as_extractor(pretrain=arg.pretrain, num_epochs=num_epochs, num_classes=num_classes, use_sche=arg.use_sche)
        torch.save(model_conv.state_dict, os.path.join(checkpoint_path, "batch_size{}_best.tar".format(arg.batch_size)))
        test_acc = test_model(model_conv, dataloaders["test"])
    
    print("test acc:{:.2f}".format(test_acc))
    if not os.path.exists(checkpoint_path.replace("checkpoints","features")):
        os.mkdir(checkpoint_path.replace("checkpoints","features"))
    with open(os.path.join( checkpoint_path.replace("checkpoints","features"), "result.txt") , 'a') as f:
        exp_str = "{epoch} epochs, {batchsize} batch size: test accuracy={acc:4.2f}%\n".format(epoch=num_epochs,\
                                                                    acc=test_acc * 100, batchsize=arg.batch_size)
        f.write(exp_str)
