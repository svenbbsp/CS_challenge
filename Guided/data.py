from torchvision.datasets import Cityscapes
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, v2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def getTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
    ])
    return trans

def getFlipTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(1)
    ])
    return trans


def loadData(root,subsize,val_size=0.2,verbose=False):
    
    # Dataset from local root folder
    dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transform=getTransforms(subsize), target_transform=getTransforms(subsize))
    flipped_dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transform=getFlipTransforms(subsize), target_transform=getFlipTransforms(subsize))
    
    dataset = torch.utils.data.ConcatDataset([dataset, flipped_dataset])

    # Train/val split
    train_set, val_set = random_split(dataset, [1-val_size,val_size])

    # Print sizes
    if verbose:
        print(f'Size of the training set: {len(train_set)}')
        print(f'Size of the validation set: {len(val_set)}')


    return train_set, val_set


def getDataLoader(train_set, val_set, batch_size, num_workers=4):
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dl, val_dl