from torchvision.datasets import Cityscapes
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, v2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from GTA5.dataset import GTA_5

class DivideBy255:
    def __call__(self, tensor):
        return tensor / 255.0

def getTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
        DivideBy255(),
    ])
    return trans

def getTargetTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
    ])
    return trans

def getFlipTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(1),
        DivideBy255(),
    ])
    return trans

def getFlipTargetTransforms(subsize):
    trans = Compose([
        transforms.PILToTensor(),
        transforms.Resize(subsize, interpolation= InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(1),
    ])
    return trans

def getGTAtransform(subsize):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(subsize, interpolation= transforms.InterpolationMode.NEAREST),
    ])
    return trans


def loadData(root,subsize,val_size=0.2, flip= False, verbose=False):
    
    # Dataset from local root folder
    dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transform=getTransforms(subsize), target_transform=getTargetTransforms(subsize))

    # Train/val split
    generator1 = torch.Generator().manual_seed(2147483647)
    train_set, val_set = random_split(dataset, [1-val_size,val_size], generator=generator1)
    
    if flip:
        flipped_dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transform=getFlipTransforms(subsize), target_transform=getFlipTargetTransforms(subsize))
        train_flip, _ = random_split(flipped_dataset, [1-val_size,val_size], generator=generator1)
        train_set = torch.utils.data.ConcatDataset([train_set, train_flip])

    # Print sizes
    if verbose:
        print(f'Size of the training set: {len(train_set)}')
        print(f'Size of the validation set: {len(val_set)}')


    return train_set, val_set

def loadGTAData(root,subsize,val_size=0.01,verbose=False):
    dataset = GTA_5(root,getGTAtransform(subsize))

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
