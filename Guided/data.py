from torchvision.datasets import Cityscapes
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, v2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def getTransforms(subsize):
    transformsobject = Compose([
        #v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.PILToTensor(),
        v2.Resize(subsize, interpolation= InterpolationMode.NEAREST_EXACT, antialias=True),
    ])
    return transformsobject


def loadData(root,transforms,val_size=0.2,verbose=False):
    
    # Dataset from local root folder
    dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transform=transforms, target_transform=transforms)
    
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