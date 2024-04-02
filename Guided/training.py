from model import Model
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import utils
import torch
import wandb

def buildModel(lr_rate=3e-4, verbose=False):

    model = Model().cuda()
    loss_fn = CrossEntropyLoss(ignore_index=255)
    optimizer = Adam(model.parameters(), lr=lr_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f'Working on device: {device}')

    return model, loss_fn, optimizer, device

def trainModel(dataloader, model, loss_fn, optimizer,device, wenb=True, verbose=False):
    """
    Train a model for 1 epoch.

    Params:
    - dataloader:   dataset to train on.
    - model:        the model object to be trained.
    - loss_fn:      the loss function.
    - optimizer:    the desired optimization.
    """
    size = len(dataloader.dataset)
    model.train() #Set the model to train mode
    for batch, (image,target) in enumerate(dataloader):
        image = image.to(device)
        #print(image.size())
        target  = (target*255).long().squeeze(dim=0)     #*255 because the id are normalized between 0-1
        target = utils.map_id_to_train_id(target).to(device)
        
        #print(target.size())
        
        #predict
        segmentation = model(image)
        #Loss
        loss = loss_fn(segmentation, target)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            #print loss during training
            loss, current = loss.item(), (batch + 1) * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if wenb:
            wandb.log({"train_loss": loss})
    
