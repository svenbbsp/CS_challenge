from model import Model
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import utils
import torch
import wandb

def buildModel(lr_rate, weights, verbose=False):

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    loss_fn = CrossEntropyLoss(weight=weights.to(device),ignore_index=255)
    optimizer = Adam(model.parameters(), lr=lr_rate)

    if verbose:
        print(f'Working on device: {device}')

    return model, loss_fn, optimizer, device

def trainSingleEpoch(dataloader, model, loss_fn, optimizer,device, wenb=True, verbose=False):
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
        image = image.float().to(device)
        #print(image.size())
        target  = (target).long().squeeze(dim=1)    
        target = utils.map_id_to_train_id(target).to(device)
        
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
    

def testModel(dataloader, model, loss_fn, device, epoch, wenb=True, verbose=False):
    """
    Test a model.

    Params:
    - dataloader:   dataset to test on.
    - model:        the model object to be tested.
    - loss_fn:      the loss function.
    """
    num_batches = len(dataloader)
    model.eval() #model in eval mode
    test_loss = 0
    with torch.no_grad():
        for _, (image,target) in enumerate(dataloader):
            image = image.float().to(device)
            target  = target.long().squeeze(dim=1)
            target = utils.map_id_to_train_id(target).to(device)

            pred = model(image)
            loss = loss_fn(pred, target).item()
            test_loss += loss
            if verbose:
                print(f'Loss: {loss}')

            if wenb:
                wandb.log({"validation_loss": loss})
            
    test_loss /= num_batches
    if verbose:
        print(f"Average validation loss epoch {epoch}: \n Avg loss: {test_loss:>8f} \n")

    
    if wenb:
        wandb.log({"average_validation_loss": test_loss, "epoch": epoch})

def trainModel(train_dataloader, val_dataloader, model, loss_fn, optimizer,device, epochs, wenb, verbose):
    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        trainSingleEpoch(train_dataloader, model, loss_fn, optimizer, device, wenb, verbose)
        testModel(val_dataloader, model, loss_fn, device, (t+1), wenb, verbose)

    print("Done!")