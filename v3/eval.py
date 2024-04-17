import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from numpy import random
import process_data
import PIL
import utils
from torch.utils.data import DataLoader


def showImageTargetAndPrediction(image, target, prediction):
    image = np.transpose(image, (1, 2, 0))
    target = np.squeeze(target)
    target[target == 255] = 0
    prediction = np.squeeze(prediction)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))


    ax1.imshow(image)
    ax1.set_title('Image')

    ax2.imshow(target, vmin = 0, vmax = 18, cmap='Set3')
    ax2.set_title('Target')

    ax3.imshow(prediction, vmin = 0, vmax = 18, cmap='Set3')
    ax3.set_title('Prediction')
    
    plt.show()

def showPixelValues(image):
    image = image.flatten()
    plt.boxplot(image)
    plt.show()


def getRandomImageAndTarget(dataset):
    sample = random.randint(len(dataset))
    image, target = dataset[sample][0].numpy(), dataset[sample][1]
    target = utils.map_id_to_train_id(target).numpy()

    return image, target

def predict(image,model,device,shape):
    image = torch.tensor(image)
    input = image.unsqueeze(0).float().to(device)
    model.eval()
    output = model(input)
    prediction = process_data.postprocess(output,shape)

    return prediction


def IOU(dataloader, model, device, shape):
    model.eval()
    iou = []
    with torch.no_grad():
        for batch, (image, target) in enumerate(dataloader):
            input = image.float().to(device)
            output = model(input)
            
            
            prediction = process_data.postprocess(output, shape)
            target = utils.map_id_to_train_id(target)
            target = target.squeeze().cpu().numpy()
            iou.append(batch_IOU(target, prediction))
    
    #print(f"The average IOU is equal to: {np.mean(iou)}")
    return np.mean(iou)


def single_IOU(target, prediction,num_classes=19):
    iou_scores = []
    for class_id in range(num_classes):
        # Create masks for the specific class
        target_masked = np.where(target == class_id, 1, 0)
        prediction_masked = np.where(prediction == class_id, 1, 0)

        if np.sum(target_masked) == 0:
            iou_scores.append(np.nan)
            continue

        intersection = np.logical_and(target_masked, prediction_masked)
        union = np.logical_or(target_masked, prediction_masked)
            
        # Compute IoU for the current class
        class_iou = np.sum(intersection) / np.sum(union)
        
        iou_scores.append(class_iou)
        
    mean_iou = (np.nanmean(iou_scores))
    return mean_iou

def batch_IOU(target, prediction):
    iou = []
    for i in range(len(target)):

        iou.append(single_IOU(target[i],prediction[i]))
    
    return np.mean(iou)



def single_DICE(target, prediction,num_classes=19):
    dice_scores = []
    for class_id in range(num_classes):
        # Create masks for the specific class
        target_masked = np.where(target == class_id, 1, 0)
        prediction_masked = np.where(prediction == class_id, 1, 0)

        if np.sum(target_masked) == 0:
            dice_scores.append(np.nan)
            continue

        intersection = np.logical_and(target_masked, prediction_masked)
        area = np.sum(target_masked) + np.sum(prediction_masked)
            
        # Compute IoU for the current class
        class_dice = (2* np.sum(intersection)) / area
        dice_scores.append(class_dice)
        
    mean_iou = (np.nanmean(dice_scores))
    return mean_iou

def batch_DICE(target, prediction):
    dice = []
    for i in range(len(target)):
        dice.append(single_DICE(target[i],prediction[i]))
    
    return np.mean(dice)

def DICE(dataloader, model, device, shape):
    model.eval()
    dice = []
    with torch.no_grad():
        for batch, (image, target) in enumerate(dataloader):
            input = image.float().to(device)
            output = model(input)
            
            
            prediction = process_data.postprocess(output, shape)
            target = utils.map_id_to_train_id(target)
            target = target.squeeze().cpu().numpy()
            dice.append(batch_DICE(target, prediction))
    
    #print(f"The average IOU is equal to: {np.mean(iou)}")
    return np.mean(dice)