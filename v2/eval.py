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

    ax2.imshow(target, vmin = 0, vmax = 18)
    ax2.set_title('Target')

    ax3.imshow(prediction, vmin = 0, vmax = 18)
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

def singleIOU(target, prediction):
    target_masked = np.where(target != 255, 1, 0)*target

    intersection = np.logical_and(target_masked, prediction)
    union = np.logical_or(target_masked, prediction)
    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculateIOU(dataset, model, device, shape, verbose=False):
    IOU = []
    #Dice = []
    size = len(dataset)
    for sample in range(size):

        image, target = dataset[sample][0].numpy(), dataset[sample][1]
        target = utils.map_id_to_train_id(target).numpy()
        prediction = predict(image, model, device, shape)
        IOU.append(mean_IOU(target,prediction))
        if verbose:
            print(f'[{sample}/{size}] mean IOU: {np.mean(IOU)}')

        #Dice.append(dice_score(target, prediction))

    averageIOU = np.mean(IOU)

    #mean_dice_per_class = np.nanmean(Dice,axis=0)
    return averageIOU#, mean_dice_per_class


def dice_score(target, prediction):
    # Initialize an array to store dice scores for each class
    dice_scores = np.zeros(19)
    
    for i in range(19):
        
        # Extract masks for class i
        pred_mask_i = np.where(prediction.int() == i, 1, 0)
        gt_mask_i = np.where(target.int() == i, 1, 0)
        
        if np.sum(gt_mask_i) == 0:
            dice_scores[i] = np.nan  # Class not present, assign NaN
            continue

        # Compute intersection, union, and dice score
        intersection = np.sum(pred_mask_i * gt_mask_i)
        union = np.sum(pred_mask_i) + np.sum(gt_mask_i)
        
        if union == 0:
            dice_scores[i] = 1.0  # Define dice score as 1.0 if union is 0
        else:
            dice_scores[i] = 2.0 * intersection / union
    
    return dice_scores

def mean_IOU(target, prediction, num_classes=19):
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
    
    mean_iou = np.nanmean(iou_scores)
    return mean_iou