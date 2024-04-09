import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from numpy import random
import process_data
import PIL
import utils
from matplotlib.colors import LinearSegmentedColormap


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
    target_masked = np.where(target != 255, 1, 0)

    intersection = np.logical_and(target_masked, prediction)
    union = np.logical_or(target_masked, prediction)
    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculateIOU(dataset, model, device, shape, verbose=False):
    IOU = 0
    size = len(dataset)
    for sample in range(size):

        image, target = dataset[sample][0].numpy(), dataset[sample][1].numpy()
        value = singleIOU(target, predict(image, model, device, shape))
        
        IOU += value
        if verbose:
            print(f'[{sample}/{size}] mean IOU: {IOU/sample}')

    averageIOU = IOU/size
    return averageIOU


def dice_score(predicted_mask, ground_truth_mask):
    # Initialize an array to store dice scores for each class
    dice_scores = np.zeros(19)
    
    for i in range(19):
        
        # Extract masks for class i
        pred_mask_i = np.where(predicted_mask == i, 1, 0)
        gt_mask_i = np.where(ground_truth_mask == i, 1, 0)
        
        # Compute intersection, union, and dice score
        intersection = np.sum(pred_mask_i * gt_mask_i)
        union = np.sum(pred_mask_i) + np.sum(gt_mask_i)
        
        if union == 0:
            dice_scores[i] = 1.0  # Define dice score as 1.0 if union is 0
        else:
            dice_scores[i] = 2.0 * intersection / union
    
    return dice_scores