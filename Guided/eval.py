import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from numpy import random
import process_data
import PIL


def showImageTargetAndPrediction(image, target, prediction):
    image = np.transpose(image, (1, 2, 0))
    target = np.squeeze(target)
    prediction = np.squeeze(prediction)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

    ax1.imshow(image)
    ax1.set_title('Image')

    ax2.imshow(target)
    ax2.set_title('Target')

    ax3.imshow(prediction)
    ax3.set_title('Prediction')
    
    plt.show()

def showPixelValues(image):
    image = image.flatten()
    plt.boxplot(image)
    plt.show()


def getRandomImageAndTarget(dataset):
    sample = random.randint(len(dataset))
    image, target = dataset[sample][0].numpy(), dataset[sample][1].numpy()

    return image, target

def predict(image,model,device,shape):
    image = torch.tensor(image)
    input = image.unsqueeze(0).float().to(device)
    model.eval()
    output = model(input)
    prediction = process_data.postprocess(output,shape)

    return prediction