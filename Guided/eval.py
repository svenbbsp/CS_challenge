import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np

def showImage(data_loader):
    image, target = next(iter(data_loader))

    pil_image = transforms.ToPILImage()(image.squeeze().to('cpu'))
    plt.imshow(pil_image)
    plt.show()

def showImageAndTarget(data_loader):
    image, target = next(iter(data_loader))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    pil_image = transforms.ToPILImage()(image.squeeze().to('cpu'))
    pil_target = transforms.ToPILImage()(target.squeeze().to('cpu'))
    ax1.imshow(pil_image)
    ax1.set_title('Image')

    ax2.imshow(pil_target)
    ax2.set_title('Target')

    plt.show()

def showImageTargetAndPrediction(image, target, prediction):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

    ax1.imshow(image)
    ax1.set_title('Image')

    ax2.imshow(target)
    ax2.set_title('Target')

    ax3.imshow(prediction)
    ax3.set_title('Prediction')
    
    plt.show()

def showPixelValues(image):
    image = (np.array(image)).flatten()
    plt.boxplot(image)
    plt.show()



def getImageTargetAndPrediction(data_loader, model):
    batch = next(iter(data_loader))
    image, target = batch[0][0].unsqueeze(0), batch[1][0]

    
    input = image.float().to('cuda')
    
    model.eval()
    output = model(input)
    
    prediction = torch.argmax(output, dim=1).squeeze()


    pil_image = transforms.ToPILImage()(image.squeeze().to('cpu'))
    pil_target = transforms.ToPILImage()((target).squeeze().to('cpu'))
    pil_prediction = transforms.ToPILImage()((prediction/255).to('cpu').numpy().astype('float32'))

    return pil_image, pil_target, pil_prediction