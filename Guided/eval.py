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

def showImageTargetAndPrediction(data_loader, model):
    image, target = next(iter(data_loader))
    image = image.float().to('cuda')

    model.eval()
    prediction = torch.argmax(model(image).squeeze(), dim=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

    pil_image = transforms.ToPILImage()(image.squeeze().to('cpu'))
    pil_target = transforms.ToPILImage()((target).squeeze().to('cpu'))
    pil_prediction = transforms.ToPILImage()((prediction/255).to('cpu').numpy().astype('float32'))
    
    
    
    ax1.imshow(pil_image)
    ax1.set_title('Image')

    ax2.imshow(pil_target)
    ax2.set_title('Target')

    ax3.imshow(pil_prediction)
    
    plt.show()

def showPixelValues(data_loader, model):
    image, target = next(iter(data_loader))
    image = image.float().to('cuda')
    model.eval()
    prediction = torch.argmax(model(image).squeeze(), dim=0)

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12,6))
    
    t_array = (target).squeeze().to('cpu').numpy().astype('float32').flatten()
    print(np.shape(t_array))

    ax2.boxplot(t_array)
    ax2.set_title('Target')

    ax3.boxplot((prediction).squeeze().to('cpu').numpy().astype('float32').flatten())
    ax3.set_title('Prediction')

    plt.show()
