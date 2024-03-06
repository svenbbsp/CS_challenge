from torchvision.datasets import Cityscapes

train_dataset = Cityscapes("E:\CityScapes", split='train', mode='fine',
                     target_type='semantic')

img, smnt = train_dataset[0]
