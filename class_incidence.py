from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Cityscapes
import numpy as np
import csv
import torch
import utils

subsize = (256, 512)


normal_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(subsize),
])
normal_target_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(subsize),
])

dataset = Cityscapes(root="E:\CityScapes", split='train', mode='fine', target_type='semantic', transform=normal_transform, target_transform=normal_target_transform)
#dataset, _ = random_split(dataset, [0.01,0.99])


class_pixel_counts = {}
speed = 1

size = len(dataset)

for i in range(size):
    target = utils.map_id_to_train_id(dataset[i][1]).numpy()
    
    unique_classes, class_counts = np.unique(target[target != 255], return_counts=True)

    for class_label, count in zip(unique_classes, class_counts):
        if class_label not in class_pixel_counts:
            class_pixel_counts[class_label] = count
        else:
            class_pixel_counts[class_label] += count

    print(f'[{i}/{size}]')


print("Class Pixel Counts:", class_pixel_counts)

# Specify the output CSV file path
csv_file_path = "class_pixel_counts.csv"

# Write the dictionary to a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Class', 'Pixel_Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for class_label, count in class_pixel_counts.items():
        writer.writerow({'Class': class_label, 'Pixel_Count': count})

print("CSV file saved successfully.")


# Compute total number of pixels
total_pixels = sum(class_pixel_counts.values())

# Compute class weights
class_weights = {}
for class_name, count in class_pixel_counts.items():
    class_weights[class_name] = total_pixels / (count * len(class_counts))

# Normalize weights
total_weight = sum(class_weights.values())
normalized_weights = {class_name: weight / total_weight for class_name, weight in class_weights.items()}


weights = list(normalized_weights.values())
weights_tensor = torch.tensor(weights)

print("Normalized Weights:", weights_tensor)
