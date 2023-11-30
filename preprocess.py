import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import numpy as np

def normalize(images):
    
    min_values = torch.amin(images, dim=(2, 3), keepdim=True)
    max_values = torch.amax(images, dim=(2, 3), keepdim=True)

    normalized_image = (images - min_values) / (max_values - min_values + 1e-6)
    return normalized_image

def rotate(tensor_images):
    rotate_images = transforms.RandomRotation(30)(tensor_images)
    return rotate_images

def translate(tensor_image):
    translate_x = np.random.normal(0, 3)
    translate_y = np.random.normal(0, 3)
    translated_image = F.affine(tensor_image, angle=0, translate=[translate_x, translate_y], scale=1, shear=0)
    return translated_image