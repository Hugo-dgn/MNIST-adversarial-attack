import os

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def load():
    # Load MNIST dataset from .npy files
    train_images = np.load('dataset/train_images.npy')
    train_labels = np.load('dataset/train_labels.npy')
    test_images = np.load('dataset/test_images.npy')
    test_labels = np.load('dataset/test_labels.npy')

    return train_images, train_labels, test_images, test_labels
    

def download():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    # Convert tensors to NumPy arrays
    train_images = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()

    test_images = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Save NumPy arrays to .npy files
    np.save('dataset/train_images.npy', train_images)
    np.save('dataset/train_labels.npy', train_labels)
    np.save('dataset/test_images.npy', test_images)
    np.save('dataset/test_labels.npy', test_labels)
