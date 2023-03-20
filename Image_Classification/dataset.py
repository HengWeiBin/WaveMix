from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
from scipy.ndimage import rotate

class NtutEMNIST(Dataset):
    def __init__(self, x, y=None, data_aug=False, aug_factor=0.5):
        self.x = x
        self.y = y
        self.data_aug = data_aug
        self.aug_factor = aug_factor

        if y is not None:
            assert x.shape[0] == y.shape[0], "Labels shape should be same with Images set"

        if self.data_aug:
            self.x = np.concatenate([self.x, self.x], axis=0)
            if y is not None:
                self.y = np.concatenate([self.y, self.y], axis=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img = self.x[index].copy()

        # Data Augmentation
        if self.data_aug and (index > len(self) / 2):
            if random.random() < self.aug_factor:
                self.translate(img, shift=5)
            if random.random() < self.aug_factor:
                self.random_crop(img, crop_size=(23, 23))
            if random.random() < self.aug_factor:
                self.rotate_img(img, 10, bg_patch=(5,5))
            if random.random() < self.aug_factor:
                self.gaussian_noise(img, mean=0, sigma=0.03)
            if random.random() < self.aug_factor:
                self.brightness(img, factor=0.8)
            if random.random() < self.aug_factor:
                self.contrast(img, factor=0.8)
        
        img = np.concatenate((img, img,  img), 2)           # Expand 1 channel to 3 channels
        img = img.transpose(2, 0, 1)                        # [w, h, c] -> [c, w, h]
        img_tensor = torch.tensor(img)    # Normalize
        if self.y is None: # Test mode
            return img_tensor
        else:
            label_tensor = torch.tensor(self.y[index], dtype=torch.int)
            return img_tensor, label_tensor

    def translate(self, origin_img, shift=10, direction=None, roll=True):
        assert direction in ['right', 'left', 'down', 'up', None], 'Directions should be top|up|left|right'
        if direction is None:
            directions = ['right', 'left', 'down', 'up']
            direction = directions[random.randint(0, 3)]
        img = origin_img.copy()
        if direction == 'right':
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:,:shift] = np.fliplr(right_slice)
        if direction == 'left':
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        if direction == 'down':
            down_slice = img[-shift:, :].copy()
            img[shift:, :] = img[:-shift,:]
            if roll:
                img[:shift, :] = down_slice
        if direction == 'up':
            upper_slice = img[:shift, :].copy()
            img[:-shift, :] = img[shift:, :]
            if roll:
                img[-shift:,:] = upper_slice
        origin_img[:] = img

    def random_crop(self, origin_img, crop_size=(10, 10)):
        assert crop_size[0] <= origin_img.shape[0] and crop_size[1] <= origin_img.shape[1], "Crop size should be less than image size"
        origin_size = origin_img.shape
        img = origin_img.copy()
        w, h = img.shape[:2]
        x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
        img = img[y:y+crop_size[0], x:x+crop_size[1]]
        origin_img[:] = cv2.resize(img, origin_size[:2]).reshape(origin_size)

    def rotate_img(self, origin_img, angle, bg_patch=(5,5)):
        assert len(origin_img.shape) <= 3, "Incorrect image shape"
        rgb = len(origin_img.shape) == 3
        img = origin_img.copy()
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
        img = rotate(img, angle, reshape=False)
        mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
        img[mask] = bg_color
        origin_img[:] = img

    def gaussian_noise(self, origin_img, mean=0, sigma=0.03):
        img = origin_img.copy()
        noise = np.random.normal(mean, sigma, img.shape)
        mask_overflow_upper = img+noise >= 1.0
        mask_overflow_lower = img+noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        img += noise
        origin_img[:] = img

    def brightness(self, img, factor=0.8):
        img[:] = img[:] * (factor + np.random.uniform(0, 0.4)) #scale channel V uniformly
        img[:][img > 1] = 1 #reset out of range values
        
    def contrast(self, img, factor=0.8):
        mean = np.mean(img)
        img[:] = (img - mean) * (factor + np.random.uniform(0, 0.4)) + mean

# test code
if __name__=="__main__":
    img_root = "emnist-byclass-train.npz"
    
    data = np.load(img_root)
    train_labels = data['training_labels']
    train_images = data['training_images']

    trn_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    mean = np.mean(trn_images, axis=(1,2), keepdims=True)
    std = np.std(trn_images, axis=(1,2), keepdims=True)
    trn_images = (trn_images - mean) / std

    val_size = int(train_images.shape[0] * 0.1)
    x_val = trn_images[:val_size]
    y_val = train_labels[:val_size]
    x_train = trn_images[val_size:]
    y_train = train_labels[val_size:]

    dataset = NtutEMNIST(x_train, y_train)
    for i, (img, label) in enumerate(dataset):
        print(img.shape, label.shape)
        break