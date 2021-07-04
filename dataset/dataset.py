import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import random
import numpy as np
import tqdm

class CovidDataset(Dataset):

    def __init__(self, root_dir, phase, transform, csv_file, device, nb_images=1000, balance=True):

        p_images = [root_dir + x for x in csv_file.loc[:, 'filename'][csv_file.loc[:, 'class'] == 1].values.tolist()]
        n_images = [root_dir + x for x in csv_file.loc[:, 'filename'][csv_file.loc[:, 'class'] == 0].values.tolist()]

        # 'balance' the dataset
        if phase == 'train':
            if balance:
                # default ration 3/2
                if nb_images is None:
                    nb_images = np.int(len(p_images))
                    n_images = random.sample(n_images, nb_images)
                else:
                    n_images = random.sample(n_images, nb_images)
                    p_images = random.sample(p_images, nb_images)

        images = p_images + n_images
        self.labels = [1]*len(p_images) + [0]*len(n_images)
        
        self.images = images

        self.transform = transform

        self.device = device


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image, PIL format
        image = Image.open(self.images[idx])

        # print("Path: ",self.images[idx])
        # print("Label: ", self.labels[idx])
        # Apply transformation
        image = self.transform(image)

        image = image.to(self.device)
        label = torch.tensor(self.labels[idx])
        label = label.to(self.device)

        return image, label



class CompetitionDataset(Dataset):

    def __init__(self, root_dir, transform, device, PATH_TO_IMAGES = None):

        if PATH_TO_IMAGES == None:
            images = [root_dir + str(i) + '.png' for i in range(1, 401)]
        else:
        	images = PATH_TO_IMAGES

    
        self.transform = transform
        self.device = device
        self.images = images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image, PIL format
        image = Image.open(self.images[idx])

        # print("Path: ",self.images[idx])
        # print("Label: ", self.labels[idx])
        # Apply transformation
        image = self.transform(image)

        image = image.to(self.device)

        return image
