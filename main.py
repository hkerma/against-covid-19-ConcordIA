from dataset.dataset import CovidDataset, CompetitionDataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image
from toolsp.train_test import *
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import pandas as pd
from toolsp.equalization import equalization
from torchvision.models import *

SHAPE = (240, 240)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


preprocess_train = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize(SHAPE),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(SHAPE, scale=(0.80, 1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=1.3))], p=0.3),
    transforms.Lambda(lambda x:equalization(x)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training
root_dir = "../dataset/train/"
csv_file = pd.read_csv("dataset/train.csv")
balance = True
train_dataset = CovidDataset(csv_file=csv_file, phase='train', root_dir=root_dir,
                             device=device, transform=preprocess_train, balance=balance, nb_images=256)

print("Train dataset length:", train_dataset.__len__())


preprocess_test = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize(SHAPE),
    transforms.Lambda(lambda x:equalization(x)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

# Validation
root_dir = "../dataset/test/"
csv_file = pd.read_csv("dataset/test.csv")
balance = True

test_dataset = CovidDataset(csv_file=csv_file, phase='test', root_dir=root_dir,
                            device=device, transform=preprocess_test,
                            balance=balance, nb_images=None)

print("Validation set length:", test_dataset.__len__())


# Test on competition set
competition = CompetitionDataset("../dataset/competition_test/", preprocess_test, device)


# Training hyperparameters
initial_lr = 0.001
num_epochs = 60
step_size = int(num_epochs/4)
batch_size = 32

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
competition_loader = DataLoader(competition, batch_size=40, shuffle=False)

# Model definition
device = torch.device(device)
model = resnext50_32x4d(num_classes=2)
model.to(device)

# Training stuff
criterion = nn.CrossEntropyLoss()
weights = [5.0,6.0]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, verbose=True)

# Train
train(model, train_loader, test_loader, device, optimizer, criterion, scheduler, batch_size, num_epochs, competition_loader)
