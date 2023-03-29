#import necessary libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
from sklearn.model_selection import train_test_split

from prepare_dataset import prepare_dataset
from dataloader import LemonDataset
from classifier import CNN
from evaluate import list2str, plot_curve

#check gpu availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#negative anf positive images folder path.
all_image_path = "drive/MyDrive/lemon_quality_dataset/all_dataset/" #--should be edited--
num_fold = 5 #--should be edited--
good_quality_fold, bad_quality_fold = prepare_dataset(all_image_path, num_fold).train_test_set()

#add augmentation
train_transforms = A.Compose(
    [A.RandomBrightnessContrast(p=0.5),
    A.Resize(250,250),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #--should be edited--
    ToTensorV2()])

test_transforms = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #--should be edited--
    A.Resize(250,250),
    ToTensorV2()
    ]
)

#start training
fold_training_loss = []
fold_test_loss = []

for each_fold in range(num_fold):

  #create train(4 folds)-test(1 fold)
  test_image_paths = good_quality_fold[each_fold] + bad_quality_fold[each_fold]
  train_image_paths = list2str([each for index, each in enumerate(good_quality_fold) if index!=each_fold]+[each for index, each in enumerate(bad_quality_fold) if index!=each_fold])

  #shuffle training images
  random.shuffle(train_image_paths)

  #create dataset 
  train_dataset = LemonDataset(train_image_paths, transform=train_transforms)
  test_dataset = LemonDataset(test_image_paths, transform=test_transforms)

  #create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  #call for the model
  model = CNN().to(device)

  #define loss function
  loss_func = nn.BCELoss()
  params = model.parameters()
  optimizer = torch.optim.SGD(params, lr=0.001)
  batch_size = 64
  train_losses = []
  test_losses = []

  #train the model
  num_epochs = 25
  loss_history = []
  print(f"***************{each_fold+1}/{num_fold} step is training...***************")
  
  for epoch in range(num_epochs):

    #training step
    model.train()
    training_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):

      images = images.to(device)
      labels = torch.from_numpy(np.array(labels)).to(torch.float32).to(device)

      optimizer.zero_grad()
      output = model(images)
      loss_train = loss_func(output, labels.unsqueeze(1))
      loss_train.backward()
      optimizer.step()

      if (i+1)%1 == 0:
        print('Epoch [%d/%d], Step [%d/%d], Loss: %4f'
                %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss_train.item()))
      
      #sum training losses coming from batches
      training_loss += loss_train.item()

    #store training losses for each epoch
    train_losses.append(training_loss)
      
    #test step
    model.eval() # important especially for batch normalization layers
    testing_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):

      images = images.to(device)
      labels = torch.from_numpy(np.array(labels)).to(torch.float32).to(device)

      optimizer.zero_grad()
      output = model(images)
      loss = loss_func(output, labels.unsqueeze(1))
      #sum test losses coming from batches
      testing_loss += loss.item()

    #store testing losses for each epoch
    test_losses.append(testing_loss)
  fold_training_loss.append([train_losses])
  fold_test_loss.append([test_losses])

#plot train loss curve
plot_curve(train_losses, num_epochs)




