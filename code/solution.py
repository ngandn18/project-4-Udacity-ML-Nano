#TODO: Import your dependencies.
#For instance, below are some dependencies you might need 
# if you are using Pytorch

# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.optim import lr_scheduler

# import argparse
import os
import time
import copy
import sys
# import io

# from PIL import ImageFile, Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


# import smdebug.pytorch as smd
# from smdebug.core.modes import ModeKeys

# from smdebug import modes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_classes = len(class_names)
    model = models.resnet34(pretrained=True)
    num_classes = 4
    # num_classes = 133
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    return model

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          
    '''   
    model.eval()   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    running_loss=0
    running_corrects=0
    # dt_sizes = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # dt_sizes += inputs.size(0)

    # test_loss = running_loss / dt_sizes
    # test_acc = running_corrects.double() / dt_sizes
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss}, Test Accu: {test_acc}')
   
    # logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(test_loss, test_acc))

# def train(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes):
def train(model, dataloaders, criterion, optimizer, scheduler):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 4


    # Training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).requires_grad_()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # logger.info('\nEpoch: {} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            # logger.info('\nBest Accuracy: {}\n'.format(best_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model    


def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # logger.info("\nbatch_size: {}\n".format(batch_size))

    mean=[0.48479516, 0.45410065, 0.39083833]
    std=[0.26203398, 0.2552936,  0.2579632]

    torch.manual_seed(18)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    data_transforms['valid'] = data_transforms['test']

    image_datasets = {x: ImageFolder(
        os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'test', 'valid']}
  
    loaders = {x: DataLoader(image_datasets[x], 
        batch_size=batch_size,
        shuffle=True, num_workers=0) 
        for x in ['train', 'test', 'valid']}
 
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}
    class_names = image_datasets['train'].classes

    # print(f'Datasets sizes: {dataset_sizes}\n')
    print('Finish loading data\n')
    # return loaders, dataset_sizes, class_names
    return loaders, class_names


def main():

    '''    TODO: Creat data     '''
    data_dir = 'dogImages'
    batch_size = 2
    # batch_size = 64
    loaders, class_names = create_data_loaders(data_dir, batch_size)
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model=train(model, loaders, loss_criterion, optimizer, scheduler)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    # test(model, test_loader, loss_criterion)
    # test(model, loaders['test'], loss_criterion)    
    '''
    TODO: Save the trained model
    '''
#     with open(os.path.join(args.model_dir, 'model.pt'), 'wb') as f:
#         torch.save(model.state_dict(), f)
    # model_dir = 'TrainedModels'

    torch.save(model.state_dict(), "TrainedModels/model.pth")
    print('saved')

if __name__=='__main__':
    main()
