
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.optim import lr_scheduler

import argparse
import os
import time
import sys
# import io


import copy
import argparse
import logging
# from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from time import time

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          
    '''
    begin = time()
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
    
    # print(f'Test Loss: {test_loss: .4f}, Test Accuracy: {test_acc: .4f}')
    # print(f"\nTest running time: {time()-begin: .1f}s")

    logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {: .4f}\n".format(test_loss, test_acc))

    # logger.info("\nTest running time: {:.1f}".format(time()-begin))
    test_time = time()-begin
    logger.info("Test running time: {:.1f}s".format(test_time))

    
# def train(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes):
def train(model, dataloaders, criterion, optimizer, scheduler):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 4    # local test # 10


    # Training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

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

            # more exact size of dataset
            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            logger.info('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            # logger.info('\nBest Accuracy: {}\n'.format(best_acc))
            

        # print()

    time_elapsed = time() - start_time
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print('\n---\nTraining complete in {:.0f}m {:.0f}s'.format(minutes, seconds))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model    


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_classes = len(class_names)
    model = models.resnet34(pretrained=True)
    num_classes = 133
    # num_classes = 4 # local test
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    return model


def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("\nbatch_size: {}".format(batch_size))

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

    print('Finish loading data: \n',dataset_sizes)
    # return loaders, daset_sizes, class_names

    return loaders, class_names


def main(args):

    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Creat data
    '''
    loaders, class_names = create_data_loaders(args.data_dir, args.batch_size)
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # criterion = nn.CrossEntropyLoss(ignore_index=133)
    # optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    logger.info("Starting Model Training")

    '''
    TODO: Create your loss and optimizer
    '''
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, loaders, loss_criterion, optimizer, scheduler)

    # model=train(model, train_loader, validation_loader, criterion, optimizer)
   
    logger.info("Testing Model")
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, loaders['test'], loss_criterion)    
    # test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
#     with open(os.path.join(args.model_dir, 'model.pt'), 'wb') as f:
#         torch.save(model.state_dict(), f)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
    logger.info("Saved Model")


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, metavar="N",
        help="batch_size for training (default: 2)")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", 
        help="learning rate (default: 0.001)")

    parser.add_argument('--learning_rate', type=float, default=0.001, metavar="LR", 
        help="learning rate (default: 0.001)")
    
    # Local test
    # parser.add_argument('--data_dir', type=str, default='dogImages')
    # parser.add_argument('--model_dir', type=str, default='model')
    # parser.add_argument('--output_dir', type=str, default='output')
    
    # Sagemanker Studio
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    #local test 
    # print(args)
    
    main(args)
