import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataloader import DataLoader
from torchvision.datasets import MNIST#construction function
from mnist import MNIST
from convnet import ConvNet
from iutils import AverageMeter
import numpy as np
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 64
learning_rate = 0.001

#others
print_freq=50
n_iter=0
model_args_saved_path='arg_model2.pth'
optimizer_args_saved_path="adam.pth"
# MNIST dataset
train_dataset = MNIST(root='/home/roit/datasets/mnist/',#要有raw and processed data
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = MNIST(root='/home/roit/datasets/mnist/',
                                          train=False, 
                                          transform=transforms.ToTensor())

print(train_dataset)
# Data loader
train_loader = DataLoader(dataset=train_dataset,num_workers=4,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


def main():
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        avg_loss = train(model,train_loader,optimizer,criterion)
        avg_acc = validate(model,test_loader)
        print('\nEpoch[{}/{}]\n avg_loss:{:.4f} avg_acc:{}'.format(epoch+1,num_epochs,avg_loss,avg_acc))

    # Save the model checkpoint
    torch.save(model.state_dict(), model_args_saved_path)
    torch.save(optimizer.state_dict(),)


def train(model,train_loader,optimizer,criterion):
    batch_loss =AverageMeter()
    model.train()
    batch_nums = len(train_loader)
    for batch_i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)


        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.update(loss.item(),batch_size)#虽然最后一个batch大小不是batch_size，但无妨
        if (batch_i + 1) % print_freq == 0:
            print('Step [{}/{}], Loss: {:.4f}'
                  .format(batch_i + 1,batch_nums , loss.item()))

    return batch_loss.avg



def validate(model,test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # acc
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print('Test Accuracy of the model on the 10000 test images: {:.4f} %'.format(acc))

    return acc


# Test the model
if __name__ =="__main__":
    main()


