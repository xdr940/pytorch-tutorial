import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from iutils import paras_info_print,AverageMeter,save_checkpoint
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm


# 超参数设置
batch_size = 32      #批处理尺寸(batch_size)
learning_rate = 0.1        #学习率
num_epochs = 100
print_freq = 100

# 准备数据集并预处理
num_wokers = 12
best_acc=-1
is_best=False
classes = ('cardboard','glass','metal','paper','plastic','trash')
nums_classes = len(classes)
model_dict_saved_path = 'model_best_acc.pth'
pretrained_dict_path = '/home/roit/models/torchvision/official/resnet18-5c106cde.pth'#pretrained by imgnt args
check_points_path = 'check_points'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差

])
train_dataset = datasets.ImageFolder(root='/home/roit/datasets/trash/train',transform=data_transform)
test_dataset = datasets.ImageFolder(root='/home/roit/datasets/trash/test',transform=data_transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_wokers)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_wokers)
# Cifar-10的标签

# 模型定义-ResNet







#windows
#MODEL_SAVED_PATH = 'F:/others/models/torchvision/img2cifar_res18.pth'
#BACK_BONE_PATH = 'F:/others/models/torchvision/resnet18-5c106cde.pth'   

#ubuntu

def make_model():#载入官方resnet18架构，并更改fc layer
    model = models.resnet18().to(device)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features,nums_classes)
    return model.to(device)


def main():
#make modle
    model=make_model()
    global  best_acc,is_best
    if os.path.exists(model_dict_saved_path) == True:
        model_dict = torch.load(model_dict_saved_path)
    else:
        print('train from imgnt pretrained resnet18 model ')
        pretrianed_dict = torch.load(pretrained_dict_path)
        model_dict = model.state_dict()
        for k in pretrianed_dict.keys():
            if 'fc' not in k:
                model_dict[k]=pretrianed_dict[k]

    model.load_state_dict(model_dict)
    paras_info_print(model)

    train_writer = SummaryWriter(check_points_path)



    print('=> will save model_args to {}'.format(check_points_path))
    if not os.path.exists(check_points_path):
        os.makedirs(check_points_path)

# Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        avg_loss = train(model,train_loader,optimizer,criterion)
        avg_acc = validate(model,test_loader)
        print('\nEpoch[{}/{}]\n avg_loss:{:.4f} avg_acc:{:.4f}'.format(epoch+1,num_epochs,avg_loss,avg_acc))
        train_writer.add_scalar('avg_loss',avg_loss,epoch)
        train_writer.add_scalar('avg_acc',avg_acc,epoch)

        if best_acc < 0:#first epoch
            best_acc = avg_acc

        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)


        save_checkpoint({  # 在模型里写超参数状态
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc
                },
                is_best,
                check_points_path)


    # Save the model checkpoint



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
        print('\nTest Accuracy of the model on the 10000 test images: {:.4f} %'.format(acc))

    return acc


# Test the model
if __name__ =="__main__":
    main()





