
#添加fc层



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet18
import numpy as np
import os
import sys



# 超参数设置
EPOCH = 135   #遍历数据集次数
pre_epoch = 12  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.1        #学习率
TEST_BATCH_SIZE = 50
PRINT_FREQ = 100
# 准备数据集并预处理

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#windows
#trainset = torchvision.datasets.CIFAR10(root='F:/others/datasets/cifar-10-python', train=True, download=False, transform=transform_train) #训练数据集
#testset = torchvision.datasets.CIFAR10(root='F:/others/datasets/cifar-10-python', train=False, download=False, transform=transform_test)

#linux
trainset = torchvision.datasets.CIFAR10(root='/home/roit/datasets/cifar-10-python', train=True, download=False, transform=transform_train) #训练数据集
testset = torchvision.datasets.CIFAR10(root='/home/roit/datasets/cifar-10-python', train=False, download=False, transform=transform_test)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        BackBone = resnet18()
        BackBone.load_state_dict(torch.load(BACK_BONE_PATH))

        add_block = []
        add_block +=[nn.Linear(1000,10)]#迁移训练
        add_block += [nn.LeakyReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        #add_block.apply(weights_init_xavier)
        self.BackBone = BackBone
        self.add_block = add_block
    def forward(self, input):   # input is 1000!
 
        x = self.BackBone(input)
        y = self.add_block(x)
        return y




#windows
#MODEL_SAVED_PATH = 'F:/others/models/torchvision/img2cifar_res18.pth'
#BACK_BONE_PATH = 'F:/others/models/torchvision/resnet18-5c106cde.pth'   

#ubuntu
MODEL_SAVED_PATH = '/home/roit/models/torchvision/img2cifar10_res18.pth'
BACK_BONE_PATH = '/home/roit/models/torchvision/official/resnet18-5c106cde.pth'#pretrained by imgnt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(MODEL_SAVED_PATH)==True:
    model = torch.load(MODEL_SAVED_PATH).to(device)
else:
    model = MyModel().to(device)


para = sum([np.prod(list(p.size())) for p in model.parameters()])
type_size=torch.FloatTensor().element_size()#返回单个元素的字节大小. 这里32bit 4B
print('Model {} : param_num: {:4f}M  param_size: {:4f}MB '.format(model._get_name(), para  / 1000 / 1000,para *type_size / 1000 / 1000))
  



if __name__ == '__main__':

#def train():
   
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("log.txt", "w")as f2:
        try:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                model.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader,start=0):
                    # 准备数据
                    length = len(trainloader)#如果batch_size==1 拿了length= 50000； 如果batch_size == 250 ，则length== 200
                    inputs, labels = data#inputs.shape = data[0].shape = [batch_size,3,32,32],labels.shape = 16
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    
                    # forward + backward
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    #predicted = torch.max(outputs.data, 1)
                    predicted=torch.max(outputs,dim=1)[1]
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()

                    if i%PRINT_FREQ ==0:

                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
        except KeyboardInterrupt:
            torch.save(model,MODEL_SAVED_PATH)
            print('KeyboardInterrupt, model saved successfully')






