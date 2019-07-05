

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np




# 超参数设置
EPOCH = 135   #遍历数据集次数
pre_epoch = 12  # 定义已经遍历数据集的次数
BATCH_SIZE = 4      #批处理尺寸(batch_size)
LR = 0.1        #学习率
NUMS_WOKER = 4
# 准备数据集并预处理
classes = ['cardboard','glass','metal','paper','plastic','trash']

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])
train_dataset = datasets.ImageFolder(root='/home/roit/datasets/trash/train',transform=data_transform)
test_dataset = datasets.ImageFolder(root='/home/roit/datasets/trash/test',transform=data_transform)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMS_WOKER)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUMS_WOKER)


def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(4):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(classes[label_])
        ax.axis('off')
        plt.pause(0.01)


plt.figure()
for i_batch, sample_batch in enumerate(train_dataloader):
    show_batch_images(sample_batch)

    plt.show()