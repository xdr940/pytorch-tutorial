import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from logger import Logger
from logger2 import AverageMeter
import argparse

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--log-save-path', default='logs', type=str,help = '训练可视化数据')

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--print-freq',default=200, type=int)
parser.add_argument('--log-freq',default=1, type=int)#log every batch

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset 
train_dataset = MNIST(root='/home/roit/datasets/mnist/',#要有raw and processed data
                                           train=True,
                                           transform=transforms.ToTensor())

val_dataset = MNIST(root='/home/roit/datasets/mnist/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=args.batch_size,
                                          num_workers = args.num_workers,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=args.batch_size,
                                          num_workers = args.num_workers,
                                          shuffle=False)
training_writer = SummaryWriter(args.log_save_path)  # for tensorboard


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

n_iter = 0#一共做了多少次forward pass, 作为batch-record data的横坐标

logger = Logger('./logs')#self defined class

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


def train(args,train_loader,model,epoch,training_writer):
    global n_iter
    epoch_losses = AverageMeter(precision=4)#自定义类只能放到cpu内存?
    epoch_acc = AverageMeter(precision=4)

    for batch_idx,(images,labels) in enumerate(train_loader):

        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

        #forwardpass
        outputs = model(images)


        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()



        epoch_losses.update(loss.item(), args.batch_size)
        epoch_acc.update(accuracy.item(), args.batch_size)

        if (batch_idx + 1) % args.print_freq == 0:#print freq
            print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                  .format(batch_idx + 1, len(train_loader), loss, accuracy))
        #if (batch_idx + 1) % args.log_freq == 0:  # print freq

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'batch_loss': loss.item(), 'batch_accuracy': accuracy.item()}

        for tag, value in info.items():
            training_writer.add_scalar(tag,value,n_iter+1)
        # 2. Log values and gradients of the parameters (histogram summary)


        # 3. Log training images (image summary)
        #info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

       # for tag, images in info.items():
          #  training_writer.add_image(tag, images, n_iter + 1)

        n_iter+=1

    return epoch_losses.avg[0],epoch_acc.avg[0]# list


def validate(args, val_loader, model, epoch, training_writer):
    losses = AverageMeter(precision=4)

    return losses.avg[0]

def main():
    #dummy_input = torch.rand(13,1,28,28)
    #with SummaryWriter(comment='MyModel') as w:
    #    w.add_graph(model,(dummy_input,))

    for epoch in range(args.epochs):
        train_loss,train_acc = train(args,train_loader,model,epoch,training_writer)
        erros = validate(args,train_loader,model,epoch,training_writer)
        #log
        training_writer.add_scalar(tag = 'epoch_loss',scalar_value= train_loss,global_step=epoch)
        training_writer.add_scalar(tag = 'epoch_acc',scalar_value= train_acc,global_step=epoch)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            training_writer.add_histogram(tag, value.data.cpu().numpy(), n_iter + 1)
            training_writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(),n_iter + 1)
            
        #print
        print('epoch [{}/{}], avg_loss: {:.4f}, avg_acc: {:.2f}'
              .format(epoch + 1, args.epochs, train_loss, train_acc))

        #model save
        torch.save(model,'model.pth.rar')



        #csv epoch data record
        '''
        with open(args.save_path / args.log_summary, 'a') as csvfile:  # 每个epoch留下结果
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])  # 第二个就是validataion 中的epoch-record
            # loss<class 'list'>: ['Total loss', 'Photo loss', 'Exp loss']
        '''

    return 0


def save_checkpoint2(save_path,net_state,is_best,filename='checkpoint.pth.tar'):
    torch.save(net_state,save_path+'/{}_{}'.format(net_state,filename))
    if is_best:
        shutil.copyfile(save_path+'/{}_{}'.format(net_state,filename),save_path/'{}_model_best.pth.tar'.format(net_state))


if __name__ =="__main__":
    main()

