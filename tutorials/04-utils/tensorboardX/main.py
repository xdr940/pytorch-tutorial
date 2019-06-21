import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import argparse
import datetime

import os
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='mnist', type=str)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.00001, type=float,help='learning rate')


parser.add_argument('--epochs', default=10, type=int)

parser.add_argument('--epoch-size', default=5, type=int)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')

parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )

parser.add_argument('--print-freq',default=200, type=int)
parser.add_argument('--log-freq',default=1, type=int)#log every batch
parser.add_argument('--log-save-path', default='logs', type=str,help = '训练可视化数据')

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
#training_writer = SummaryWriter(args.log_save_path)  # for tensorboard



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


#对于一个长度为n1， 均值为avg1的数列， 添加长度为n2，均值为avg2的数列后，整个数列的n和avg
#epoch步和batch步很有效果
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)



model = NeuralNet().to(device)

n_iter = 0#一共做了多少次forward pass, 作为batch-record data的横坐标
n_iter_val=0

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
assert args.solver in ['sgd','adam']
if args.solver == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(args,train_loader,model,epoch,train_writer):
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
            print('Step [{}/{}], train_batch_Loss: {:.4f}, train_batch_Acc: {:.2f}'
                  .format(batch_idx + 1, len(train_loader), loss, accuracy))



        train_writer.add_scalar('batch_data/batch_loss',loss.item(),n_iter+1)
        train_writer.add_scalar('batch_data/batch_acc',accuracy.item(),n_iter+1)




        n_iter+=1

    return epoch_losses.avg[0],epoch_acc.avg[0]# list

#@torch.no_grad()
def validate(args, val_loader, model, epoch, output_writers):
    val_epoch_losses = AverageMeter(precision=4)
    val_epoch_acc = AverageMeter(precision = 4)
    global n_iter_val


    for batch_idx,(images,labels) in enumerate(val_loader):

        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

        #forwardpass
        outputs = model(images)


        loss = criterion(outputs, labels)

        # NO Backward and optimize!!

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()




        for i in range(3):
            output_writers[i].add_image('inputs_img',images[0].reshape(1,28,28).cpu(),i)

        n_iter_val+=1



        val_epoch_losses.update(loss.item(), args.batch_size)
        val_epoch_acc.update(accuracy.item(), args.batch_size)


    return val_epoch_losses.avg[0],val_epoch_acc.avg[0]# list



def main():
# checkpoints and model_args, 标准存储checkpoint
    save_path = '{},{}epochs{},b{},lr{}'.format(
        args.solver,
        args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:#保存with时间
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset + '_checkpoints', save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# tensorboardX-writer of train,test,output
    train_writer = SummaryWriter(
        os.path.join(save_path, 'train'))  # 'KITTI_occ/05-29-11:36/flownets,adam,300epochs,epochSize1000,b8,lr0.0001'

    test_writer = SummaryWriter(os.path.join(save_path, 'test'))  #

    output_writers = []#for validate
    for i in range(3):  # for test img nums
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    #dummy_input = torch.rand(13,1,28,28)
    #with SummaryWriter(comment='MyModel') as w:
    #    w.add_graph(model,(dummy_input,))

    for epoch in range(args.epochs):
        train_loss,train_acc = train(args,train_loader,model,epoch,train_writer)
        with torch.no_grad():
            val_loss,val_acc = validate(args,train_loader,model,epoch,output_writers)
        #epoch-record data log
        train_writer.add_scalar(tag = 'epoch_data/epoch_loss',scalar_value= train_loss,global_step=epoch)
        train_writer.add_scalar(tag = 'epoch_data/epoch_acc',scalar_value= train_acc,global_step=epoch)

        test_writer.add_scalar(tag = 'epoch_data/epoch_loss',scalar_value= val_loss,global_step=epoch)
        test_writer.add_scalar(tag = 'epoch_data/epoch_acc',scalar_value= val_acc,global_step=epoch)
        '''
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            training_writer.add_histogram(tag, value.data.cpu().numpy(), n_iter + 1)
            training_writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(),n_iter + 1)
        '''
        # 3. Log training images (image summary)



        #print epoch
        print('epoch [{}/{}]\n'
              'train: avg_loss: {:.4f}, avg_acc: {:.2f}'
              '\nvalidate: avg_loss: {:.4f},avg_acc:{:.2f}'
              .format(epoch + 1,args.epochs,
                      train_loss, train_acc,
                      val_loss, val_acc))

        #model save
        torch.save(model,'model.pth.rar')



        #csv epoch data record
        '''
        with open(args.save_path / args.log_summary, 'a') as csvfile:  # 每个epoch留下结果
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])  # 第二个就是validataion 中的epoch-record
            # loss<class 'list'>: ['Total loss', 'Photo loss', 'Exp loss']
        '''
    train_writer.close()
    test_writer.close()
    for i in output_writers:
        i.close()
    return 0




if __name__ =="__main__":
    main()
   # print(len(train_loader))
   # print(len(val_loader))

