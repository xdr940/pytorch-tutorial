
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#define as np
x_train_np=np.array([[3.3], [4.4 ], [5.5 ], [6.71], [6.93], 
[4.168], [9.779 ], [6.182], [7.59], [2.167], [7.042], [10.791] , 
[5.313 ] , [7.997], [3.1]],dtype=np.float32)


y_train_np=np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
[3.366], [2.596], [2.53], [1.221], [2.827],
[3.465], [1.65], [2.904], [1.3]],dtype=np.float32)


#np 2 tensor_cpu
x_train_cpu=torch.from_numpy(x_train_np)
y_train_cpu=torch.from_numpy(y_train_np)

#tensor_cpu 2 tensor_cuda
x_train_cuda=x_train_cpu.cuda()
y_train_cuda=y_train_cpu.cuda()

#model define
class LR (nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_features=1,out_features=1,bias = True)
    def forward(self, x):
        out = self.linear(x)
        return out


#train
def training(inputs,target):
    
    num_epochs = 1000

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    
    for epoch in range(num_epochs):
    
        '''if torch.cuda.is_available():
            inputs = Variable(x_train_cuda)
            target = Variable(y_train_cuda)
        else:
            inputs =Variable(x_train)
            target = Variable(y_train)'''
     
        #forward
        out = model(inputs)#equal to model(inputs)
        loss = criterion(out, target)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1)%50==0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1,num_epochs, loss.item()))


if __name__=='__main__':

    if torch.cuda.is_available():
        model = LR().cuda()
    else: 
        model = LR()
    
    model.train()#训练前要运行一次
    training(x_train_cuda,y_train_cuda)
    
    model.eval()

    predict = model(Variable(x_train_cuda))
    predict = predict.cpu().data.numpy()
    
    print(predict)
    print(y_train_np)

    for param in model.parameters():
        print(type(param.data), param.data,param.size())
    plt.plot(x_train_np,y_train_np,'ro',label = 'original data')
    plt.plot(x_train_np,predict,label = 'Fitting Line')
    plt.show()



