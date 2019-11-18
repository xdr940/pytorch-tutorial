import torch
from torch.autograd import Variable
import torch.nn as nn
def scalar_linear():
    '''
    scalar grad
    :return:
    '''
    # Create tensors.
    x = torch.tensor(1.2, requires_grad=True)
    w = torch.tensor(2.4, requires_grad=True)
    b = torch.tensor(3.1, requires_grad=True)

    # Build a computational graph.
    a = w * x
    y = a +b

    #no-leaf node
    a.retain_grad()
    y.retain_grad()
    # Compute gradients.
    y.backward()

    # Print out the gradients.

    #leaf
    print('x.grad ',x.grad)
    print('w.grad ',w.grad)
    print('b.grad ',b.grad)
    #no-leaf
    print('a.grad ',a.grad)    #None 非叶子节点(计算得出而非从disk or user得来)
    print('y.grad ',y.grad)
    '''
    out
    
    x.grad  tensor(2.4000)
    w.grad  tensor(1.2000)
    b.grad  tensor(1.)
    a.grad  tensor(1.)
    y.grad  tensor(1.)
    '''



def order1_hadamad():

    #leaf
    x = torch.tensor([1.,2,3,4],requires_grad=True)
    w = torch.tensor([2.,3,4,5],requires_grad=True)
    b = torch.tensor([ 3, 4, 5, 6],
                     dtype=torch.float,requires_grad=True)

    # non-leaf
    a=w*x

    #root
    y=a+b

    #bkwd
    a.retain_grad()
    y.retain_grad()

    y[0].backward()

    #leaf
    print('x.grad ',x.grad)
    print('w.grad ',w.grad)
    print('b.grad ',b.grad)

    #non-leaf
    print('a.grad ',a.grad)

    #root
    print('y.grad ',y.grad)



    '''
x.grad  tensor([2., 0., 0., 0.])
w.grad  tensor([1., 0., 0., 0.])
b.grad  tensor([1., 0., 0., 0.])
a.grad  tensor([1., 0., 0., 0.])
y.grad  tensor([1., 0., 0., 0.])
    '''





def transpose_grad_test():
    a = torch.tensor([1,2.,3,4],requires_grad=True).reshape(1,4)
    b = a.transpose(1,0)
    a.retain_grad()
    b.retain_grad()#shape = [4,1]
    #b.sum().backward()
    #b[0][0].backward()#[1,0,0,0].reshape(4,1)
    b[1][0].backward()#[0,1,0,0].reshape(4,1)
    #b[2][0].backward()#[0,1,0,0].reshape(4,1)
    #b[3][0].backward()#[0,1,0,0].reshape(4,1)
    print(b.grad)

    '''
    b.grad
    tensor([[1.],
        [1.],
        [1.],
        [1.]])
        
    tensor([[1.],
        [0.],
        [0.],
        [0.]])
        
    tensor([[0.],
        [1.],
        [0.],
        [0.]])
    
    tensor([[0.],
        [0.],
        [1.],
        [0.]])
    
    tensor([[0.],
        [0.],
        [0.],
        [1.]])
    '''
    print(a.grad)
    '''
        a.grad
        
        tensor([[1., 0., 0., 0.]])
        tensor([[0., 1., 0., 0.]])
        tensor([[0., 0., 1., 0.]])
        tensor([[0., 0., 0., 1.]])
    
    '''


def func21(bwd=22):
    #order2 - MmBackward
    A = torch.tensor([1.,2,3,4,5,6],requires_grad=True).reshape(2,3)
    B = torch.tensor([1.,2,
                      3,4,
                      5,6],requires_grad=True).reshape(3,2)
    A.requires_grad_()
    B.requires_grad_()


    D = torch.tensor([1.,2,3,4],requires_grad=True).reshape(2,2)

    C = A @ B
    E = C + D


    A.retain_grad()
    B.retain_grad()
    C.retain_grad()
    D.retain_grad()
    E.retain_grad()


    #z.retain_grad()

    if bwd ==11:
        E[0][0].backward()
    if bwd ==12:
        E[0][1].backward()
    elif bwd ==21:
        E[1][0].backward()
    elif bwd == 22:
        E[1][1].backward()
    elif bwd ==12:
        E[1][1].backward()

        mat = torch.ones(2,2)
        E.backward(mat)
    elif bwd ==3:
        mat = torch.tensor([1,0,
                        1,1],dtype=torch.float32).reshape(2,2)
        E.backward(mat)



    print('A.grad ',A.grad)
    print('B.grad ',B.grad)
    print('D.grad ',D.grad)

    print('C.grad ',C.grad)
    print('E.grad ',E.grad)
    '''
E[0][0].backward
    
A.grad  tensor([[1., 3., 5.],
        [0., 0., 0.]])
B.grad  tensor([[1., 0.],
        [2., 0.],
        [3., 0.]])

D.grad  tensor([[1., 0.],
        [0., 0.]])

C.grad  tensor([[1., 0.],
        [0., 0.]])

E.grad  tensor([[1., 0.],
        [0., 0.]])


#E[0][1].backward

A.grad  tensor([[2., 4., 6.],
        [0., 0., 0.]])
B.grad  tensor([[0., 1.],
        [0., 2.],
        [0., 3.]])
D.grad  tensor([[0., 1.],
        [0., 0.]])
C.grad  tensor([[0., 1.],
        [0., 0.]])
E.grad  tensor([[0., 1.],
        [0., 0.]])
        
#E[1][0].backward
    
A.grad  tensor([[0., 0., 0.],
        [1., 3., 5.]])
B.grad  tensor([[4., 0.],
        [5., 0.],
        [6., 0.]])
D.grad  tensor([[0., 0.],
        [1., 0.]])
C.grad  tensor([[0., 0.],
        [1., 0.]])
E.grad  tensor([[0., 0.],
        [1., 0.]])   

#E[1][1].BACKWARD

A.grad  tensor([[0., 0., 0.],
        [2., 4., 6.]])
B.grad  tensor([[0., 4.],
        [0., 5.],
        [0., 6.]])
D.grad  tensor([[0., 0.],
        [0., 1.]])
C.grad  tensor([[0., 0.],
        [0., 1.]])
E.grad  tensor([[0., 0.],
        [0., 1.]])
        
    '''

def func22():
    #hadamard
    x = torch.tensor([1.,2,3,4,5,6],requires_grad=True).reshape(2,3)
    y = torch.tensor([1.,2,3,4,5,6],requires_grad=True).reshape(2,3)
    b = torch.tensor([1.,2,3,4,5,6],requires_grad=True).reshape(2,3)

    z = x*y+b
   #x.retain_grad()
    z.backward(torch.ones(2,3),retain_graph=True)
    #z[0][0].backward()



    print('x.grad ',x.grad)
    print('y.grad ',y.grad)

    print('b.grad ',b.grad)
    print('z.grad ',z.grad)


    #print(z[0][0].grad)

    #print(z)

def hist_grad():


    x = torch.tensor([1.,3,4,3,4,2,3,1,2,3,9,7,0,8,6,5,8,9,4,7]).cuda()
    x.requires_grad = True
    y = torch.histc(x,bins=10,min=0,max=9)
    print(y)
    x.retain_grad()
    y.retain_grad()

    y[0].backward()

    #print(x.grad)  # the derivative for 'histc' is not implemented
    print(y.grad)

    print(y)

def branch_grad():
    x1 = torch.tensor([1.,2.,3.],requires_grad=True)
    x2 = torch.tensor([4.,5.,6.],requires_grad=True)
    w1 = torch.tensor([7.,8.,9.],requires_grad=True)
    w2 = torch.tensor([10.,11.,12.],requires_grad=True)

    y1 = x1*w1
    y2 = x2*w2

    z1 = y1*y2
    z2 = y1+y2
    z = z1+z2

    x1.retain_grad()
    x2.retain_grad()
    y1.retain_grad()
    y2.retain_grad()
    z1.retain_grad()
    z2.retain_grad()
    z.sum().backward()
    #print(z1.grad)
    print(y1.grad)# = dz1/dy1 + dz2/dy1 = [40,55,72] + [1,1,1] = (41,56,73)
    #print(z2.grad)
    print(z)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer_one = nn.Sequential(
            #nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
           # nn.BatchNorm2d(16),
           # nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=1),
            #nn.AvgPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_two = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer_one(x)
        #out = self.layer_two(out)
       # out = out.reshape(out.size(0), -1)
       # out = self.fc(out)
        return out

def pooling_grad():
    I = torch.tensor([12,42,1,24,
                      52,63,3,8,
                      9,11,21,12,
                      1,14,15,16.],requires_grad=True).reshape(1,4,4).cuda()
    I2 = torch.tensor([1,2,3,4,
                      5,3,7,8,
                      9,10,11,12,
                      13,14,15,16.],requires_grad=True).reshape(1,4,4).cuda()
    torch.backends.cudnn.deterministic = True
    net = ConvNet()
    out =   net(I2)
    out.retain_grad()
    #out.sum().backward()


    I2.retain_grad()


#    out[0][1][1].backward()
    out.sum().backward()


   # print(out)
  #  print(out.grad)
    print(I2.grad)




def main():
    pooling_grad()

if __name__ == "__main__":
    hist_grad()
