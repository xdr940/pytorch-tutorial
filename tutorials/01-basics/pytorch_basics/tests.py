

import torch
from  time import  time
def main1():
    a = torch.ones([4,1,800,600])


    b = torch.ones([100,800,1])



    print((a-b).shape)
    print((b-a).shape)



def main2():
    a = torch.tensor([2,3,4])

    ls = []
    i =0
    while i<3:
        ls.append(a)
        i+=1
    a = a.unsqueeze(dim=0)
    out = torch.cat(ls,dim=0)


    print(ls)
    print(out)

def testmul():

    a = torch.ones([3,4,5])
    b = torch.ones([5,6])
    c = a@b
    print(c.shape)

    e = torch.ones(3,4,5)
    f = torch.ones(2,4)
    g = f@e
    print(g.shape)


def testif():

    i=0
    a='asdgs'
    list_time =[]
    while i<100:
       st = time()
       if a=='asdg':
           pass
       else:
           pass
       list_time.append((time()-st)*1000000)
       i+=1
    print(list_time)


testif()
