
import torch


a = torch.ones(3,2,5,6)
b = torch.ones(3,1,5,6)
c = torch.tensor([1,2,3,4,5,6,7,8,9]).reshape(3,3)
print(c[:,1:2])
