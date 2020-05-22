from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from path import Path
class ToyDataset(Dataset):
    def __init__(self,root=None):
        self.root = root
        pass
    def __len__(self):
        return len(self.root.files())
    def __getitem__(self, index):
        ret = np.load(self.root/'{:03d}.npy'.format(index))
        train = ret[:4]
        val = ret[-1]
        return torch.tensor(train), torch.tensor(val)

if __name__ == "__main__":
    root = Path('/home/roit/aws/tutorial_pt/tutorials/02-intermediate/mono_works')/'data'
    train_data = ToyDataset(root)
    train_loader = DataLoader(dataset=train_data,batch_size=4)

    for x,y in train_loader:
        print('ok')


    print('ok')